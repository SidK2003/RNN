"""
Training loop for the GRU+Attention prediction model (Stage 1).

In simple words: This is the gym where the AI works out and gets smart.
It handles several highly complex tasks automatically:
1. Walk-Forward Windows: It trains the AI on 2000-2009, tests on 2010-2011, then steps forward. This prevents the AI from cheating by seeing the future.
2. Z-Score Normalization: It converts raw stock prices/indicators into standardized scales (like bell curves) using ONLY training data, stopping "Look-Ahead Bias".
3. AMP (Automatic Mixed Precision): Uses the RTX 4070's special Tensor Cores to train twice as fast using half-precision math (FP16).
4. Early Stopping: If the AI stops improving on the test test, it stops training early to prevent "overfitting" (memorizing the test answers).
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from models.gru_attention import GRUAttentionModel
from models.dataset import StockSequenceDataset

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================
# CONFIG & UTILITIES
# =============================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load project config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Set random seeds. Why? AI training uses a lot of randomness (like Dropout).
    Setting a seed ensures that if we run the code twice, we get the exact same results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Checks if we have a GPU (CUDA) and activates it, otherwise falls back to slow CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  WARNING: CUDA not available, using CPU (training will be very slow)")
    return device


# =============================================
# FEATURE COLUMNS
# =============================================

# These are the columns the AI is allowed to look at.
# We explicitly exclude 'date' (AI can't read dates directly) and 'target' (the answer key!).
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "log_return", "bollinger_pctb", "atr", "rsi",
    "macd", "macd_signal", "macd_histogram",
    "stoch_k", "stoch_d", "obv", "volume_sma_ratio",
    "india_vix", "vix_available",
]

# We need to normalize these so large numbers (like volume) don't overpower small numbers (like RSI).
# We exclude 'vix_available' because it's just a 0 or 1 flag.
NORMALIZE_COLS = [c for c in FEATURE_COLS if c != "vix_available"]


# =============================================
# WALK-FORWARD WINDOW LOGIC
# =============================================

def compute_walk_forward_windows(
    df: pd.DataFrame, config: dict
) -> List[Dict[str, pd.Timestamp]]:
    """
    Calculates the sliding date ranges for training.
    e.g., Window 1: Train 2000-2009, Test 2010-2011.
          Window 2: Train 2002-2011, Test 2012-2013.
    This simulates how a real trading bot would be continually retrained as years pass.
    """
    train_years = config["walk_forward"]["train_years"]
    test_years = config["walk_forward"]["test_years"]
    step_years = config["walk_forward"]["step_years"]

    start_year = df["date"].dt.year.min()
    end_year = df["date"].dt.year.max()

    windows = []
    train_start_year = start_year

    while True:
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1

        if test_end_year > end_year:
            break

        windows.append({
            "train_start": pd.Timestamp(f"{train_start_year}-01-01"),
            "train_end": pd.Timestamp(f"{train_end_year}-12-31"),
            "test_start": pd.Timestamp(f"{test_start_year}-01-01"),
            "test_end": pd.Timestamp(f"{test_end_year}-12-31"),
        })

        train_start_year += step_years

    return windows


def split_by_window(
    df: pd.DataFrame, window: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cuts the big dataset into the specific Train and Test slices for a window."""
    train_mask = (df["date"] >= window["train_start"]) & (df["date"] <= window["train_end"])
    test_mask = (df["date"] >= window["test_start"]) & (df["date"] <= window["test_end"])

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    return train_df, test_df


# =============================================
# NORMALIZATION
# =============================================

def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score normalization: Converts data into standard deviations from the mean.
    
    CRITICAL ANTI-BIAS FEATURE: 
    We calculate the "mean" and "std" (standard deviation) using ONLY the training data.
    Then we apply that exact same math to the test data.
    If we used test data to calculate the mean, the AI would be "looking into the future" 
    to see what the average price of 2011 was while still living in 2009.
    """
    mean = train_df[NORMALIZE_COLS].mean()
    std = train_df[NORMALIZE_COLS].std()

    # Guard against division by zero if a column is completely flat (std=0)
    std = std.replace(0, 1.0)

    train_norm = train_df.copy()
    test_norm = test_df.copy()
    train_norm[NORMALIZE_COLS] = (train_df[NORMALIZE_COLS] - mean) / std
    test_norm[NORMALIZE_COLS] = (test_df[NORMALIZE_COLS] - mean) / std

    return train_norm, test_norm, mean, std


# =============================================
# DATA PREPARATION
# =============================================

def prepare_dataloaders(
    train_df: pd.DataFrame,
    seq_length: int,
    batch_size: int,
    val_split: float = 0.2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaders are PyTorch helpers that feed data to the GPU in small "batches" 
    (e.g., 64 sequences at a time) instead of loading the whole 10 years at once.
    """
    features = train_df[FEATURE_COLS].values.astype(np.float32)
    targets = train_df["target"].values.astype(np.float32)

    # We take the last 20% of the training data to use as a "Validation" set.
    # The AI is tested on this Validation set during training to see if it's learning.
    split_idx = int(len(features) * (1 - val_split))

    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]

    train_dataset = StockSequenceDataset(train_features, train_targets, seq_length)
    val_dataset = StockSequenceDataset(val_features, val_targets, seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Mix up the order so the AI doesn't just memorize the sequence of days
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Keep validation in chronological order
        pin_memory=pin_memory,
    )

    print(f"  Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader


# =============================================
# TRAINING LOOP
# =============================================

def train_one_window(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    device: torch.device,
    save_path: str,
) -> Dict:
    """
    The actual training loop where the AI learns.
    """
    model_cfg = config["model"]
    gpu_cfg = config["gpu"]

    # 1. Create the AI Brain
    model = GRUAttentionModel(
        num_features=num_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # 2. Setup the "Teacher"
    # BCEWithLogitsLoss is the mathematical formula used to grade the AI.
    # It checks how close the AI's prediction was to the real answer (UP or DOWN).
    # "WithLogits" means it handles the final 0-100% conversion internally for safety.
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW is the "Optimizer" — the algorithm that actually tweaks the brain's 
    # neurons based on the grade it got from the Teacher.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=1e-5,
    )

    # 3. Setup AMP (Fast GPU Math)
    # This uses the RTX 4070's Tensor Cores to do math in FP16 (Half Precision)
    # instead of FP32. It uses half the VRAM and is twice as fast.
    use_amp = gpu_cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Gradient Accumulation: If batch size is 64 and accumulation is 4, 
    # the AI effectively trains on a batch size of 256. This saves VRAM.
    accumulation_steps = gpu_cfg["gradient_accumulation_steps"]

    # 4. Setup Early Stopping
    # If the AI's test grade doesn't improve for 'patience' (e.g., 10) times in a row, we stop training.
    patience = model_cfg["early_stopping_patience"]
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    train_losses = []
    val_losses = []

    max_epochs = model_cfg["max_epochs"]
    print(f"  Training for up to {max_epochs} epochs (patience={patience})...")
    print(f"  AMP={'ON' if use_amp else 'OFF'}, "
          f"Accumulation steps={accumulation_steps}, "
          f"Effective batch={model_cfg['batch_size'] * accumulation_steps}")

    for epoch in range(max_epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        optimizer.zero_grad() # Erase old grades

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Move data to the GPU
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # AI makes a guess (using Fast Math / AMP)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(x_batch, return_logits=True)
                loss = criterion(predictions, y_batch) # Teacher grades the guess
                loss = loss / accumulation_steps

            # Tell the optimizer how wrong it was
            scaler.scale(loss).backward()

            # Every 'accumulation_steps', tweak the brain neurons to do better next time
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_loss += loss.item() * accumulation_steps
            num_train_batches += 1

        # Handle any leftover batches
        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = epoch_train_loss / num_train_batches

        # --- VALIDATION PHASE ---
        # Test the AI on data it hasn't seen during this training loop
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad(): # Don't learn, just test
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    predictions = model(x_batch, return_logits=True)
                    loss = criterion(predictions, y_batch)

                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / max(num_val_batches, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # --- EARLY STOPPING CHECK ---
        # Did the AI get its best grade yet?
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            # Yes! Save its brain to the hard drive.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
            }, save_path)
        else:
            # No, it got worse or didn't improve.
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            marker = " ★" if patience_counter == 0 else ""
            print(f"    Epoch {epoch:3d} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Patience: {patience_counter}/{patience}{marker}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(train_losses),
    }


# =============================================
# MAIN ORCHESTRATOR
# =============================================

def train_stock(
    stock_name: str,
    config: dict,
    window_idx: Optional[int] = None,
) -> List[Dict]:
    """
    The main boss function. It loads the Excel-like data, sets up the windows, 
    and commands train_one_window() to run for each window.
    """
    print(f"\n{'=' * 70}")
    print(f"STAGE 1 TRAINING: {stock_name}")
    print(f"{'=' * 70}")

    set_seed(config["seed"])
    device = get_device()

    processed_path = os.path.join("Data", "processed", f"{stock_name.lower()}_features.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} rows from {processed_path}")

    windows = compute_walk_forward_windows(df, config)
    print(f"  Walk-forward windows: {len(windows)}")

    if window_idx is not None:
        if window_idx >= len(windows):
            raise ValueError(f"Window index {window_idx} out of range (max {len(windows) - 1})")
        windows = [windows[window_idx]]
        print(f"  Training only window {window_idx}")

    results = []
    seq_length = config["model"]["seq_length"]
    batch_size = config["model"]["batch_size"]
    num_features = len(FEATURE_COLS)

    for i, window in enumerate(windows):
        actual_idx = window_idx if window_idx is not None else i
        print(f"\n  --- Window {actual_idx}: "
              f"Train {window['train_start'].year}-{window['train_end'].year}, "
              f"Test {window['test_start'].year}-{window['test_end'].year} ---")

        train_df, test_df = split_by_window(df, window)
        print(f"  Train rows: {len(train_df)}, Test rows: {len(test_df)}")

        if len(train_df) < seq_length + 10:
            print(f"  SKIPPING — insufficient training data ({len(train_df)} rows)")
            continue

        train_norm, test_norm, mean, std = normalize_features(train_df, test_df)

        # Save normalization stats to the hard drive so the RL Agent (Stage 2) can use them later!
        stats_dir = os.path.join(config["results_dir"], "norm_stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{stock_name.lower()}_window{actual_idx}_stats.npz")
        np.savez(stats_path, mean=mean.values, std=std.values, columns=NORMALIZE_COLS)

        train_loader, val_loader = prepare_dataloaders(
            train_norm, seq_length, batch_size,
            val_split=0.2,
            pin_memory=config["gpu"]["pin_memory"],
        )

        save_path = os.path.join(
            config["results_dir"], "models",
            f"{stock_name.lower()}_window{actual_idx}.pt"
        )

        result = train_one_window(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            num_features=num_features,
            device=device,
            save_path=save_path,
        )

        result["window_idx"] = actual_idx
        result["stock"] = stock_name
        result["save_path"] = save_path
        results.append(result)

        print(f"  Window {actual_idx} done: "
              f"best_epoch={result['best_epoch']}, "
              f"best_val_loss={result['best_val_loss']:.6f}, "
              f"total_epochs={result['total_epochs']}")

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE: {stock_name}")
    print(f"{'=' * 70}")
    for r in results:
        print(f"  Window {r['window_idx']}: "
              f"val_loss={r['best_val_loss']:.6f}, "
              f"epochs={r['total_epochs']}, "
              f"saved to {r['save_path']}")

    return results


# =============================================
# CLI ENTRY POINT
# =============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU+Attention model (Stage 1)")
    parser.add_argument("--stock", type=str, default="RELIANCE",
                        help="Stock name (default: RELIANCE)")
    parser.add_argument("--window", type=int, default=None,
                        help="Specific walk-forward window index (default: all)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train_stock(args.stock, config, window_idx=args.window)
