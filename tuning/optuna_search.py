"""
Hyperparameter optimization for Stage 1 GRU+Attention model using Optuna.

Strategy:
  - Bayesian optimization (TPE sampler) — smarter than grid/random search
  - Trains on 2 representative stocks (TCS, HDFCBANK) for speed
  - Uses 2 walk-forward windows per stock to avoid overfitting to a single period
  - Optimizes for MCC (Matthews Correlation Coefficient) — the best single metric
    for imbalanced binary classification. MCC = 0 is random, MCC = 1 is perfect.
  - Each trial takes ~1-3 minutes on RTX 4070, 30 trials ≈ 30-90 minutes total

Usage:
    python -m tuning.optuna_search                     # Run 30 trials (default)
    python -m tuning.optuna_search --n-trials 50       # Run 50 trials
    python -m tuning.optuna_search --resume             # Resume previous study

Output:
    - Best hyperparameters printed to console
    - Optuna study saved to tuning/optuna_study.db (SQLite, resumable)
    - Best config written to tuning/best_config.yaml
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
from sklearn.metrics import matthews_corrcoef
from typing import Dict, List, Tuple

import optuna
from optuna.samplers import TPESampler

from models.gru_attention import GRUAttentionModel
from models.dataset import StockSequenceDataset
from models.train_predictor import (
    load_config,
    get_feature_cols,
    get_scale_cols,
    compute_walk_forward_windows,
    split_by_window,
    normalize_features,
    set_seed,
    get_device,
)

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# Stocks to tune on — chosen for diversity (IT + Banking) and data quality
TUNE_STOCKS = ["TCS", "HDFCBANK"]

# Windows to evaluate — use 2 spread-out windows for each stock
# This prevents overfitting HPs to a single market regime
TUNE_WINDOW_INDICES = [1, 3]  # Middle windows — more balanced data


def train_and_evaluate_trial(
    config: dict,
    trial_params: dict,
    stock_name: str,
    window_idx: int,
    df: pd.DataFrame,
    feature_cols: list,
    scale_cols: list,
    device: torch.device,
) -> float:
    """
    Train a model with trial hyperparameters on one stock/window and return MCC.
    
    This is a lightweight version of train_one_window — no checkpointing,
    no loss curve saving, just fast training + test evaluation.
    """
    horizon = config["target"]["horizon"]
    
    # Get the specific window
    windows = compute_walk_forward_windows(df, config)
    if window_idx >= len(windows):
        return 0.0  # Skip if window doesn't exist for this stock
    window = windows[window_idx]
    
    # Split and normalize
    train_df, test_df = split_by_window(df, window, horizon=horizon)
    
    seq_length = trial_params["seq_length"]
    if len(train_df) < seq_length + 20 or len(test_df) < seq_length + 10:
        return 0.0
    
    train_norm, test_norm, _ = normalize_features(train_df, test_df, scale_cols)
    
    # Prepare training data
    train_features = train_norm[feature_cols].values.astype(np.float32)
    train_targets = train_norm["target"].values.astype(np.float32)
    
    # Split into train/val (80/20)
    split_idx = int(len(train_features) * 0.8)
    train_dataset = StockSequenceDataset(
        train_features[:split_idx], train_targets[:split_idx], seq_length
    )
    val_dataset = StockSequenceDataset(
        train_features[split_idx:], train_targets[split_idx:], seq_length
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=trial_params["batch_size"],
        shuffle=True, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=trial_params["batch_size"],
        shuffle=False, pin_memory=True,
    )
    
    # Build model with trial hyperparameters
    num_features = len(feature_cols)
    model = GRUAttentionModel(
        num_features=num_features,
        hidden_size=trial_params["hidden_size"],
        num_layers=trial_params["num_layers"],
        num_heads=trial_params["num_heads"],
        dropout=trial_params["dropout"],
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trial_params["learning_rate"],
        weight_decay=trial_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # AMP setup
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Train with early stopping (reduced patience for speed)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 7  # Faster early stopping for HPO
    max_epochs = 50  # Cap for speed
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(batch_x, return_logits=True).squeeze()
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(batch_x, return_logits=True).squeeze()
                    val_loss = criterion(output, batch_y)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state in memory (no disk I/O)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Evaluate on TEST set using best model
    model.load_state_dict(best_state)
    model.eval()
    
    test_features = test_norm[feature_cols].values.astype(np.float32)
    test_targets = test_norm["target"].values.astype(np.float32)
    test_dataset = StockSequenceDataset(test_features, test_targets, seq_length, filter_neutrals=True)
    
    if len(test_dataset) == 0:
        return 0.0
    
    all_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    all_y = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))]).cpu().numpy()
    
    with torch.no_grad():
        probs = model(all_x).cpu().numpy()
    
    preds = (probs > 0.5).astype(float)
    
    # Return MCC — ranges from -1 to +1, 0 = random
    try:
        mcc = matthews_corrcoef(all_y, preds)
    except Exception:
        mcc = 0.0
    
    return mcc


def objective(trial: optuna.Trial, config: dict, device: torch.device) -> float:
    """
    Optuna objective function. Suggests hyperparameters, trains across
    multiple stocks/windows, and returns the average MCC.
    """
    # --- Suggest hyperparameters ---
    trial_params = {
        "seq_length": trial.suggest_categorical("seq_length", [15, 20, 30, 40, 60]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "dropout": trial.suggest_float("dropout", 0.15, 0.50, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    
    # Constraint: hidden_size must be divisible by num_heads
    if trial_params["hidden_size"] % trial_params["num_heads"] != 0:
        raise optuna.TrialPruned()
    
    print(f"\n  Trial {trial.number}: seq={trial_params['seq_length']}, "
          f"hidden={trial_params['hidden_size']}, layers={trial_params['num_layers']}, "
          f"heads={trial_params['num_heads']}, dropout={trial_params['dropout']:.2f}, "
          f"lr={trial_params['learning_rate']:.2e}, wd={trial_params['weight_decay']:.2e}, "
          f"batch={trial_params['batch_size']}")
    
    # --- Evaluate across stocks and windows ---
    mccs = []
    
    for stock_name in TUNE_STOCKS:
        processed_path = os.path.join("Data", "processed", f"{stock_name.lower()}_features.csv")
        if not os.path.exists(processed_path):
            continue
        
        df = pd.read_csv(processed_path)
        df["date"] = pd.to_datetime(df["date"])
        feature_cols = get_feature_cols(df)
        scale_cols = get_scale_cols(feature_cols)
        
        for win_idx in TUNE_WINDOW_INDICES:
            mcc = train_and_evaluate_trial(
                config, trial_params, stock_name, win_idx,
                df, feature_cols, scale_cols, device,
            )
            mccs.append(mcc)
            print(f"    {stock_name} W{win_idx}: MCC={mcc:.4f}")
    
    avg_mcc = np.mean(mccs) if mccs else 0.0
    print(f"  → Avg MCC: {avg_mcc:.4f}")
    
    return avg_mcc


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Stage 1")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous study from database")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    
    print(f"\n{'=' * 70}")
    print(f"HYPERPARAMETER OPTIMIZATION — Stage 1")
    print(f"{'=' * 70}")
    print(f"  Stocks: {TUNE_STOCKS}")
    print(f"  Windows per stock: {TUNE_WINDOW_INDICES}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Objective: Maximize MCC (Matthews Correlation Coefficient)")
    
    # Create study (resumable via SQLite)
    db_path = os.path.join("tuning", "optuna_study.db")
    storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name="stage1_hpo",
        direction="maximize",  # Maximize MCC
        sampler=TPESampler(seed=config["seed"]),
        storage=storage,
        load_if_exists=args.resume,
    )
    
    study.optimize(
        lambda trial: objective(trial, config, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # --- Print results ---
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    
    best = study.best_trial
    print(f"\n  Best Trial: #{best.number}")
    print(f"  Best MCC: {best.value:.4f}")
    print(f"\n  Best Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    
    # --- Save best config ---
    best_config = config.copy()
    best_config["model"]["seq_length"] = best.params["seq_length"]
    best_config["model"]["hidden_size"] = best.params["hidden_size"]
    best_config["model"]["num_layers"] = best.params["num_layers"]
    best_config["model"]["num_heads"] = best.params["num_heads"]
    best_config["model"]["dropout"] = best.params["dropout"]
    best_config["model"]["learning_rate"] = best.params["learning_rate"]
    best_config["model"]["batch_size"] = best.params["batch_size"]
    
    best_config_path = os.path.join("tuning", "best_config.yaml")
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n  Best config saved to: {best_config_path}")
    print(f"  Study database: {db_path}")
    
    # --- Top 5 trials ---
    print(f"\n  Top 5 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:5]
    for t in top_trials:
        if t.value is not None:
            print(f"    #{t.number}: MCC={t.value:.4f} | "
                  f"seq={t.params.get('seq_length')}, "
                  f"hidden={t.params.get('hidden_size')}, "
                  f"layers={t.params.get('num_layers')}, "
                  f"lr={t.params.get('learning_rate', 0):.2e}")
    
    print(f"\n  Next steps:")
    print(f"  1. Copy best params to config.yaml")
    print(f"  2. Retrain all stocks: python -m models.train_predictor --stock RELIANCE")
    print(f"  3. Evaluate: python -m models.evaluate")
    print(f"  4. Compare: python _compare_metrics.py")


if __name__ == "__main__":
    main()
