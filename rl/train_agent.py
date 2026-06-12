"""
Train the Stage 2 Reinforcement Learning Agent.

In simple words: 
This script trains the RL agent (MaskablePPO) using "Out-Of-Fold" (OOF) predictions.
It takes a 10-year training window, splits it into 70/30.
It trains the Stage 1 model on the 70%, makes predictions on the 30%, 
and gives those predictions to the RL agent to learn how to trade.
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from models.train_predictor import (
    load_config,
    get_feature_cols,
    get_scale_cols,
    compute_walk_forward_windows,
    split_by_window,
    normalize_features,
    set_seed,
    get_device,
    prepare_dataloaders,
    train_one_window
)
from models.dataset import StockSequenceDataset
from models.inference import mc_dropout_predict
from models.gru_attention import GRUAttentionModel

from rl.trading_env import TradingEnv
from rl.reward import compute_sortino

class LoggingCallback(BaseCallback):
    """
    Custom callback to log specific metrics from the environment's info dict.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if len(self.locals.get("dones", [])) > 0 and self.locals["dones"][0]:
            info = self.locals["infos"][0]
            if "episode" in info and "hold_pct" in info["episode"]:
                ep_info = info["episode"]
                self.logger.record("custom/hold_pct", ep_info["hold_pct"])
                self.logger.record("custom/total_trades", ep_info["total_trades"])
                self.logger.record("custom/final_portfolio", ep_info["final_portfolio_value"])
        return True


def generate_oof_predictions(stock_name: str, window_idx: int, config: dict, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    """
    Generates Out-Of-Fold (OOF) predictions for the RL agent to train on.
    
    1. Takes the train portion of the current walk-forward window.
    2. Splits that train portion into 70% sub-train, 30% sub-val.
    3. Trains a fresh Stage 1 model on the 70%.
    4. Predicts on the 30%.
    5. Returns a DataFrame of [date, close, india_vix, p_up, confidence] for the 30%.
    """
    print(f"\n--- Generating OOF Data for {stock_name} Window {window_idx} ---")
    
    model_cfg = config["model"]
    seq_length = model_cfg["seq_length"]
    horizon = config["target"]["horizon"]
    
    windows = compute_walk_forward_windows(df, config)
    window = windows[window_idx]
    
    # 1. Get the main training data for this window
    train_df, _ = split_by_window(df, window, horizon=horizon)
    
    # 2. Split it 70/30 chronologically
    split_idx = int(len(train_df) * 0.7)
    sub_train_df = train_df.iloc[:split_idx].copy()
    sub_val_df = train_df.iloc[split_idx:].copy()
    
    print(f"  Sub-Train: {sub_train_df['date'].min().date()} to {sub_train_df['date'].max().date()} ({len(sub_train_df)} rows)")
    print(f"  Sub-Val (OOF):   {sub_val_df['date'].min().date()} to {sub_val_df['date'].max().date()} ({len(sub_val_df)} rows)")
    
    # 3. Normalize using only sub-train stats
    feature_cols = get_feature_cols(df)
    scale_cols = get_scale_cols(feature_cols)
    sub_train_norm, sub_val_norm, _ = normalize_features(sub_train_df, sub_val_df, scale_cols)
    
    # 4. Train Stage 1 model
    # We use a smaller epochs count since this is just for OOF generation
    original_epochs = model_cfg.get("epochs", 50)
    model_cfg["epochs"] = min(original_epochs, 20) 
    
    # The actual train_one_window function needs a results_dir that doesn't overwrite our real models.
    # We will pass a dummy temp dir to it.
    oof_dir = os.path.join(config["results_dir"], stock_name, "oof_temp")
    os.makedirs(oof_dir, exist_ok=True)
    save_path = os.path.join(oof_dir, f"oof_model_window{window_idx}.pt")
    
    train_loader, val_loader = prepare_dataloaders(
        sub_train_norm, feature_cols, seq_length, 
        model_cfg.get("batch_size", 32),
        val_split=0.2, pin_memory=config["gpu"].get("pin_memory", True)
    )
    
    result = train_one_window(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=len(feature_cols),
        device=device,
        save_path=save_path
    )
    
    # Load the best model weights
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model = GRUAttentionModel(
        num_features=len(feature_cols),
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_cfg["epochs"] = original_epochs
    
    # 5. Run MC Dropout Inference on the sub-val (OOF) data
    test_features = sub_val_norm[feature_cols].values.astype(np.float32)
    test_targets = sub_val_norm["target"].values.astype(np.float32)
    
    # filter_neutrals=False because we need contiguous predictions for the RL environment
    test_dataset = StockSequenceDataset(test_features, test_targets, seq_length, filter_neutrals=False)
    
    if len(test_dataset) == 0:
        raise ValueError("Not enough OOF data to form sequences.")
        
    all_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    
    mc_result = mc_dropout_predict(model, all_x, num_passes=30)
    
    p_up = mc_result["p_up"].cpu().numpy()
    confidence = mc_result["confidence"].cpu().numpy()
    
    # 6. Assemble the RL DataFrame
    # Note: dataset drops the first (seq_length - 1) days because they don't have enough history
    # The first prediction corresponds to index (seq_length - 1) in the dataframe
    
    oof_dates = sub_val_df["date"].iloc[seq_length - 1:].values
    oof_close = sub_val_df["close"].iloc[seq_length - 1:].values
    
    # Get raw VIX for the observation (un-normalized)
    if "india_vix" in sub_val_df.columns:
        oof_vix = sub_val_df["india_vix"].iloc[seq_length - 1:].values
    else:
        oof_vix = np.zeros_like(oof_close)
        
    rl_df = pd.DataFrame({
        "date": oof_dates,
        "close": oof_close,
        "india_vix": oof_vix,
        "p_up": p_up,
        "confidence": confidence
    })
    
    print(f"  Generated OOF RL Data: {len(rl_df)} rows")
    return rl_df


def train_rl_agent(stock_name: str, window_idx: int, config: dict):
    """
    Main orchestration for training the RL agent on a single window.
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING RL AGENT: {stock_name} (Window {window_idx})")
    print(f"{'=' * 70}")
    
    set_seed(config["seed"])
    device = get_device()
    
    # 1. Load raw data
    processed_path = os.path.join("Data", "processed", f"{stock_name.lower()}_features.csv")
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Generate OOF data for RL training
    rl_train_df = generate_oof_predictions(stock_name, window_idx, config, df, device)
    
    # 3. Setup Environment
    # We wrap the environment creation in a lambda for SB3
    def make_env():
        return TradingEnv(rl_train_df, config)
        
    env = make_vec_env(make_env, n_envs=1)
    
    # 4. Initialize MaskablePPO
    rl_cfg = config.get("rl", {})
    # Defaults in case config is missing them
    lr = rl_cfg.get("learning_rate", 3e-4)
    clip_range = rl_cfg.get("clip_range", 0.2)
    ent_coef = rl_cfg.get("ent_coef", 0.01)
    n_steps = rl_cfg.get("n_steps", 512)
    batch_size = rl_cfg.get("batch_size", 64)
    total_timesteps = rl_cfg.get("total_timesteps", 100000)
    
    # Ensure n_steps is not larger than our OOF dataset
    actual_n_steps = min(n_steps, len(rl_train_df) - 10)
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_steps=actual_n_steps,
        batch_size=batch_size,
        verbose=1,
        seed=config["seed"],
        device=device
    )
    
    # 5. Train
    print("\n--- Starting PPO Training ---")
    callback = LoggingCallback()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 6. Save Model
    save_dir = os.path.join(config["results_dir"], stock_name, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rl_window{window_idx}.zip")
    
    model.save(save_path)
    print(f"\n--- RL Model Saved to {save_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True, help="Stock name")
    parser.add_argument("--window", type=int, required=True, help="Walk-forward window index")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    train_rl_agent(args.stock, args.window, config)
