"""
Walk-Forward Evaluation Orchestrator — Phase 4 main entry point.

In simple words: This is the BOSS SCRIPT. It ties the entire pipeline together:
1. Loads data for a stock
2. For each walk-forward window:
   a. Normalizes test data using the SAVED training scaler (no data leakage)
   b. Runs Stage 1 MC Dropout inference to produce predictions
   c. Ensures an RL model exists (trains one if missing)
   d. Runs all 3 strategies (Buy-Hold, Predictor-Only, Full RL)
   e. Computes metrics and generates charts
3. Saves results to JSON and generates an aggregate summary

This is the script you run to answer: "Does this system actually make money?"

Usage:
    python -m evaluation.walk_forward --stock RELIANCE --window 0   # One window
    python -m evaluation.walk_forward --stock RELIANCE              # All windows
    python -m evaluation.walk_forward                               # All stocks
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Import project modules
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
from models.gru_attention import GRUAttentionModel
from models.dataset import StockSequenceDataset
from models.inference import mc_dropout_predict

from evaluation.backtest import (
    run_all_strategies,
    generate_tearsheet,
    print_comparison_table,
)
from evaluation.visualise import (
    generate_all_charts,
    plot_aggregate_table,
)


# =============================================
# CORE: GENERATE TEST PREDICTIONS
# =============================================

def generate_test_predictions(
    stock_name: str,
    window_idx: int,
    window: dict,
    config: dict,
    df: pd.DataFrame,
    device: torch.device,
) -> Optional[pd.DataFrame]:
    """
    Generate Stage 1 predictions on the OUT-OF-SAMPLE test data.
    
    This is the critical data pipeline step:
    1. Split the full dataset by the window's date boundaries
    2. Load the SAVED scaler from training (never refit on test data!)
    3. Load the trained model checkpoint
    4. Run MC Dropout inference (T=50) to get p_up and confidence
    5. Return a DataFrame ready for backtesting
    
    Args:
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        window: Dict with train_start, train_end, test_start, test_end
        config: Project config
        df: Full processed DataFrame for this stock
        device: Torch device (CUDA or CPU)
        
    Returns:
        DataFrame with [date, close, india_vix, p_up, confidence] or None if failed.
    """
    model_cfg = config["model"]
    seq_length = model_cfg["seq_length"]
    horizon = config["target"]["horizon"]
    
    # --- 1. Split data by window ---
    train_df, test_df = split_by_window(df, window, horizon=horizon)
    
    if len(test_df) < seq_length + 10:
        print(f"    SKIP: Test set too small ({len(test_df)} rows)")
        return None
    
    # --- 2. Load the saved scaler (fitted during training) ---
    # CRITICAL ANTI-BIAS: We use the scaler that was fit on ONLY training data.
    # This ensures test data is normalized consistently without leaking test statistics.
    scaler_path = os.path.join(
        config["results_dir"], stock_name, "norm_stats", f"window{window_idx}_scaler.pkl"
    )
    
    if os.path.exists(scaler_path):
        # Load the saved scaler — this is the correct approach
        with open(scaler_path, "rb") as f:
            saved_scaler = pickle.load(f)
        
        feature_cols = get_feature_cols(df)
        scale_cols = get_scale_cols(feature_cols)
        
        # Apply the saved scaler to both splits
        # We need train_norm for the scaler stats, but we only USE test_norm
        test_norm = test_df.copy()
        test_norm[scale_cols] = saved_scaler.transform(test_df[scale_cols])
        
        # We also need to normalize train for the sequence padding at the start
        # But we don't actually use train predictions — this is just for scaler consistency
        
    else:
        # Fallback: recompute normalization (less ideal, but still correct 
        # because normalize_features fits on train_df only)
        print(f"    WARNING: No saved scaler found at {scaler_path}, recomputing...")
        feature_cols = get_feature_cols(df)
        scale_cols = get_scale_cols(feature_cols)
        _, test_norm, _ = normalize_features(train_df, test_df, scale_cols)
    
    # --- 3. Load the trained Stage 1 model ---
    checkpoint_path = os.path.join(
        config["results_dir"], stock_name, "models", f"window{window_idx}.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"    SKIP: No Stage 1 checkpoint at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get feature_cols from checkpoint (guaranteed consistency with training)
    feature_cols = checkpoint.get("feature_cols", get_feature_cols(df))
    num_features = len(feature_cols)
    
    model = GRUAttentionModel(
        num_features=num_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # --- 4. Create test sequences ---
    # filter_neutrals=False because we need contiguous predictions for the RL environment
    # (neutral-filtered gaps would break the daily stepping logic)
    test_features = test_norm[feature_cols].values.astype(np.float32)
    test_targets = test_norm["target"].values.astype(np.float32)
    
    test_dataset = StockSequenceDataset(
        test_features, test_targets, seq_length, filter_neutrals=False
    )
    
    if len(test_dataset) == 0:
        print(f"    SKIP: No valid test sequences")
        return None
    
    # Stack all sequences into a single batch for GPU inference
    all_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    
    # --- 5. Run MC Dropout Inference ---
    mc_passes = config["model"].get("mc_dropout_passes", 50)
    print(f"    Running MC Dropout ({mc_passes} passes) on {len(test_dataset)} sequences...")
    
    mc_result = mc_dropout_predict(model, all_x, num_passes=mc_passes)
    
    p_up = mc_result["p_up"].cpu().numpy()
    confidence = mc_result["confidence"].cpu().numpy()
    
    # --- 6. Assemble the backtesting DataFrame ---
    # The dataset drops the first (seq_length - 1) days because they need lookback history.
    # So the first prediction corresponds to index (seq_length - 1) in test_df.
    offset = seq_length - 1
    
    pred_dates = test_df["date"].iloc[offset:offset + len(p_up)].values
    pred_close = test_df["close"].iloc[offset:offset + len(p_up)].values
    
    # Get raw VIX (un-normalized) for the observation vector
    if "india_vix" in test_df.columns:
        pred_vix = test_df["india_vix"].iloc[offset:offset + len(p_up)].values
    else:
        pred_vix = np.zeros(len(p_up))
    
    # Build the final DataFrame that backtest.py and TradingEnv expect
    result_df = pd.DataFrame({
        "date": pred_dates,
        "close": pred_close,
        "india_vix": pred_vix,
        "p_up": p_up,
        "confidence": confidence,
    })
    
    print(f"    Generated predictions: {len(result_df)} days "
          f"({result_df['date'].iloc[0]} to {result_df['date'].iloc[-1]})")
    
    return result_df


# =============================================
# CORE: ENSURE RL MODEL EXISTS
# =============================================

def ensure_rl_model(
    stock_name: str,
    window_idx: int,
    config: dict,
) -> str:
    """
    Check if a trained RL model exists for this stock/window.
    If not, train one on the fly using rl.train_agent.
    
    This implements the "Option B" approach from the implementation plan:
    train on-the-fly, cache the result. First run is slow, subsequent runs are instant.
    
    Args:
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        config: Project config
        
    Returns:
        Path to the RL model .zip file.
    """
    rl_model_path = os.path.join(
        config["results_dir"], stock_name, "models", f"rl_window{window_idx}.zip"
    )
    
    if os.path.exists(rl_model_path):
        print(f"    RL model found: {rl_model_path}")
        return rl_model_path
    
    # No RL model exists — train one
    print(f"    No RL model found. Training RL agent for {stock_name} Window {window_idx}...")
    print(f"    (This will take a few minutes. The model is cached for future runs.)")
    
    from rl.train_agent import train_rl_agent
    train_rl_agent(stock_name, window_idx, config)
    
    if not os.path.exists(rl_model_path):
        raise FileNotFoundError(
            f"RL training completed but model not found at {rl_model_path}. "
            f"Check rl/train_agent.py for errors."
        )
    
    return rl_model_path


# =============================================
# ORCHESTRATION: SINGLE WINDOW
# =============================================

def evaluate_window(
    stock_name: str,
    window_idx: int,
    window: dict,
    config: dict,
    df: pd.DataFrame,
    device: torch.device,
) -> Optional[Dict]:
    """
    Run the full evaluation pipeline for a single walk-forward window.
    
    Steps:
    1. Generate Stage 1 predictions on test data
    2. Ensure RL model exists (train if missing)
    3. Run all 3 strategies
    4. Generate charts
    5. Generate tearsheet
    6. Return results dict
    
    Args:
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        window: Dict with date boundaries
        config: Project config
        df: Full processed DataFrame
        device: Torch device
        
    Returns:
        Dict with strategy results and metadata, or None if skipped.
    """
    print(f"\n  --- Window {window_idx}: "
          f"Test {window['test_start'].year}-{window['test_end'].year} ---")
    
    # Step 1: Generate predictions on test data
    test_df = generate_test_predictions(
        stock_name, window_idx, window, config, df, device
    )
    
    if test_df is None or len(test_df) < 20:
        print(f"    SKIPPING — insufficient test data")
        return None
    
    # Step 2: Ensure RL model exists
    rl_model_path = ensure_rl_model(stock_name, window_idx, config)
    
    # Step 3: Run all 3 strategies
    confidence_threshold = config.get("rl", {}).get("confidence_threshold", 0.6)
    strategy_results = run_all_strategies(
        test_df, config, rl_model_path, confidence_threshold
    )
    
    # Print comparison table to console
    print_comparison_table(strategy_results)
    
    # Step 4: Generate charts
    plots_dir = os.path.join(config["results_dir"], stock_name, "plots")
    generate_all_charts(strategy_results, test_df, stock_name, window_idx, plots_dir)
    
    # Step 5: Generate quantstats tearsheet (RL vs Buy-and-Hold)
    tearsheet_dir = os.path.join(config["results_dir"], stock_name, "tearsheets")
    os.makedirs(tearsheet_dir, exist_ok=True)
    
    if "full_rl" in strategy_results and "buy_and_hold" in strategy_results:
        generate_tearsheet(
            daily_returns=strategy_results["full_rl"]["daily_returns"],
            dates=strategy_results["full_rl"]["dates"],
            benchmark_returns=strategy_results["buy_and_hold"]["daily_returns"],
            output_path=os.path.join(tearsheet_dir, f"window{window_idx}.html"),
            title=f"{stock_name} Window {window_idx} — RL vs Buy-and-Hold",
        )
    
    # Step 6: Package results (JSON-serializable)
    window_result = {
        "window_idx": window_idx,
        "test_period": f"{window['test_start'].year}-{window['test_end'].year}",
        "test_days": len(test_df),
    }
    
    # Add metrics for each strategy (but NOT the large arrays — those are for charts only)
    for key, res in strategy_results.items():
        window_result[key] = {
            "strategy": res["strategy"],
            "metrics": res["metrics"],
            "num_trades": len(res["trade_log"]),
        }
    
    return window_result


# =============================================
# ORCHESTRATION: FULL STOCK
# =============================================

def evaluate_stock(stock_name: str, config: dict, window_idx: Optional[int] = None) -> Dict:
    """
    Run evaluation for all (or one) walk-forward windows of a stock.
    
    Args:
        stock_name: e.g., "RELIANCE"
        config: Project config
        window_idx: If set, evaluate only this window. If None, evaluate all.
        
    Returns:
        Dict with per-window results and aggregate metrics.
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {stock_name}")
    print(f"{'=' * 70}")
    
    set_seed(config["seed"])
    device = get_device()
    
    # Load the processed features CSV
    processed_path = os.path.join("Data", "processed", f"{stock_name.lower()}_features.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")
    
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    
    print(f"  Loaded {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
    
    # Compute walk-forward windows
    windows = compute_walk_forward_windows(df, config)
    print(f"  Total walk-forward windows: {len(windows)}")
    
    # If specific window requested, validate and select it
    if window_idx is not None:
        if window_idx >= len(windows):
            raise ValueError(f"Window {window_idx} out of range (max {len(windows) - 1})")
        windows_to_eval = [(window_idx, windows[window_idx])]
    else:
        windows_to_eval = list(enumerate(windows))
    
    # Run evaluation for each window
    all_results = []
    all_strategy_results = []  # For aggregate summary chart
    
    for idx, window in windows_to_eval:
        result = evaluate_window(stock_name, idx, window, config, df, device)
        if result is not None:
            all_results.append(result)
            # We need to re-run strategies to get equity curves for the aggregate chart
            # But we've already computed metrics — just store them
            all_strategy_results.append(result)
    
    # Generate aggregate summary chart (if we evaluated multiple windows)
    if len(all_results) > 1:
        plots_dir = os.path.join(config["results_dir"], stock_name, "plots")
        # Convert results to the format plot_aggregate_table expects
        agg_data = []
        for r in all_results:
            window_data = {}
            for strategy_key in ["buy_and_hold", "predictor_only", "full_rl"]:
                if strategy_key in r:
                    window_data[strategy_key] = r[strategy_key]
            agg_data.append(window_data)
        
        plot_aggregate_table(agg_data, stock_name, plots_dir)
    
    # Save results to JSON
    output = {
        "stock": stock_name,
        "evaluation_date": datetime.now().isoformat(),
        "total_windows": len(windows),
        "evaluated_windows": len(all_results),
        "windows": all_results,
    }
    
    # Compute aggregate metrics across all windows
    if all_results:
        for strategy_key in ["buy_and_hold", "predictor_only", "full_rl"]:
            strategy_metrics = [
                r[strategy_key]["metrics"] 
                for r in all_results if strategy_key in r
            ]
            if strategy_metrics:
                # Average each metric across windows
                avg_metrics = {}
                for key in strategy_metrics[0]:
                    values = [m[key] for m in strategy_metrics]
                    avg_metrics[key] = round(float(np.mean(values)), 4)
                output[f"{strategy_key}_aggregate"] = avg_metrics
    
    # Save JSON
    output_path = os.path.join(config["results_dir"], stock_name, "backtest_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results saved: {output_path}")
    
    # Print aggregate summary
    if all_results:
        print(f"\n  === AGGREGATE SUMMARY ({stock_name}) ===")
        for strategy_key in ["buy_and_hold", "predictor_only", "full_rl"]:
            agg_key = f"{strategy_key}_aggregate"
            if agg_key in output:
                m = output[agg_key]
                strategy_name = {"buy_and_hold": "Buy-Hold", "predictor_only": "Pred-Only", 
                                "full_rl": "Full RL"}[strategy_key]
                print(f"  {strategy_name:>12}: Return={m['total_return_pct']:>8.2f}%  "
                      f"Sortino={m['sortino_ratio']:>7.3f}  MaxDD={m['max_drawdown_pct']:>7.2f}%")
    
    return output


# =============================================
# CLI ENTRY POINT
# =============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-Forward Evaluation Orchestrator — Phase 4"
    )
    parser.add_argument(
        "--stock", type=str, default=None,
        help="Stock name (e.g., RELIANCE). If omitted, evaluates all stocks."
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Specific window index. If omitted, evaluates all windows."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file."
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.stock:
        # Evaluate a specific stock
        evaluate_stock(args.stock, config, window_idx=args.window)
    else:
        # Evaluate all stocks defined in config
        all_stocks_results = {}
        for stock_cfg in config["stocks"]:
            stock_name = stock_cfg["name"]
            try:
                result = evaluate_stock(stock_name, config, window_idx=args.window)
                all_stocks_results[stock_name] = result
            except Exception as e:
                print(f"\n  ERROR evaluating {stock_name}: {e}")
                continue
        
        # Print final cross-stock summary
        print(f"\n{'=' * 70}")
        print(f"CROSS-STOCK SUMMARY")
        print(f"{'=' * 70}")
        
        for stock_name, result in all_stocks_results.items():
            agg = result.get("full_rl_aggregate", {})
            if agg:
                print(f"  {stock_name:>12}: Return={agg.get('total_return_pct', 0):>8.2f}%  "
                      f"Sortino={agg.get('sortino_ratio', 0):>7.3f}  "
                      f"MaxDD={agg.get('max_drawdown_pct', 0):>7.2f}%")
