"""
End-to-end feature engineering pipeline.

Orchestrates: load CSV → clean → compute indicators → merge VIX → 
add target → save to Data/processed/.

CRITICAL: This module does NOT perform normalization.
Normalization must be done per walk-forward window at training time
to prevent look-ahead bias. See evaluation/walk_forward.py.
"""

import os
import yaml
import pandas as pd
import numpy as np

from features.technical_indicators import compute_all_indicators
from features.vix import load_vix, merge_vix


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project config from YAML."""
    # Opens and parses the central config file which holds hyperparameters, 
    # paths to stock files, and walk-forward settings.
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_clean_stock(csv_path: str) -> pd.DataFrame:
    """
    Load a stock CSV and clean it.
    
    Steps:
    1. Parse timestamp → extract date only (strip time & timezone)
    2. Drop 'oi' column (always 0 for equity)
    3. Drop 'asset' column (redundant — file name identifies stock)
    4. Sort by date ascending
    5. Remove any duplicate dates
    6. Verify no NaN in raw OHLCV
    
    Args:
        csv_path: Path to the stock CSV file.
    
    Returns:
        Clean DataFrame with columns [date, open, high, low, close, volume].
    """
    # Read the raw data outputted by the Upstox API script
    df = pd.read_csv(csv_path)

    # The raw timestamp contains exact hours/minutes. We are building a daily model,
    # so we extract just the calendar date to make merging (e.g. with VIX) robust.
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    # 'oi' (Open Interest) is only relevant for futures/options, not equity.
    # 'asset' is just the ticker symbol, which is implicit in the filename.
    cols_to_drop = [c for c in ["timestamp", "oi", "asset"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Protect against bad API data returning the same day twice
    df = df.drop_duplicates(subset="date", keep="first")

    # Time-series models absolutely require chronological ordering
    df = df.sort_values("date").reset_index(drop=True)

    # If the raw data is missing open/close prices, our indicators will silently 
    # produce garbage. We fail fast here if raw data is corrupted.
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    nan_counts = df[ohlcv_cols].isnull().sum()
    if nan_counts.any():
        raise ValueError(
            f"NaN found in raw OHLCV data at {csv_path}:\n{nan_counts[nan_counts > 0]}"
        )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary direction target.
    
    target = 1 if close[t+1] > close[t] else 0
    
    The last row will have NaN target (no next day) → dropped later.
    """
    result = df.copy()
    
    # We define 'UP' as tomorrow's close being strictly greater than today's close.
    # We shift the close column by -1 (bringing tomorrow's close to today's row),
    # perform the > comparison, and convert the boolean (True/False) to float (1.0/0.0).
    result["target"] = (result["close"].shift(-1) > result["close"]).astype(float)

    # The very last day in our dataset has no "tomorrow". We cannot know its target.
    # Shifting creates a false 'False' (0.0) here, so we explicitly overwrite it to NaN
    # so it gets dropped by the pipeline's NaN cleaner later.
    result.loc[result.index[-1], "target"] = np.nan

    return result


def run_pipeline(config_path: str = "config.yaml") -> dict:
    """
    Run the full feature engineering pipeline for all stocks.
    
    Steps per stock:
    1. Load & clean raw CSV
    2. Compute all technical indicators
    3. Merge India VIX (with vix_available flag)
    4. Add binary target (next-day direction)
    5. Drop initial NaN rows (from rolling indicators)
    6. Drop final row (no target)
    7. Verify no NaN remains
    8. Save to Data/processed/{stock}_features.csv
    
    Returns:
        Dict of {stock_name: summary_dict} with stats about each processed file.
    """
    # 1. Load the central configuration
    config = load_config(config_path)
    
    # 2. Ensure the output directory exists
    processed_dir = os.path.join("Data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 3. Load VIX data into memory once, as it will be merged with every stock
    vix_path = config["vix"]["file"]
    vix_df = load_vix(vix_path)
    print(f"VIX loaded: {len(vix_df)} rows, range {vix_df['date'].min()} to {vix_df['date'].max()}")

    summaries = {}

    # Loop over every stock defined in config.yaml
    for stock_cfg in config["stocks"]:
        name = stock_cfg["name"]
        csv_path = stock_cfg["file"]

        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {name}")
        print(f"{'=' * 60}")

        # Step A: Load and clean raw OHLCV
        df = load_and_clean_stock(csv_path)
        print(f"  Raw: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")

        # Step B: Compute the 11 technical indicators (RSI, MACD, etc.)
        df = compute_all_indicators(df)
        indicator_cols = [
            "log_return", "bollinger_pctb", "atr", "rsi",
            "macd", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "obv", "volume_sma_ratio"
        ]
        print(f"  After indicators: {len(df)} rows, {len(indicator_cols)} indicators added")

        # Step C: Merge the India VIX data and the vix_available binary flag
        df = merge_vix(df, vix_df)
        vix_count = df["vix_available"].sum()
        print(f"  VIX merged: {int(vix_count)} rows with VIX data, {len(df) - int(vix_count)} without")

        # Step D: Add the binary prediction target (1.0 for UP, 0.0 for DOWN)
        df = add_target(df)

        # Step E: Handle NaNs
        # Rolling indicators (like 20-day SMA) inherently produce NaNs for the first 19 days.
        # The add_target function produces a NaN for the very last day.
        # We drop all of these rows.
        nan_before = len(df)
        df = df.dropna().reset_index(drop=True)
        nan_dropped = nan_before - len(df)
        print(f"  Dropped {nan_dropped} NaN rows (initial rolling warmup + final target row)")

        # Step F: Final Sanity Check
        # If any NaNs slipped through our logic, the PyTorch model will crash later.
        # We fail fast here if the dataset is not perfectly clean.
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            nan_cols = df.isnull().sum()
            raise ValueError(
                f"NaN still present in {name} after processing!\n"
                f"{nan_cols[nan_cols > 0]}"
            )

        # Log target distribution (should ideally be close to 50/50 for stocks)
        target_dist = df["target"].value_counts()
        print(f"  Target distribution: UP={int(target_dist.get(1.0, 0))}, "
              f"DOWN={int(target_dist.get(0.0, 0))}")

        # Step G: Save to disk
        # This file will be read by the PyTorch Dataset loaders in Stage 1
        out_path = os.path.join(processed_dir, f"{name.lower()}_features.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

        # Record summary stats for final output
        summaries[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "target_up_pct": round(target_dist.get(1.0, 0) / len(df) * 100, 1),
            "vix_coverage_pct": round(vix_count / len(df) * 100, 1),
            "output_path": out_path,
        }

    # Print a nice table of all processed stocks at the end
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    for name, s in summaries.items():
        print(f"  {name}: {s['rows']} rows, {s['date_range']}, "
              f"UP={s['target_up_pct']}%, VIX coverage={s['vix_coverage_pct']}%")

    # List the final feature set. These are the explicit inputs that the neural network will see.
    feature_cols = [c for c in df.columns if c not in ["date", "target"]]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    return summaries


if __name__ == "__main__":
    run_pipeline()
