"""
End-to-end feature engineering pipeline.

Orchestrates: load CSV → clean → compute indicators → merge VIX →
merge NIFTY → compute relative features → add target → save to Data/processed/.

CRITICAL: This module does NOT perform normalization.
Normalization must be done per walk-forward window at training time
to prevent look-ahead bias. See evaluation/walk_forward.py.

CRITICAL: The forward_return column (used to compute the target) is
NEVER saved to disk. It is dropped before the CSV is written.
This prevents fatal look-ahead leakage (Guardrail 1).
"""

import os
import yaml
import pandas as pd
import numpy as np

from features.technical_indicators import compute_all_indicators
from features.vix import load_vix, merge_vix
from features.nifty import load_nifty, compute_nifty_features, merge_nifty


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


def add_target(df: pd.DataFrame, horizon: int = 5, neutral_zone: float = 0.005) -> pd.DataFrame:
    """
    Add 3-class direction target based on N-day forward return with neutral zone.
    
    Labels:
      1.0 (UP)      if forward_return > +neutral_zone
      0.0 (DOWN)    if forward_return < -neutral_zone  
      -1.0 (NEUTRAL) if in between (filtered out during training)
    
    CRITICAL: The forward_return column is computed as an intermediate
    but is DROPPED before returning. Only 'target' survives in the output.
    This prevents the future return from becoming a model feature (Guardrail 1).
    
    The last 'horizon' rows will have NaN target (no future data) → dropped later.
    
    Args:
        df: DataFrame with a 'close' column.
        horizon: Number of days forward to compute the return (default: 5).
        neutral_zone: Threshold for the neutral zone (default: 0.005 = ±0.5%).
    
    Returns:
        DataFrame with 'target' column added. No 'forward_return' column.
    """
    result = df.copy()
    
    # Step 1: Compute the N-day forward percentage return.
    # shift(-horizon) brings the close price from 'horizon' days in the future to today's row.
    # Example: if horizon=5, row for Jan 1 gets the close price from Jan 8.
    forward_return = (result["close"].shift(-horizon) - result["close"]) / result["close"]

    # Step 2: Apply the neutral zone thresholds to create 3-class labels.
    # Moves > +0.5% → UP (1.0), moves < -0.5% → DOWN (0.0), everything else → NEUTRAL (-1.0).
    # NEUTRAL sequences will be filtered out during training by StockSequenceDataset.
    result["target"] = np.where(
        forward_return > neutral_zone, 1.0,
        np.where(forward_return < -neutral_zone, 0.0, -1.0)
    )

    # Step 3: The last 'horizon' rows have no future data → forward_return is NaN.
    # Set their target to NaN so the NaN cleaner drops them.
    result.loc[result.index[-horizon:], "target"] = np.nan

    # GUARDRAIL 1: forward_return is NEVER saved. It was only used to compute the target.
    # We do NOT add it to the DataFrame at all — it stays a local variable.
    # This prevents any possibility of the model seeing future returns as an input feature.

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

    # 4. Load NIFTY50 data and pre-compute its features once
    # (same pattern as VIX — load once, merge per stock)
    nifty_path = config["nifty50"]["file"]
    nifty_df = load_nifty(nifty_path)
    nifty_features_df = compute_nifty_features(nifty_df)
    print(f"NIFTY loaded: {len(nifty_df)} rows, range {nifty_df['date'].min()} to {nifty_df['date'].max()}")

    # 5. Read target parameters from config
    target_horizon = config["target"]["horizon"]
    target_neutral_zone = config["target"]["neutral_zone"]
    print(f"Target: {target_horizon}-day horizon, ±{target_neutral_zone*100:.1f}% neutral zone")

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

        # Step B: Compute the 12 technical indicators (RSI, MACD, stock_volatility_20d, etc.)
        df = compute_all_indicators(df)
        print(f"  After indicators: {len(df)} rows")

        # Step C: Merge the India VIX data and the vix_available binary flag
        df = merge_vix(df, vix_df)
        vix_count = df["vix_available"].sum()
        print(f"  VIX merged: {int(vix_count)} rows with VIX data, {len(df) - int(vix_count)} without")

        # Step C2: Compute VIX change (daily % change of VIX)
        # Only meaningful when VIX data exists. Set to 0.0 when vix_available == 0.
        # Also replace Inf values that occur at the boundary where VIX transitions
        # from 0.0 (pre-2009) to a real value (pct_change divides by zero there).
        df["vix_change"] = df["india_vix"].pct_change().fillna(0.0)
        df["vix_change"] = df["vix_change"].replace([np.inf, -np.inf], 0.0)
        df.loc[df["vix_available"] == 0, "vix_change"] = 0.0

        # Step D: Merge NIFTY50 features (log returns, 5d returns, 20d volatility)
        df = merge_nifty(df, nifty_features_df)
        print(f"  NIFTY merged")

        # Step E: Compute relative features (stock vs market)
        # Relative strength: how much the stock outperformed/underperformed NIFTY today
        # Both are log returns, so subtraction is mathematically correct (Guardrail 4)
        df["relative_strength"] = df["log_return"] - df["nifty_log_return"]

        # Relative volatility: is this stock more or less volatile than the market?
        # Guard against division by zero (if NIFTY vol is 0, set ratio to 1.0)
        df["relative_volatility"] = df["stock_volatility_20d"] / df["nifty_volatility_20d"].replace(0, np.nan)
        df["relative_volatility"] = df["relative_volatility"].fillna(1.0)
        print(f"  Relative features computed")

        # Step F: Add the prediction target (5-day forward return with neutral zone)
        # CRITICAL: forward_return is computed inside add_target() but NEVER saved
        df = add_target(df, horizon=target_horizon, neutral_zone=target_neutral_zone)

        # Step G: Handle NaNs
        # Rolling indicators (like 20-day SMA) produce NaNs for the first ~25 days.
        # The add_target function produces NaN for the last 'horizon' days.
        # NIFTY 5-day return produces NaN for first 5 days.
        # We drop all rows with any NaN.
        nan_before = len(df)
        df = df.dropna().reset_index(drop=True)
        nan_dropped = nan_before - len(df)
        print(f"  Dropped {nan_dropped} NaN rows (rolling warmup + final target rows)")

        # Step H: Final Sanity Check — fail fast if any NaNs slipped through
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            nan_cols = df.isnull().sum()
            raise ValueError(
                f"NaN still present in {name} after processing!\n"
                f"{nan_cols[nan_cols > 0]}"
            )

        # GUARDRAIL 1 CHECK: Verify forward_return is NOT in the DataFrame
        if "forward_return" in df.columns:
            raise ValueError(
                f"FATAL LEAKAGE: 'forward_return' column found in {name} DataFrame! "
                f"This column contains future price data and must never be saved."
            )

        # Log target distribution (3-class: UP, DOWN, NEUTRAL)
        target_dist = df["target"].value_counts()
        up_count = int(target_dist.get(1.0, 0))
        down_count = int(target_dist.get(0.0, 0))
        neutral_count = int(target_dist.get(-1.0, 0))
        total = len(df)
        print(f"  Target distribution: UP={up_count} ({up_count/total*100:.1f}%), "
              f"DOWN={down_count} ({down_count/total*100:.1f}%), "
              f"NEUTRAL={neutral_count} ({neutral_count/total*100:.1f}%)")

        # Step I: Save to disk
        out_path = os.path.join(processed_dir, f"{name.lower()}_features.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

        # Record summary stats for final output
        summaries[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "target_up_pct": round(up_count / total * 100, 1),
            "target_neutral_pct": round(neutral_count / total * 100, 1),
            "vix_coverage_pct": round(vix_count / len(df) * 100, 1),
            "output_path": out_path,
        }

    # Print a nice table of all processed stocks at the end
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE — SUMMARY")
    print(f"{'=' * 60}")
    for name, s in summaries.items():
        print(f"  {name}: {s['rows']} rows, {s['date_range']}, "
              f"UP={s['target_up_pct']}%, NEUTRAL={s['target_neutral_pct']}%, "
              f"VIX coverage={s['vix_coverage_pct']}%")

    # List the final feature set (excluding metadata and target)
    feature_cols = [c for c in df.columns if c not in ["date", "target"]]
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    return summaries


if __name__ == "__main__":
    run_pipeline()
