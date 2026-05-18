"""
NIFTY50 index merge logic.

Merges NIFTY50 index data with individual stock data to provide
market context. This lets the model distinguish between:
- "This stock went up because the WHOLE market went up" (not real alpha)
- "This stock went up DESPITE the market dropping" (real relative strength)

Features computed:
- nifty_log_return: Daily log return of NIFTY50 (same math as stock log_return)
- nifty_log_return_5d: 5-day rolling log return of NIFTY50
- nifty_volatility_20d: 20-day rolling std of nifty_log_return

Pattern: Follows the same load/merge architecture as vix.py.
"""

import pandas as pd
import numpy as np


def load_nifty(nifty_path: str) -> pd.DataFrame:
    """
    Load and clean NIFTY50 index data.

    Args:
        nifty_path: Path to nifty50_daily.csv.

    Returns:
        DataFrame with columns [date, nifty_close] sorted by date.
    """
    # 1. Read the raw NIFTY50 CSV (same Upstox format as stock CSVs)
    nifty_df = pd.read_csv(nifty_path)

    # 2. Extract the date part, stripping timezone info for clean merging
    nifty_df["date"] = pd.to_datetime(nifty_df["timestamp"]).dt.date
    nifty_df["date"] = pd.to_datetime(nifty_df["date"])

    # 3. Keep only date and close price, rename to avoid column collision with stock 'close'
    nifty_df = nifty_df[["date", "close"]].rename(columns={"close": "nifty_close"})

    # 4. Safety: remove duplicate dates (keep first)
    nifty_df = nifty_df.drop_duplicates(subset="date", keep="first")

    # 5. Sort chronologically for time-series integrity
    nifty_df = nifty_df.sort_values("date").reset_index(drop=True)

    return nifty_df


def compute_nifty_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NIFTY-derived features from the raw NIFTY close prices.

    All returns use LOG returns to be mathematically consistent with
    the stock's log_return (Guardrail 4: consistent return math).

    Args:
        nifty_df: DataFrame from load_nifty() with [date, nifty_close].

    Returns:
        DataFrame with [date, nifty_log_return, nifty_log_return_5d, nifty_volatility_20d].
    """
    result = nifty_df.copy()

    # Daily log return of NIFTY — same formula as stock log_return in technical_indicators.py
    # Using log returns ensures we can subtract stock_log_return - nifty_log_return coherently
    result["nifty_log_return"] = np.log(result["nifty_close"] / result["nifty_close"].shift(1))

    # 5-day log return — captures the weekly trend of the broader market
    result["nifty_log_return_5d"] = np.log(result["nifty_close"] / result["nifty_close"].shift(5))

    # 20-day rolling volatility — measures how jittery the overall market has been recently
    result["nifty_volatility_20d"] = result["nifty_log_return"].rolling(window=20).std()

    # Drop the raw close price — the model should only see derived features, not raw NIFTY prices
    # (raw prices are non-stationary and would mess up normalization)
    result = result.drop(columns=["nifty_close"])

    return result


def merge_nifty(stock_df: pd.DataFrame, nifty_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-computed NIFTY features with stock data.

    Uses a left join so every stock date is preserved. For dates before
    NIFTY data starts (shouldn't happen since NIFTY starts year 2000),
    features will be NaN and get dropped later by the NaN cleaner.

    Args:
        stock_df: DataFrame with a "date" column (datetime).
        nifty_features_df: DataFrame from compute_nifty_features().

    Returns:
        stock_df with added columns: nifty_log_return, nifty_log_return_5d, nifty_volatility_20d.
    """
    # 1. Copy to avoid mutating the original DataFrame
    result = stock_df.copy()

    # 2. Left join on date — keeps all stock rows, adds NIFTY features where available
    result = result.merge(nifty_features_df, on="date", how="left")

    return result
