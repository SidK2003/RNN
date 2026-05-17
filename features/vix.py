"""
India VIX merge logic.

Merges India VIX data with stock data on date.
For dates before VIX data availability (~2009), adds a binary
`vix_available` flag so the model can distinguish between
"VIX is low" vs "VIX data doesn't exist for this date".
"""

import pandas as pd
import numpy as np


def load_vix(vix_path: str) -> pd.DataFrame:
    """
    Load and clean India VIX data.
    
    Args:
        vix_path: Path to india_vix_daily.csv.
    
    Returns:
        DataFrame with columns [date, india_vix] sorted by date.
    """
    # 1. Read the raw VIX CSV into a pandas DataFrame
    vix_df = pd.read_csv(vix_path)

    # 2. Extract only the date part from the timestamp string
    # VIX timestamps usually look like "2009-03-02 09:15:00". We strip the time 
    # to allow merging with the daily stock data.
    vix_df["date"] = pd.to_datetime(vix_df["timestamp"]).dt.date
    vix_df["date"] = pd.to_datetime(vix_df["date"])

    # 3. Filter columns to keep only the date and the closing value of VIX
    # We rename 'close' to 'india_vix' to avoid column name collisions later.
    vix_df = vix_df[["date", "close"]].rename(columns={"close": "india_vix"})

    # 4. Remove any duplicate dates (keeping the first occurrence)
    # This acts as a safety measure against bad data from the API.
    vix_df = vix_df.drop_duplicates(subset="date", keep="first")

    # 5. Sort the dataframe chronologically to ensure time-series integrity
    vix_df = vix_df.sort_values("date").reset_index(drop=True)

    return vix_df


def merge_vix(stock_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VIX data with stock data.
    
    For dates where VIX data is available: india_vix = actual VIX value, vix_available = 1.
    For dates before VIX availability: india_vix = 0, vix_available = 0.
    
    This approach lets the model learn that the VIX feature is meaningless
    when vix_available=0, rather than hallucinating that VIX is actually 0.
    
    Args:
        stock_df: DataFrame with a "date" column (datetime).
        vix_df: DataFrame from load_vix() with [date, india_vix].
    
    Returns:
        stock_df with added columns: india_vix, vix_available.
    """
    # 1. Create a copy of the stock dataframe to avoid mutating the original
    result = stock_df.copy()

    # 2. Perform a left join on the 'date' column
    # This keeps every row from the stock dataframe. If VIX data doesn't exist
    # for a particular stock date (e.g., pre-2009), the 'india_vix' column will be NaN.
    result = result.merge(vix_df, on="date", how="left")

    # 3. Create a binary flag indicating if VIX data was successfully joined
    # 1 means VIX data is real. 0 means it's missing (NaN).
    result["vix_available"] = result["india_vix"].notna().astype(int)

    # 4. Fill the missing VIX values with 0.0
    # Since we have the 'vix_available' flag, the neural network can learn to 
    # ignore the 'india_vix' feature when 'vix_available' is 0.
    result["india_vix"] = result["india_vix"].fillna(0.0)

    return result
