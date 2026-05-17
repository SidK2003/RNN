"""
Technical indicators computed from raw OHLCV data.

All indicators use only past data (no look-ahead bias).
Each function takes a DataFrame with columns: open, high, low, close, volume
and returns a Series or DataFrame with the computed indicator(s).
"""

import numpy as np
import pandas as pd


# =============================================
# PRICE-BASED INDICATORS
# =============================================

def log_return(close: pd.Series) -> pd.Series:
    """Log return: log(close[t] / close[t-1])."""
    # Computes the natural logarithm of the ratio between today's close and yesterday's close.
    # This transforms multiplicative price changes into additive, stationary returns.
    return np.log(close / close.shift(1))


def bollinger_pctb(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Bollinger %B: (close - SMA) / (2 * rolling_std).
    
    Measures where the price is relative to the Bollinger Bands.
    Values > 1 = above upper band, < 0 = below lower band.
    """
    # 1. Calculate the Simple Moving Average (SMA) over the specified window
    sma = close.rolling(window=window).mean()
    
    # 2. Calculate the rolling standard deviation over the same window
    std = close.rolling(window=window).std()
    
    # 3. Calculate %B. It tells us how far the price has deviated from the SMA
    # relative to the standard deviation bands (usually set at 2 std devs).
    return (close - sma) / (2 * std)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR) — 14-day rolling.
    
    Measures volatility as the average of true ranges.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    """
    # 1. Shift the close price to get yesterday's close
    prev_close = close.shift(1)
    
    # 2. Calculate the three components of True Range
    tr1 = high - low                             # Current day's high-low range
    tr2 = (high - prev_close).abs()              # Gap between yesterday's close and today's high
    tr3 = (low - prev_close).abs()               # Gap between yesterday's close and today's low
    
    # 3. The True Range is the maximum of these three components for each day
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 4. ATR is the rolling simple moving average of the True Range
    return true_range.rolling(window=window).mean()


# =============================================
# MOMENTUM INDICATORS
# =============================================

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) — 14-day.
    
    Uses exponential moving average (Wilder's smoothing) for
    average gains/losses, matching the standard RSI formula.
    Output range: [0, 100].
    """
    # 1. Calculate daily price differences
    delta = close.diff()
    
    # 2. Separate the gains (positive differences) and losses (negative differences)
    # where() replaces values that don't meet the condition with 0.0
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # 3. Calculate the Exponential Moving Average of gains and losses
    # We use Wilder's smoothing method (alpha=1/window)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    # 4. Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # 5. Convert RS into an oscillator between 0 and 100
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> pd.DataFrame:
    """
    MACD, Signal line, and Histogram.
    
    - MACD = EMA(12) - EMA(26)
    - Signal = EMA(9) of MACD
    - Histogram = MACD - Signal
    
    Returns DataFrame with columns: macd, macd_signal, macd_histogram.
    """
    # 1. Calculate the fast (12-day) and slow (26-day) Exponential Moving Averages
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    
    # 2. The MACD line is the difference between the fast and slow EMAs
    macd_line = ema12 - ema26
    
    # 3. The Signal line is a 9-day EMA of the MACD line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # 4. The Histogram represents the divergence between MACD and its Signal line
    histogram = macd_line - signal_line

    # Return all three components as they each contain predictive value
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).
    
    - %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    - %D = SMA(%K, 3)
    
    Output range: [0, 100].
    """
    # 1. Find the lowest low and highest high over the lookback window (e.g., 14 days)
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    # 2. Calculate %K. It measures where the current close is relative to the recent range
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # 3. Calculate %D, which is just a Simple Moving Average of %K (acts as a signal line)
    stoch_d = stoch_k.rolling(window=d_window).mean()

    return pd.DataFrame({
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    })


# =============================================
# VOLUME INDICATORS
# =============================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV).
    
    Cumulative sum of volume where:
    - Volume is added on up days (close > prev_close)
    - Volume is subtracted on down days (close < prev_close)
    - No change on flat days
    """
    # 1. Determine the direction of the price movement (+1 for up, -1 for down, 0 for flat)
    direction = np.sign(close.diff())
    
    # 2. Multiply daily volume by the direction (giving signed volume), then take cumulative sum
    return (direction * volume).cumsum()


def volume_sma_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Volume SMA Ratio: volume / SMA(volume, 20).
    
    Values > 1 indicate above-average volume.
    """
    # 1. Calculate the moving average of the volume
    sma = volume.rolling(window=window).mean()
    
    # 2. Divide today's volume by the average. Captures volume spikes/anomalies.
    return volume / sma


# =============================================
# MASTER FUNCTION
# =============================================

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from raw OHLCV data.
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].
            Must be sorted by timestamp ascending.
    
    Returns:
        DataFrame with original columns + all computed indicator columns.
        Does NOT drop NaN rows — caller handles that.
    """
    # Create a copy so we don't mutate the original raw DataFrame passed in
    result = df.copy()

    # --- Compute Price-based Indicators ---
    result["log_return"] = log_return(df["close"])
    result["bollinger_pctb"] = bollinger_pctb(df["close"], window=20)
    result["atr"] = atr(df["high"], df["low"], df["close"], window=14)

    # --- Compute Momentum Indicators ---
    result["rsi"] = rsi(df["close"], window=14)
    
    # MACD returns a DataFrame with 3 columns, so we concatenate them along the column axis
    macd_df = macd(df["close"])
    result = pd.concat([result, macd_df], axis=1)
    
    # Stochastic returns 2 columns (%K and %D), concatenate them as well
    stoch_df = stochastic(df["high"], df["low"], df["close"], k_window=14, d_window=3)
    result = pd.concat([result, stoch_df], axis=1)

    # --- Compute Volume Indicators ---
    result["obv"] = obv(df["close"], df["volume"])
    result["volume_sma_ratio"] = volume_sma_ratio(df["volume"], window=20)

    # Return the full DataFrame enriched with all the new feature columns
    return result
