"""
Three-way backtesting engine for the evaluation pipeline.

In simple words: This module answers "Does our system actually make money?"
by running 3 strategies on the EXACT SAME test data and comparing them head-to-head:

1. Buy-and-Hold (Baseline): Buy on day 1, hold forever. The "dumb" strategy.
2. Predictor-Only: Follow Stage 1's predictions naively. No RL agent.
3. Full RL System: Stage 1 predictions fed into the trained RL agent.

Each strategy produces an equity curve, daily returns, a trade log, and a full
set of risk metrics. This allows us to answer:
- Does the predictor add value over Buy-and-Hold?
- Does the RL agent add value over naive prediction following?

Usage:
    This module is called by walk_forward.py — not run standalone.
    
Output:
    Per-strategy dict with equity_curve, daily_returns, trade_log, and metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from evaluation.metrics import compute_all_metrics


# =============================================
# STRATEGY 1: BUY-AND-HOLD (BASELINE)
# =============================================

def run_buy_and_hold(test_df: pd.DataFrame) -> Dict:
    """
    The simplest possible strategy: Buy on day 1, sell on the last day.
    No transaction costs applied (single buy, never traded again).
    
    This is the baseline that any "smart" system must beat.
    If our RL agent can't beat Buy-and-Hold, we should just buy an index fund.
    
    Args:
        test_df: DataFrame with columns [date, close, india_vix, p_up, confidence]
        
    Returns:
        Dictionary with equity_curve, daily_returns, trade_log, and metrics.
    """
    close_prices = test_df["close"].values
    dates = test_df["date"].values
    
    # Equity curve: portfolio value relative to day 1
    # If stock goes from 1000 to 1200, equity = [1.0, ..., 1.2]
    equity_curve = close_prices / close_prices[0]
    
    # Daily returns: percentage change day-over-day
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    
    # Trade log: just one buy on day 1 (for visualization)
    trade_log = [
        {"date": str(dates[0]), "action": "BUY", "price": float(close_prices[0]), 
         "portfolio_value": 1.0}
    ]
    
    # Compute all metrics
    metrics = compute_all_metrics(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        trade_returns=None  # No round-trip trades to report
    )
    
    return {
        "strategy": "Buy-and-Hold",
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "trade_log": trade_log,
        "dates": dates,
        "metrics": metrics,
    }


# =============================================
# STRATEGY 2: PREDICTOR-ONLY
# =============================================

def run_predictor_only(
    test_df: pd.DataFrame,
    config: dict,
    confidence_threshold: float = 0.6,
) -> Dict:
    """
    Follow Stage 1's predictions naively:
    - BUY when p_up > 0.5 AND confidence >= threshold AND currently flat
    - SELL when p_up <= 0.5 AND currently long
    - HOLD otherwise
    
    Transaction costs are applied on every trade (same rates as the RL env).
    This tests: "What if we just did what the model says without RL?"
    
    Args:
        test_df: DataFrame with [date, close, india_vix, p_up, confidence]
        config: Project config dict (for transaction cost rates)
        confidence_threshold: Minimum confidence to allow trading
        
    Returns:
        Dictionary with equity_curve, daily_returns, trade_log, and metrics.
    """
    close_prices = test_df["close"].values
    dates = test_df["date"].values
    p_up_values = test_df["p_up"].values
    confidence_values = test_df["confidence"].values
    
    # Transaction cost rates from config
    brokerage = config["costs"]["brokerage_pct"]
    stt = config["costs"]["stt_pct"]
    slippage = config["costs"]["slippage_pct"]
    
    # State tracking
    portfolio_value = 1.0
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    
    # Output arrays
    equity_curve = np.zeros(len(close_prices))
    equity_curve[0] = portfolio_value
    
    trade_log = []
    trade_returns = []  # Per round-trip trade returns
    
    for i in range(1, len(close_prices)):
        p_up = p_up_values[i]
        conf = confidence_values[i]
        
        # --- Decision logic ---
        if position == 0 and p_up > 0.5 and conf >= confidence_threshold:
            # BUY signal: model is confident the stock goes up
            position = 1
            entry_price = close_prices[i]
            # Deduct buy-side costs (brokerage + slippage)
            cost = brokerage + slippage
            portfolio_value *= (1.0 - cost)
            trade_log.append({
                "date": str(dates[i]), "action": "BUY", 
                "price": float(close_prices[i]),
                "portfolio_value": float(portfolio_value)
            })
            
        elif position == 1 and p_up <= 0.5:
            # SELL signal: model says stock not going up anymore
            # Calculate return from the trade (price movement since buy)
            price_return = (close_prices[i] - entry_price) / entry_price
            portfolio_value *= (1.0 + price_return)
            
            # Deduct sell-side costs (brokerage + slippage + STT)
            cost = brokerage + slippage + stt
            portfolio_value *= (1.0 - cost)
            
            # Record the round-trip trade return
            trade_returns.append(float(portfolio_value / equity_curve[
                max(0, len([t for t in trade_log if t["action"] == "BUY"]) - 1)
            ] - 1.0) if len(trade_log) > 0 else 0.0)
            
            position = 0
            entry_price = 0.0
            trade_log.append({
                "date": str(dates[i]), "action": "SELL",
                "price": float(close_prices[i]),
                "portfolio_value": float(portfolio_value)
            })
            
        elif position == 1:
            # Holding: update portfolio value with daily price movement
            daily_stock_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            portfolio_value *= (1.0 + daily_stock_return)
        
        # Record equity value for this day
        equity_curve[i] = portfolio_value
    
    # Calculate daily returns from equity curve
    daily_returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    
    # Compute trade returns properly: from BUY equity to SELL equity
    trade_returns_clean = _compute_trade_returns(trade_log)
    
    # Compute all metrics
    metrics = compute_all_metrics(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        trade_returns=np.array(trade_returns_clean) if trade_returns_clean else None,
    )
    
    return {
        "strategy": "Predictor-Only",
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "trade_log": trade_log,
        "dates": dates,
        "metrics": metrics,
    }


# =============================================
# STRATEGY 3: FULL RL SYSTEM
# =============================================

def run_full_rl(
    test_df: pd.DataFrame,
    config: dict,
    rl_model_path: str,
) -> Dict:
    """
    The Full System: Stage 1 predictions → Stage 2 RL agent decisions.
    
    Loads the trained MaskablePPO agent and lets it make all trading decisions.
    Transaction costs are embedded in the TradingEnv (same as during training).
    
    Args:
        test_df: DataFrame with [date, close, india_vix, p_up, confidence]
        config: Project config dict
        rl_model_path: Path to the saved RL model (.zip file)
        
    Returns:
        Dictionary with equity_curve, daily_returns, trade_log, and metrics.
    """
    # Import here to avoid circular imports and heavy loading at module level
    from sb3_contrib import MaskablePPO
    from rl.trading_env import TradingEnv
    
    # Load the trained RL agent
    agent = MaskablePPO.load(rl_model_path)
    
    # Create a fresh TradingEnv with the test data
    env = TradingEnv(test_df, config)
    
    # Reset and run the episode
    obs, info = env.reset()
    
    # Track equity and actions at every step
    equity_values = [env.portfolio_value]
    dates_list = [info["date"]]
    trade_log = []
    
    done = False
    while not done:
        # The RL agent decides what to do, respecting action masks
        action_masks = env.action_masks()
        action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
        
        # Convert numpy action to int (SB3 returns numpy)
        action = int(action)
        
        # Record trade if BUY or SELL
        if action == TradingEnv.BUY or action == TradingEnv.SELL:
            action_name = "BUY" if action == TradingEnv.BUY else "SELL"
            trade_log.append({
                "date": str(info["date"]),
                "action": action_name,
                "price": float(env.df.loc[env.current_step, "close"]),
                "portfolio_value": float(env.portfolio_value),
            })
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record equity value after the step
        equity_values.append(env.portfolio_value)
        dates_list.append(info["date"])
    
    # Convert to arrays
    equity_curve = np.array(equity_values)
    dates_arr = np.array(dates_list)
    
    # Calculate daily returns from equity curve
    daily_returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    
    # Compute per-trade returns from trade log
    trade_returns = _compute_trade_returns(trade_log)
    
    # Compute all metrics
    metrics = compute_all_metrics(
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        trade_returns=np.array(trade_returns) if trade_returns else None,
    )
    
    return {
        "strategy": "Full-RL",
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "trade_log": trade_log,
        "dates": dates_arr,
        "metrics": metrics,
    }


# =============================================
# UTILITY FUNCTIONS
# =============================================

def _compute_trade_returns(trade_log: List[Dict]) -> List[float]:
    """
    Extract per-trade (round-trip) returns from a trade log.
    
    A round-trip is BUY followed by SELL. The return is computed as:
    (sell_portfolio_value - buy_portfolio_value) / buy_portfolio_value
    
    Args:
        trade_log: List of trade dicts with 'action' and 'portfolio_value' keys.
        
    Returns:
        List of per-trade returns (one entry per BUY-SELL pair).
    """
    trade_returns = []
    buy_value = None
    
    for trade in trade_log:
        if trade["action"] == "BUY":
            buy_value = trade["portfolio_value"]
        elif trade["action"] == "SELL" and buy_value is not None:
            sell_value = trade["portfolio_value"]
            trade_return = (sell_value - buy_value) / buy_value
            trade_returns.append(trade_return)
            buy_value = None  # Reset for next round-trip
    
    return trade_returns


def run_all_strategies(
    test_df: pd.DataFrame,
    config: dict,
    rl_model_path: str,
    confidence_threshold: float = 0.6,
) -> Dict[str, Dict]:
    """
    Run all 3 strategies on the same test data and return their results.
    
    This is the main entry point called by walk_forward.py.
    
    Args:
        test_df: DataFrame with [date, close, india_vix, p_up, confidence]
        config: Project config dict
        rl_model_path: Path to saved RL model (.zip)
        confidence_threshold: For predictor-only strategy
        
    Returns:
        Dict mapping strategy name to its results dict.
    """
    results = {}
    
    # 1. Buy-and-Hold
    print("    Running Buy-and-Hold...")
    results["buy_and_hold"] = run_buy_and_hold(test_df)
    
    # 2. Predictor-Only
    print("    Running Predictor-Only...")
    results["predictor_only"] = run_predictor_only(test_df, config, confidence_threshold)
    
    # 3. Full RL System
    print("    Running Full RL System...")
    results["full_rl"] = run_full_rl(test_df, config, rl_model_path)
    
    return results


def generate_tearsheet(
    daily_returns: np.ndarray,
    dates: np.ndarray,
    benchmark_returns: np.ndarray,
    output_path: str,
    title: str = "RL Strategy Tearsheet",
) -> None:
    """
    Generate a quantstats HTML tearsheet comparing the RL strategy to Buy-and-Hold.
    
    This is a nice-to-have visual report — the heavy lifting is done by quantstats.
    If quantstats fails (unstable API), we catch the error and skip gracefully.
    
    Args:
        daily_returns: RL strategy daily returns.
        dates: Array of dates for the returns.
        benchmark_returns: Buy-and-Hold daily returns (the benchmark).
        output_path: Where to save the HTML file.
        title: Title for the tearsheet.
    """
    try:
        import quantstats as qs
        
        # quantstats expects a pandas Series with DatetimeIndex.
        # The RL strategy and Buy-and-Hold may have different lengths because
        # TradingEnv starts at step 5 (needs history for recent_returns).
        # We align both to the shorter length to avoid index mismatch.
        min_len = min(len(daily_returns), len(benchmark_returns))
        
        # Use the RL dates for both (shorter series — starts later due to env warmup)
        aligned_dates = pd.to_datetime(dates[1:min_len + 1])
        
        returns_series = pd.Series(
            daily_returns[:min_len],
            index=aligned_dates,
            name="Strategy"
        )
        benchmark_series = pd.Series(
            benchmark_returns[:min_len],
            index=aligned_dates,
            name="Benchmark"
        )
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate the HTML report
        qs.reports.html(
            returns_series,
            benchmark=benchmark_series,
            output=output_path,
            title=title,
        )
        print(f"    Tearsheet saved: {output_path}")
        
    except Exception as e:
        # quantstats has a fragile API — don't crash the whole pipeline if it fails
        print(f"    WARNING: Could not generate tearsheet: {e}")


def print_comparison_table(results: Dict[str, Dict]) -> None:
    """
    Print a formatted comparison table of all strategies to the console.
    
    Args:
        results: Dict mapping strategy name to results dict (from run_all_strategies).
    """
    print("\n" + "=" * 85)
    print(f"{'Strategy':<18} {'Return%':>10} {'Sortino':>10} {'Sharpe':>10} "
          f"{'MaxDD%':>10} {'WinRate%':>10} {'Trades':>8}")
    print("-" * 85)
    
    for key, res in results.items():
        m = res["metrics"]
        print(f"{res['strategy']:<18} "
              f"{m['total_return_pct']:>10.2f} "
              f"{m['sortino_ratio']:>10.3f} "
              f"{m['sharpe_ratio']:>10.3f} "
              f"{m['max_drawdown_pct']:>10.2f} "
              f"{m['win_rate_pct']:>10.1f} "
              f"{m['total_trades']:>8d}")
    
    print("=" * 85)
