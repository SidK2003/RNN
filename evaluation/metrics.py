"""
Risk and performance metrics for the evaluation pipeline.

In simple words: This module is a calculator for financial risk metrics.
Given an array of daily portfolio returns and/or a list of trades,
it computes standard industry metrics that tell us how well a strategy did
and how much risk it took to get there.

All functions are PURE — no side effects, no file I/O, no state.
They take numbers in, return numbers out.

Convention: Indian markets have ~250 trading days per year.
All annualized metrics use sqrt(250) or 250 as appropriate.
"""

import numpy as np
from typing import Dict, List, Optional


# =============================================
# CONSTANTS
# =============================================

# Indian equity markets trade ~250 days per year.
# Used for annualizing returns and volatility.
TRADING_DAYS_PER_YEAR = 250


# =============================================
# CORE RISK METRICS
# =============================================

def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Sortino Ratio — our PRIMARY metric.
    
    Like Sharpe, but only penalizes DOWNSIDE volatility.
    A strategy that has big gains and small losses gets a high Sortino,
    even if its total volatility is high. This is exactly what we want
    for a trading system: big wins, small losses.
    
    Formula: (mean_excess_return * sqrt(250)) / annualized_downside_std
    
    Args:
        returns: Array of daily portfolio returns (e.g., [0.01, -0.005, 0.003, ...])
        risk_free: Daily risk-free rate (default 0 — we compare against zero, not T-bills)
        
    Returns:
        Annualized Sortino ratio. Higher is better. 0 = no edge. Negative = losing money.
    """
    if len(returns) < 2:
        return 0.0
    
    # Excess returns: how much better than the risk-free rate
    excess = returns - risk_free
    
    # Downside returns: only keep the negative ones (losses)
    downside = np.minimum(excess, 0.0)
    
    # Downside deviation: standard deviation of losses only
    downside_std = np.sqrt(np.mean(downside ** 2))
    
    if downside_std < 1e-8:
        # No downside volatility — either all positive returns or all zeros
        return 0.0
    
    # Annualize: multiply mean by 250 (daily to yearly return),
    # multiply downside_std by sqrt(250) (volatility scales with sqrt of time)
    annualized_return = np.mean(excess) * TRADING_DAYS_PER_YEAR
    annualized_downside = downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(annualized_return / annualized_downside)


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Sharpe Ratio — the classic risk-adjusted return measure.
    
    Unlike Sortino, this penalizes ALL volatility equally (both up and down).
    A strategy with wild swings (even upward) gets penalized. This is more
    conservative than Sortino.
    
    Args:
        returns: Array of daily portfolio returns.
        risk_free: Daily risk-free rate.
        
    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    
    excess = returns - risk_free
    std = np.std(excess, ddof=1)  # ddof=1 for sample standard deviation
    
    if std < 1e-8:
        return 0.0
    
    # Annualize both numerator and denominator
    annualized_return = np.mean(excess) * TRADING_DAYS_PER_YEAR
    annualized_std = std * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return float(annualized_return / annualized_std)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum Drawdown — the worst peak-to-trough decline.
    
    In simple words: If you invested at the best possible time and sold
    at the worst possible time within this period, how much would you lose?
    
    This is the most intuitive risk metric: "What's the worst that happened?"
    
    Args:
        equity_curve: Array of portfolio values over time (NOT returns).
                      e.g., [1.0, 1.02, 0.98, 1.05, ...]
                      
    Returns:
        Max drawdown as a positive fraction (e.g., 0.15 means -15% drawdown).
        Higher is worse.
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Running maximum: the highest portfolio value seen so far at each point
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point: how far below the peak we are
    drawdowns = (running_max - equity_curve) / running_max
    
    return float(np.max(drawdowns))


def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calmar Ratio — annualized return divided by maximum drawdown.
    
    Answers: "For every 1% of worst-case pain, how much annual return do I get?"
    
    Args:
        returns: Array of daily portfolio returns.
        
    Returns:
        Calmar ratio. Higher is better. Undefined if no drawdown (returns 0).
    """
    if len(returns) < 2:
        return 0.0
    
    # Reconstruct equity curve from returns
    equity = np.cumprod(1.0 + returns)
    
    mdd = max_drawdown(equity)
    
    if mdd < 1e-8:
        # No drawdown at all — can't compute ratio
        return 0.0
    
    # Annualized total return
    total_return = equity[-1] / equity[0] - 1.0
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    
    if n_years < 1e-8:
        return 0.0
    
    annualized_return = (1 + total_return) ** (1.0 / n_years) - 1.0
    
    return float(annualized_return / mdd)


# =============================================
# TRADE-LEVEL METRICS
# =============================================

def win_rate(trade_returns: np.ndarray) -> float:
    """
    Win Rate — percentage of trades that were profitable.
    
    Args:
        trade_returns: Array of per-trade returns (one entry per round-trip trade).
                       e.g., [0.02, -0.01, 0.03, -0.005, ...]
        
    Returns:
        Win rate as a fraction [0, 1]. e.g., 0.6 means 60% of trades were winners.
    """
    if len(trade_returns) == 0:
        return 0.0
    
    winners = np.sum(trade_returns > 0)
    return float(winners / len(trade_returns))


def profit_factor(trade_returns: np.ndarray) -> float:
    """
    Profit Factor — gross profits / gross losses.
    
    A profit factor > 1 means the strategy makes more money on winners
    than it loses on losers. A profit factor of 2.0 means for every $1 lost,
    $2 was gained.
    
    Args:
        trade_returns: Array of per-trade returns.
        
    Returns:
        Profit factor. > 1 is profitable. 0 if no losing trades.
    """
    if len(trade_returns) == 0:
        return 0.0
    
    # Sum of all positive returns (gross profit)
    gross_profit = np.sum(trade_returns[trade_returns > 0])
    
    # Sum of all negative returns (gross loss) — take absolute value
    gross_loss = np.abs(np.sum(trade_returns[trade_returns < 0]))
    
    if gross_loss < 1e-10:
        # No losing trades — profit factor is technically infinite
        return float(gross_profit) if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def total_return(equity_curve: np.ndarray) -> float:
    """
    Total Return — simple start-to-end return of the equity curve.
    
    Args:
        equity_curve: Array of portfolio values.
        
    Returns:
        Total return as a fraction (e.g., 0.25 = 25% gain).
    """
    if len(equity_curve) < 2:
        return 0.0
    
    return float(equity_curve[-1] / equity_curve[0] - 1.0)


def annualized_return(returns: np.ndarray) -> float:
    """
    Annualized Return — CAGR equivalent from daily returns.
    
    Args:
        returns: Array of daily portfolio returns.
        
    Returns:
        Annualized return as a fraction.
    """
    if len(returns) < 2:
        return 0.0
    
    # Compound daily returns into total growth
    total_growth = np.prod(1.0 + returns)
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    
    if n_years < 1e-8:
        return 0.0
    
    return float(total_growth ** (1.0 / n_years) - 1.0)


# =============================================
# MASTER FUNCTION
# =============================================

def compute_all_metrics(
    daily_returns: np.ndarray,
    equity_curve: np.ndarray,
    trade_returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all risk and performance metrics in one call.
    
    This is the main entry point used by backtest.py. It computes every
    metric we care about and returns a clean dictionary ready for JSON serialization.
    
    Args:
        daily_returns: Array of daily portfolio returns.
        equity_curve: Array of portfolio values over time.
        trade_returns: Optional array of per-trade (round-trip) returns.
                       If None, trade-level metrics are set to 0.
        
    Returns:
        Dictionary of all metrics with human-readable keys.
    """
    metrics = {
        # Return metrics
        "total_return_pct": round(total_return(equity_curve) * 100, 4),
        "annualized_return_pct": round(annualized_return(daily_returns) * 100, 4),
        
        # Risk-adjusted metrics
        "sortino_ratio": round(sortino_ratio(daily_returns), 4),
        "sharpe_ratio": round(sharpe_ratio(daily_returns), 4),
        "calmar_ratio": round(calmar_ratio(daily_returns), 4),
        
        # Risk metrics
        "max_drawdown_pct": round(max_drawdown(equity_curve) * 100, 4),
        "daily_volatility_pct": round(float(np.std(daily_returns, ddof=1)) * 100, 4),
        
        # Summary stats
        "total_trading_days": len(daily_returns),
        "positive_days": int(np.sum(daily_returns > 0)),
        "negative_days": int(np.sum(daily_returns < 0)),
        "flat_days": int(np.sum(daily_returns == 0)),
    }
    
    # Trade-level metrics (only if trades were provided)
    if trade_returns is not None and len(trade_returns) > 0:
        metrics.update({
            "total_trades": len(trade_returns),
            "win_rate_pct": round(win_rate(trade_returns) * 100, 2),
            "profit_factor": round(profit_factor(trade_returns), 4),
            "avg_trade_return_pct": round(float(np.mean(trade_returns)) * 100, 4),
            "best_trade_pct": round(float(np.max(trade_returns)) * 100, 4),
            "worst_trade_pct": round(float(np.min(trade_returns)) * 100, 4),
        })
    else:
        metrics.update({
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return_pct": 0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
        })
    
    return metrics
