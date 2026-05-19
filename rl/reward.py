"""
Sortino-based reward calculations for the RL agent.
"""

import numpy as np

def compute_step_reward(daily_return: float) -> float:
    """
    Computes the asymmetric step reward.
    Penalizes losses 2x more than it rewards gains (Sortino-style penalty).
    
    Args:
        daily_return: The portfolio's return over the current step.
        
    Returns:
        The reward scalar.
    """
    if daily_return >= 0:
        return daily_return
    return daily_return * 2.0


def compute_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Computes the episode-level Sortino ratio for logging purposes.
    
    Args:
        returns: Array of daily returns.
        risk_free: Risk-free rate per step (default 0).
        
    Returns:
        Sortino ratio scalar.
    """
    if len(returns) < 2:
        return 0.0
        
    excess = returns - risk_free
    downside = np.minimum(excess, 0)
    downside_std = np.sqrt(np.mean(downside**2))
    
    if downside_std < 1e-8:
        return 0.0
        
    return float(np.mean(excess) / downside_std)
