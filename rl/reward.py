"""
Sortino-based reward calculations for the RL agent.
"""

import numpy as np

def compute_step_reward(
    daily_return: float,
    position: int,
    p_up: float,
    confidence: float,
    actual_stock_return: float,
    action: int,
    conf_threshold: float = 0.6
) -> float:
    """
    Computes a scaled, position-aware reward that includes opportunity cost.
    
    Args:
        daily_return: The portfolio's return over the current step.
        position: The agent's position during the step (0=flat, 1=long).
        p_up: Stage 1 model's probability of UP move.
        confidence: Stage 1 model's confidence score.
        actual_stock_return: The actual return of the underlying stock over the step.
        action: The action taken by the agent (0=HOLD, 1=BUY, 2=SELL).
        conf_threshold: The confidence threshold to consider a signal strong.
        
    Returns:
        The scaled reward scalar.
    """
    # 1. Base Portfolio Return Reward
    # Scale up daily returns (typically ~0.01) by 100 so PPO gradients don't vanish.
    base_reward = daily_return * 100.0
        
    # 2. Opportunity Cost (for staying flat when the model predicted UP)
    opp_cost = 0.0
    if position == 0 and p_up > 0.5 and confidence >= conf_threshold:
        # The model predicted UP with high confidence, but agent stayed flat.
        # If the stock ACTUALLY went up, penalize the agent heavily.
        if actual_stock_return > 0:
            opp_cost = -actual_stock_return * 100.0
            
    # 3. Action Bonus (to overcome initial transaction costs during exploration)
    # Give a massive immediate reward for following a high-confidence signal
    action_bonus = 0.0
    if action == 1 and p_up > 0.5 and confidence >= conf_threshold:
        action_bonus = 1.0  # Massive bonus to force the agent to try trading
    elif action == 2 and p_up <= 0.5:
        action_bonus = 0.5  # Bonus for closing bad positions
        
    return base_reward + opp_cost + action_bonus


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
