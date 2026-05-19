"""
Custom Gymnasium environment for the Stage 2 RL trading agent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from rl.reward import compute_step_reward

class TradingEnv(gym.Env):
    """
    Custom Trading Environment that follows gymnasium interface.
    
    Observation space (11-dim vector):
        - p_up: probability of up move from Stage 1 [0, 1]
        - confidence: Stage 1 confidence score [0.5, 1.0]
        - position: -1, 0, 1 (currently only 0 and 1 used)
        - unrealised_pnl: normalized P&L of current position
        - days_in_position: normalized holding duration
        - india_vix: normalized VIX value
        - recent_returns (5 days): portfolio returns over last 5 days
        
    Action space (Discrete(3)):
        0 = HOLD
        1 = BUY
        2 = SELL
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Action constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, prediction_df: pd.DataFrame, config: dict):
        super().__init__()
        
        self.df = prediction_df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        
        # Transaction costs
        self.brokerage = config["costs"]["brokerage_pct"]
        self.stt = config["costs"]["stt_pct"]
        self.slippage = config["costs"]["slippage_pct"]
        
        # Action Space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 10 features, bounded where appropriate but generally open to float32
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 5  # start at 5 to have history for recent_returns
        self.portfolio_value = 1.0
        self.position = 0  # 0=flat, 1=long
        self.entry_price = 0.0
        self.days_in_position = 0
        
        # Tracking history for state and logging
        self.history_returns = np.zeros(self.max_steps + 1)
        self.action_history = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        
        self.current_step = 5
        self.portfolio_value = 1.0
        self.position = 0
        self.entry_price = 0.0
        self.days_in_position = 0
        
        self.history_returns.fill(0.0)
        self.action_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""
        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        current_price = self.df.loc[self.current_step, "close"]
        prev_portfolio_value = self.portfolio_value
        
        trade_executed = False
        
        # 1. Execute Action
        if action == self.BUY and self.position == 0:
            # Go Long
            self.position = 1
            # Deduct brokerage + slippage from portfolio
            cost = self.brokerage + self.slippage
            self.portfolio_value *= (1 - cost)
            self.entry_price = current_price
            self.days_in_position = 0
            trade_executed = True
            
        elif action == self.SELL and self.position == 1:
            # Close Long
            self.position = 0
            # Deduct brokerage + slippage + STT from portfolio
            cost = self.brokerage + self.slippage + self.stt
            self.portfolio_value *= (1 - cost)
            self.entry_price = 0.0
            self.days_in_position = 0
            trade_executed = True
            
        elif action == self.HOLD:
            pass  # Do nothing
        else:
            # Action mask failed or invalid action (e.g. SELL when flat)
            # We silently treat as HOLD, but action_masks() should prevent this.
            action = self.HOLD

        # Log action
        self.action_history.append(action)

        # 2. Advance Time
        self.current_step += 1
        next_price = self.df.loc[self.current_step, "close"]
        
        # 3. Calculate new portfolio value based on price movement
        if self.position == 1:
            # Portfolio value scales with the daily return of the stock
            daily_stock_return = (next_price - current_price) / current_price
            self.portfolio_value *= (1 + daily_stock_return)
            self.days_in_position += 1

        # 4. Calculate Step Reward
        daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.history_returns[self.current_step] = daily_return
        
        reward = compute_step_reward(daily_return)

        # 5. Check Termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 6. Gather Observation and Info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add logging metrics at end of episode
        if terminated:
            # Calculate % of holds
            holds = sum(1 for a in self.action_history if a == self.HOLD)
            info["episode"] = {
                "r": sum(self.history_returns), # simple sum of returns roughly
                "l": self.current_step,
                "hold_pct": holds / max(1, len(self.action_history)),
                "total_trades": sum(1 for a in self.action_history if a in [self.BUY, self.SELL]) // 2,
                "final_portfolio_value": self.portfolio_value
            }

        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return boolean array indicating which actions are valid.
        Used by sb3-contrib MaskablePPO.
        [HOLD, BUY, SELL]
        """
        # Always can hold
        mask = [True, False, False]
        
        if self.position == 0:
            # Flat: can BUY
            mask[self.BUY] = True
        elif self.position == 1:
            # Long: can SELL
            mask[self.SELL] = True
            
        return np.array(mask, dtype=bool)

    def _get_observation(self) -> np.ndarray:
        """Construct the 11-dim observation vector."""
        row = self.df.loc[self.current_step]
        
        # 1. p_up
        p_up = row["p_up"]
        
        # 2. confidence
        confidence = row["confidence"]
        
        # 3. position
        position = self.position
        
        # 4. unrealised_pnl (clipped to +/- 0.2)
        if self.position == 1 and self.entry_price > 0:
            upnl = (row["close"] - self.entry_price) / self.entry_price
            upnl = np.clip(upnl, -0.2, 0.2)
        else:
            upnl = 0.0
            
        # 5. days_in_position (normalized to ~60 days max)
        dip_norm = min(self.days_in_position / 60.0, 1.0)
        
        # 6. india_vix (normalized roughly by historical median of ~50)
        vix_norm = row["india_vix"] / 50.0
        
        # 7-11. recent_returns (last 5 days)
        # We start at step=5, so this is always safe
        recent = self.history_returns[self.current_step-5:self.current_step]
        
        obs = np.array([
            p_up,
            confidence,
            position,
            upnl,
            dip_norm,
            vix_norm,
            *recent
        ], dtype=np.float32)
        
        return obs

    def _get_info(self) -> dict:
        return {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "date": self.df.loc[self.current_step, "date"]
        }
