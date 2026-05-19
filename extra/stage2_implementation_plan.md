# Phase 3: RL Environment & Agent — Technical Review & Revised Plan

## Current State (Confirmed)

All 5 stocks fully trained and evaluated with Optuna-optimized hyperparameters. Phase 2.75 is complete.

| Stock | Windows | Accuracy | Baseline | Lift | MCC |
|---|---|---|---|---|---|
| RELIANCE | 8 | 51.3% | 54.3% | -3.0% | +0.050 |
| TCS | 6 | 53.4% | 54.6% | -1.2% | +0.062 |
| HDFCBANK | 7 | 53.1% | 55.7% | -2.6% | -0.002 |
| HINDUNILVR | 7 | 48.5% | 54.8% | -6.3% | -0.009 |
| SUNPHARMA | 7 | 51.3% | 55.0% | -3.7% | +0.042 |

---

## Technical Review of the Original RL Design

I've reviewed the `final_implementation_plan.md` RL spec, all Stage 1 code, the metrics, and the System Theory doc. Here are the issues I found:

---

### Issue 1: OOF Data Scarcity — The Most Critical Problem

The original plan says to train Stage 1 with an 80/20 split within each walk-forward window, then use Stage 1's predictions on the 20% validation portion as "out-of-fold" (OOF) data to train the RL agent.

**The problem:** The 20% validation split of a 10-year window is only ~2 years ≈ **500 trading days**. That is ONE episode for the RL agent. With `total_timesteps=200,000` and `n_steps=1024`, PPO needs to replay that same 500-step sequence **~400 times**. The agent will memorize the exact sequence of market moves — a severe form of overfitting.

Worse: `n_steps=1024` in `config.yaml` is larger than the episode length (500). PPO's rollout buffer can't even fill up in a single episode. It would need to collect partial rollouts across multiple episode resets, and each reset replays the same 500 days.

**Proposed fix — 3-fold temporal CV:**
Instead of a single 80/20 split, use 3 temporal folds within the 10-year training window:

```
Fold 1: Train Stage 1 on years 1-7,  predict years 8-10  → ~750 OOF days
Fold 2: Train Stage 1 on years 1-3 + 7-10, predict years 4-6  → ~750 OOF days
Fold 3: Train Stage 1 on years 4-10, predict years 1-3  → ~750 OOF days
                                                    Total: ~2,250 OOF days
```

Wait — fold 3 violates temporal ordering (predicting the past with future data). That's look-ahead bias.

**Corrected approach — expanding-window OOF:**
```
Fold 1: Train Stage 1 on years 1-4,  predict years 5-6   → ~500 OOF days
Fold 2: Train Stage 1 on years 1-6,  predict years 7-8   → ~500 OOF days
Fold 3: Train Stage 1 on years 1-8,  predict years 9-10  → ~500 OOF days
                                                    Total: ~1,500 OOF days
```

Each fold trains Stage 1 on only past data and predicts forward. We concatenate the 3 OOF prediction sequences. The RL agent now has ~1,500 steps across 3 episodes, covering different market regimes. With `total_timesteps=200,000`, that's ~133 replays — still a lot, but 3× better than before, and across diverse market conditions.

> [!IMPORTANT]
> This is extra compute: we train **3 throwaway Stage 1 models** per walk-forward window just to generate OOF predictions. Each takes ~2-3 minutes on the RTX 4070 with current hyperparameters. For 8 windows, that's ~8 × 3 × 2.5min = **~60 extra minutes**. Acceptable.

---

### Issue 2: Double Punishment for Trading

The original reward function:
```python
# Transaction costs already deducted from portfolio value before this
daily_return = (portfolio_value[t] - portfolio_value[t-1]) / portfolio_value[t-1]
reward = daily_return if daily_return >= 0 else daily_return * 2.0
if action in [BUY, SELL]:
    reward -= 0.001  # explicit penalty ON TOP of transaction costs
```

Transaction costs are already deducted from portfolio value (0.1% brokerage + 0.05% slippage per side = ~0.31% round-trip). The `daily_return` already reflects this cost. Adding another `-0.001` penalty per trade means the agent is punished **twice**.

A round-trip would cost: 0.31% (from portfolio) + 0.2% (two × 0.001 explicit penalty) = **0.51% total effective cost**. Since our target is >0.5% return in 5 days, the agent needs the stock to move >0.51% just to break even — that's right at the neutral zone boundary. Trading becomes nearly impossible to justify.

**Proposed fix:** Remove the explicit overtrading penalty. Transaction costs embedded in the portfolio already serve this purpose. If we find the agent trades too frequently, we can add a small penalty later (e.g., 0.0002), but start without it.

**Revised reward:**
```python
daily_return = (portfolio_value[t] - portfolio_value[t-1]) / portfolio_value[t-1]
if daily_return >= 0:
    reward = daily_return
else:
    reward = daily_return * 2.0  # Sortino asymmetry
```

---

### Issue 3: Confidence Threshold Is a No-Op

The plan says: "If `confidence < 0.6`: mask BUY and SELL — force HOLD."

Looking at the metrics, mean confidence across stocks is **0.75–0.93** (from `confidence_analysis.mean_confidence`). At threshold 0.6, the cumulative coverage is 100% for almost every window — meaning the threshold masks **nothing**. It's dead code.

**Two options:**
1. **Raise the threshold** to something meaningful (e.g., 0.8) — but this eliminates most trading opportunities
2. **Remove confidence masking entirely** and instead let the RL agent learn when to act from the confidence value in the observation vector

**Proposed fix:** Option 2. Keep confidence as an observation feature but remove it from action masking. The RL agent can learn the relationship between confidence and profitable trades on its own. This is more flexible and avoids hardcoding a threshold that depends on model behavior.

The only action masks should be the structural ones:
- Cannot BUY if already long (position == 1)
- Cannot SELL if flat (position == 0)

---

### Issue 4: Observation Normalization Is Unspecified

PPO is sensitive to input scale. The proposed observation vector has wildly different ranges:

| Feature | Typical Range | Problem |
|---|---|---|
| p_up | [0, 1] | Fine |
| confidence | [0.5, 1.0] | Fine |
| position | {0, 1} | Fine |
| unrealised_pnl | [-0.1, +0.1]? | Grows with portfolio size |
| days_in_position | [0, 500] | Huge range, unbounded |
| india_vix | [10, 80] | Order of magnitude larger |
| recent_returns × 5 | [-0.05, +0.05] | Tiny |

**Proposed fix:** Use `VecNormalize` from Stable-Baselines3 (running mean/std normalization) or manually normalize:
- `days_in_position`: clip to [0, 60], divide by 60
- `india_vix`: divide by 50 (approximate median VIX, or use training-window stats)
- `unrealised_pnl`: clip to [-0.2, +0.2], keep as-is (already a fraction)
- `recent_returns`: keep as-is (already small fractions)

Simpler: just wrap the env in `VecNormalize` and let SB3 handle it. But we need to save the running stats for inference consistency.

---

### Issue 5: Missing Data Pipeline for RL

The RL environment needs two things for each day:
1. **Stage 1 output:** `p_up` and `confidence` — from MC Dropout inference
2. **Raw market data:** `close` price (to compute portfolio returns), `india_vix`

The plan doesn't specify how to assemble this DataFrame. We need a function like:

```python
def generate_rl_training_data(stock_name, window, config) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    [date, close, india_vix, p_up, confidence]
    One row per trading day in the OOF period.
    """
```

This requires:
1. Loading the processed features CSV
2. Splitting by window
3. Training a Stage 1 model (or loading existing checkpoint)
4. Running MC Dropout inference on the OOF data (50 passes per day)
5. Merging predictions with raw close prices and VIX

**This is the most complex data engineering step** and must be carefully designed. I'll include it as part of `train_agent.py`.

---

### Issue 6: 5-Day Prediction Horizon vs Daily Stepping

Stage 1 predicts: "Will the stock be >0.5% higher in **5 trading days**?"

But the RL agent steps daily. On Monday, the model predicts Friday's direction. On Tuesday, it predicts next Monday's direction. These are **overlapping, partially correlated predictions** — Monday's prediction and Tuesday's prediction share 4 of 5 future days.

This isn't wrong — it just means the agent sees slowly-shifting signals rather than independent ones. The daily reward reflects actual daily portfolio change, so the feedback loop is honest.

**Design decision:** Daily stepping is correct and simpler. The overlapping predictions are fine. If the model says "UP for the next 5 days" every day, the agent should hold. If it flip-flops, the agent should be cautious. The agent can learn this from experience.

No code change needed, but this should be noted in code comments.

---

### Issue 7: PPO May Converge to Always-HOLD

When the agent is flat, `daily_return = 0`, `reward = 0`. If it trades, it risks negative reward from losses AND pays transaction costs. A risk-averse PPO agent may learn that always-HOLD is the safe choice (local optimum with reward = 0 forever).

**Mitigations already built-in:**
- Entropy coefficient (`ent_coef=0.01`) encourages exploration beyond always-HOLD
- If the model has any predictive signal, occasional BUY actions will produce positive rewards, reinforcing trading behavior

**Additional mitigation:** Log the percentage of HOLD actions per episode during training. If it's >95%, entropy coefficient needs to be increased or the reward needs to give a small positive signal for profitable trading. We'll add this as a diagnostic metric.

---

## Revised Implementation Plan

### [NEW] `rl/reward.py`

Sortino-shaped reward function — **no explicit overtrading penalty** (transaction costs handle that):

```python
def compute_step_reward(daily_return: float) -> float:
    """Asymmetric reward: penalize losses 2x more than gains."""
    if daily_return >= 0:
        return daily_return
    return daily_return * 2.0

def compute_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Episode-level Sortino ratio for logging."""
    excess = returns - risk_free
    downside = np.minimum(excess, 0)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-8:
        return 0.0
    return float(np.mean(excess) / downside_std)
```

---

### [NEW] `rl/trading_env.py`

Key design changes from the original plan:

1. **No confidence masking in `action_masks()`** — only structural masks (can't BUY when long, can't SELL when flat)
2. **Normalized observations** — VIX divided by 50, days_in_position clipped and divided by 60
3. **Uses pre-computed prediction DataFrame** — receives a DataFrame with `[date, close, india_vix, p_up, confidence]` at construction
4. **Tracks trade log** for diagnostic analysis

```python
class TradingEnv(gym.Env):
    def __init__(self, prediction_df, config):
        # Discrete(3): HOLD=0, BUY=1, SELL=2
        self.action_space = spaces.Discrete(3)
        # 10-dim observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        # Transaction costs from config
        self.brokerage = config["costs"]["brokerage_pct"]
        self.stt = config["costs"]["stt_pct"]
        self.slippage = config["costs"]["slippage_pct"]
```

---

### [NEW] `rl/train_agent.py`

The most complex file. Key design:

1. **Expanding-window OOF generation** — 3 folds within each walk-forward training window
2. **MC Dropout inference** to produce `p_up` and `confidence` per OOF day
3. **MaskablePPO training** with `n_steps` capped at episode length
4. **Diagnostic logging** — HOLD%, Sortino ratio, total trades per episode
5. **Saves RL checkpoint** alongside Stage 1 checkpoint in `results/STOCK/models/`

```
CLI: python -m rl.train_agent --stock RELIANCE --window 0
```

**Critical `n_steps` fix:** If OOF data has ~1,500 days across 3 episodes, `n_steps=1024` works. But if episodes are ~500 steps each, PPO's rollout buffer spans >1 episode, which is fine — SB3 handles auto-reset. The key is that `n_steps` must be ≤ `total episode length × num_envs`. With 1 env and 500-step episodes, `n_steps=512` is safer.

---

### [MODIFY] `config.yaml`

```yaml
model:
  weight_decay: 0.000794  # Add: missing from main config, present in best_config.yaml

rl:
  n_steps: 512            # Change from 1024: must fit within single OOF episode
  confidence_threshold: 0.6  # Keep in config but NOT used for action masking — only for predictor-only baseline comparison
```

---

### [MODIFY] `For_AI.md`

Update TODO list: Phase 2.75 → ✅ Done, Phase 3 → ← CURRENT

---

## Proposed File Structure

```
rl/
├── __init__.py             # Module docstring
├── reward.py               # compute_step_reward(), compute_sortino()
├── trading_env.py          # TradingEnv(gym.Env) — observation, step, reset, action_masks
└── train_agent.py          # OOF generation, MaskablePPO training, CLI entry point
```

---

## Verification Plan

### Unit Tests (manual, in-script)
1. **`trading_env.py`**: Instantiate with synthetic data (100 days, constant predictions). Verify:
   - `reset()` returns shape `(10,)`, dtype `float32`
   - After `step(BUY)`, `action_masks()` returns `[True, False, True]` (can HOLD or SELL, not BUY)
   - After `step(SELL)` from flat, action should be masked — env should raise or force HOLD
   - Buy → hold 5 days → sell: verify portfolio value reduced by exactly `brokerage + slippage` per side + `stt` on sell

2. **`reward.py`**: Verify `compute_step_reward(0.01) == 0.01`, `compute_step_reward(-0.01) == -0.02`

### Smoke Test
```
python -m rl.train_agent --stock RELIANCE --window 0
```
Expected: Runs for 200,000 timesteps, prints episode Sortino ratios, HOLD%, total trades. Saves checkpoint.

### Diagnostic Checks
- **HOLD% < 95%**: If the agent always HOLDs, increase `ent_coef` or investigate reward scale
- **Sortino > 0**: Agent should learn to generate positive risk-adjusted returns on OOF data (even if marginal)
- **No impossible trades**: BUY never appears when already long in the trade log

---

## Open Questions

> [!IMPORTANT]
> **Expanding-window OOF vs single-split OOF:** The 3-fold expanding-window approach triples compute time for OOF generation (~60 extra minutes for all stocks). An alternative is the simpler single 80/20 split with `n_steps=256` (smaller rollout buffer). The agent would see less market diversity but saves engineering complexity. Which do you prefer?
>
> **My recommendation:** Start with the simpler single 80/20 split. If the agent converges to always-HOLD or overfits, upgrade to expanding-window.
