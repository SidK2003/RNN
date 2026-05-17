# NIFTY 50 Deep RL Trading System — Implementation Plan

## My Honest Take on the Concept Document

Your `Product_Concept_RNN.md` is genuinely well-written. The two-stage framing (prediction → decision) is the right architecture — it's how real quant desks think. The "why two stages" section is the strongest paragraph in the doc; it shows you understand that prediction ≠ trading decision, which is a distinction most student projects completely miss.

**What's strong:**
- The separation of concerns (LSTM predicts, RL decides) is architecturally sound and mirrors how institutional trading systems work
- MC Dropout for uncertainty is a legitimate production technique, not a toy
- The three-way evaluation (buy-and-hold vs LSTM-only vs LSTM+RL) is exactly how you'd present this in an interview
- The "what this is NOT" section shows maturity — interviewers love honesty
- The resume narrative tying three projects together is smart positioning

**What needs sharpening before you build:**
- The current concept has a few gaps that a sharp interviewer at a quant fund would probe immediately: no mention of transaction costs, no walk-forward validation, no risk metrics beyond Sharpe. These are exactly the improvements you were suggested, and they're all valid.
- The concept says "predicts price direction" but the LSTM section says "predicted price direction" somewhat vaguely — you need to decide upfront: are you predicting raw price, returns, or binary direction? This matters architecturally.

---

## Evaluation of Every Suggested Improvement

### 1. Transaction Costs & Slippage in Backtests

**Verdict: MUST HAVE ✅**

This is non-negotiable. Without transaction costs, any backtested strategy looks better than it actually is. Every interviewer who has touched trading will ask "did you account for costs?"

**What it means:**
- Every time the RL agent executes a BUY or SELL, subtract a fixed percentage (e.g., 0.1% for brokerage + 0.01% for STT/SEBI charges on Indian markets, plus ~0.05% for slippage)
- Slippage = the difference between the price you expected and the price you actually got. In a backtest you simulate this by assuming you execute at a slightly worse price than the close

**Implementation:** Add a `transaction_cost` parameter to your Gym environment's `step()` function. When the agent executes a trade, deduct costs from the portfolio value before computing the reward.

**Complexity:** Very low. Maybe 10 lines of code. Huge credibility boost.

---

### 2. Walk-Forward Validation

**Verdict: MUST HAVE ✅**

A single 5-year train / 1-year test split is the weakest part of the current concept. The problem: your model might just have gotten lucky on that specific test year. Walk-forward validation is how real quant shops validate strategies.

**What it means:**
Instead of one split, you do multiple overlapping splits:
```
Window 1: Train on 2019-2022, Test on 2023
Window 2: Train on 2020-2023, Test on 2024
Window 3: Train on 2021-2024, Test on 2025
```
You retrain the model for each window and report aggregate metrics across all test windows. This proves your strategy works across different market regimes (bull, bear, sideways).

**Implementation:** A loop that shifts the train/test window forward by 1 year each iteration. You'll need a wrapper script that orchestrates training and evaluation for each window.

**Complexity:** Medium. Requires some engineering to automate, but not conceptually hard.

---

### 3. Predict Returns/Direction Instead of Exact Price

**Verdict: YES — Predict Direction (Binary Classification) ✅**

This is the right call, and here's the detailed reasoning:

| Approach | Pros | Cons |
|---|---|---|
| **Predict exact price** | Intuitive to explain | Stock prices are non-stationary; the model learns "TCS is around ₹4000" which is useless. Error metrics (RMSE) are misleading — you can have low RMSE and still lose money |
| **Predict returns** | Stationary series (easier to model), directly actionable | Still a regression problem; magnitude is hard to predict accurately |
| **Predict direction** (UP/DOWN) | Clean binary classification; aligns perfectly with trading decisions; easier to evaluate (accuracy, precision/recall); works better with confidence scores | You lose magnitude information |

**My recommendation:** **Predict direction.** Here's why it fits YOUR architecture perfectly:

Your RL agent doesn't need to know "the stock will go up by 1.7%." It needs to know "the stock will go UP, and I'm 82% confident." The magnitude of the move is irrelevant to the agent's BUY/HOLD/SELL decision — what matters is direction + confidence. The RL agent learns the position sizing and risk management itself through training.

Also, from a resume perspective, you can frame binary classification with MC Dropout confidence as "probabilistic directional forecasting" — which sounds serious and is technically accurate.

**Implementation:** Change the LSTM's final layer from a regression output to a sigmoid (binary classification: UP=1, DOWN=0). Loss function changes from MSE to Binary Cross-Entropy. The MC Dropout confidence score now naturally represents P(UP) across stochastic forward passes.

---

### 4. Confidence Threshold Before RL Agent Acts

**Verdict: YES ✅**

This is a great idea and directly complements the MC Dropout uncertainty. The concept: if the LSTM's confidence is below some threshold (e.g., 60%), the RL agent is forced to HOLD regardless of what it would have otherwise done.

**Why it's good:**
- Prevents the agent from acting on noisy predictions
- Reduces overtrading (which eats into returns via transaction costs)
- Easy to explain in interviews: "The system knows when it doesn't know"

**Implementation:** In the Gym environment, if `confidence < threshold`, mask the BUY and SELL actions (force HOLD). The threshold itself can be a hyperparameter you tune.

**Complexity:** Very low. 5 lines of code in the environment's action processing.

---

### 5. Track Risk Metrics, Not Just Returns

**Verdict: MUST HAVE ✅**

Returns alone don't tell you if a strategy is good. A strategy that returns 30% but has a 50% drawdown is terrible. You need:

| Metric | What It Tells You |
|---|---|
| **Cumulative Return** | Total P&L |
| **Sharpe Ratio** | Risk-adjusted return (return per unit of total volatility) |
| **Sortino Ratio** | Risk-adjusted return penalising only downside volatility |
| **Maximum Drawdown** | Worst peak-to-trough loss — the moment you'd panic |
| **Win Rate** | % of trades that were profitable |
| **Calmar Ratio** | Annual return / max drawdown — measures how painful the ride is |
| **Profit Factor** | Gross profits / gross losses — above 1.5 is decent |

**Implementation:** Use the `quantstats` Python library. It generates all of these from a returns series in one line. You can even generate a full HTML tearsheet.

**Complexity:** Very low. `quantstats` does most of the work.

---

### 6. Sortino Ratio as Primary Metric & Reward Shaping

**Verdict: YES ✅ — as reward function, not just reporting metric**

The Sharpe Ratio penalises ALL volatility equally — upside and downside. But in trading, upside volatility is good (you WANT big winners). The Sortino Ratio only penalises downside volatility, which is a more accurate measure of risk.

**Using it as the RL reward:**
Instead of rewarding raw portfolio return, shape the reward to penalise downside moves more than it rewards upside moves. A simple approach:

```
reward = portfolio_return if portfolio_return >= 0
reward = portfolio_return * 2.0 if portfolio_return < 0  (double penalty for losses)
```

A more sophisticated approach: compute a rolling Sortino Ratio over the last N steps and use that as the reward signal.

**Why this matters for your project:** It teaches the RL agent to be risk-averse, which is what you want. An agent trained on raw returns will take huge risks. An agent trained on Sortino-shaped rewards learns to protect capital — which is much more impressive to demonstrate.

**Complexity:** Low-Medium. The reward shaping is maybe 15 lines; the full rolling Sortino calculation is a bit more involved but very doable.

---

### 7. Macro Context Feature (India VIX)

**Verdict: YES — but keep it simple ✅**

The India VIX measures implied volatility of the NIFTY 50 index. High VIX = market fear/uncertainty. It's a macro signal that tells the model "the overall market is nervous right now."

**Why it helps:**
- Individual stock predictions in high-VIX environments are unreliable — adding VIX as a feature lets the model learn this
- It's a single additional feature column, not a major architectural change
- Easy to source: available on NSE website and via yfinance (`^INDIAVIX`)

**Implementation:** Download India VIX daily data, merge it with your stock OHLCV data by date, and add it as an additional input feature to the LSTM.

**Complexity:** Very low. One extra column in your feature dataframe.

---

### 8. Action Space Masking

**Verdict: YES ✅**

This means: don't let the RL agent take impossible actions. For example:
- Can't SELL if you don't hold any stock
- Can't BUY if you don't have enough capital
- (From suggestion #4) Can't BUY/SELL if confidence is below threshold

**Why it matters:**
- Without masking, the agent wastes training time exploring impossible actions and getting penalised for them
- With masking, training is more efficient and the agent converges faster
- It's a best practice in RL — interviewers who know RL will expect it

**Implementation:** Stable-Baselines3 supports action masking via `sb3-contrib`'s `MaskablePPO`. You override `action_masks()` in your Gym environment to return a boolean array of valid actions.

**Complexity:** Medium. Requires switching from vanilla PPO to MaskablePPO and implementing the mask logic. Well-documented in sb3-contrib though.

---

## GPU Utilization — RTX 4070 Super (12GB VRAM)

Your 4070 Super is a serious card. Here's how to use it effectively:

### Suggestion 1: Replace LSTM with a Better Architecture

**Verdict: Use GRU + Multi-Head Attention + MC Dropout ✅**

Let me compare your options:

| Architecture | VRAM Need | Training Time | Resume Impact | Implementation Complexity |
|---|---|---|---|---|
| **LSTM** (current) | ~1-2 GB | Fast | Common, not impressive | Easy |
| **GRU + Multi-Head Attention** | ~2-4 GB | Moderate | Very strong — shows you understand attention mechanisms | Medium |
| **Temporal Fusion Transformer (TFT)** | ~6-10 GB | Slow | Impressive but risky — complex to get right, may overfit on small datasets | Hard |

**My recommendation: GRU + Multi-Head Attention + MC Dropout.**

Here's why:
- **GRU vs LSTM:** GRU has fewer parameters (no separate cell state), trains faster, and performs equally well on most time series tasks. It's a more modern choice.
- **Multi-Head Attention:** This is the key upgrade. Instead of the LSTM/GRU treating all timesteps equally, attention learns to focus on the most relevant past days. You already mentioned attention visualisation in your concept doc — with multi-head attention, this becomes a genuinely powerful explainability tool.
- **TFT is overkill:** The Temporal Fusion Transformer is designed for datasets with hundreds of thousands of rows and dozens of covariates. On 6 years of daily data (~1,500 rows per stock), a TFT will almost certainly overfit. It's also a nightmare to tune. The GRU+Attention approach gives you 80% of the benefit at 20% of the complexity.

**What Multi-Head Attention Actually Is (quick refresher):**
Instead of one attention head deciding "day -3 was important," you have multiple heads (say 4) that each independently learn different patterns. One head might learn to focus on volume spikes, another on price reversals, another on recent trends. The outputs are concatenated and transformed. This is the same mechanism used in Transformers/GPT, but applied to your sequence model's hidden states.

---

### Suggestion 2: Hyperparameter Search

**Verdict: YES — use Optuna ✅**

With 12GB VRAM, you can run multiple training experiments in parallel or sequence. Use Optuna (a Bayesian hyperparameter optimization library) to search over:

- Learning rate (1e-4 to 1e-2)
- GRU hidden size (64, 128, 256)
- Number of attention heads (2, 4, 8)
- Dropout rate (0.1 to 0.4)
- Sequence length (30, 60, 90 days)
- PPO hyperparameters (clip range, entropy coefficient, etc.)

**Why Optuna over grid search:** Optuna uses a Tree-structured Parzen Estimator (TPE) which is much smarter than trying every combination. It learns from past trials to focus on promising regions of the hyperparameter space. You might get a good set in 50-100 trials instead of thousands.

**Complexity:** Medium. Optuna integrates cleanly with PyTorch. You wrap your training loop in an `objective()` function and let Optuna drive it.

---

### Suggestion 3: Train Multi-Stock Models

**Verdict: YES — but carefully ✅**

Instead of training a separate model per stock, train ONE model on all 5 stocks. This has a name in ML: **multi-task learning**.

**Why it helps:**
- More training data (5× as much)
- The model learns market-wide patterns, not just stock-specific ones
- Cross-sector learning (e.g., banking stocks crash → pharma stocks are safe)

**How to implement:** Add a stock identifier as a categorical feature (one-hot encoded or learned embedding). The model learns both shared market patterns and stock-specific behaviours.

> [!WARNING]
> **Risk:** If the stocks are too different, a multi-stock model may learn nothing useful. Start with single-stock, then try multi-stock as an experiment. Report both results — that's the honest thing to do.

---

### Suggestion 4: Use Longer Sequences

**Verdict: YES ✅**

With a GPU, you can easily handle 90-120 day lookback windows instead of the typical 30-60. Longer sequences let the model see:
- Quarterly earnings cycles
- Longer-term trends
- Seasonal patterns

The GRU+Attention architecture handles long sequences much better than a plain LSTM because the attention mechanism can directly "look at" day -90 without the signal degrading through 90 recurrent steps.

**Make this a hyperparameter** and let Optuna find the optimal window length.

---

### Suggestion 5: Probabilistic Forecasting

**Verdict: Already covered by MC Dropout ✅ — don't add more complexity**

You're already doing probabilistic forecasting via MC Dropout (run inference N times with dropout enabled, get a distribution of outputs). This gives you:
- Mean prediction (direction)
- Variance (confidence/uncertainty)
- Confidence intervals

Adding a separate probabilistic layer (like a Mixture Density Network) on top of this would be over-engineering. MC Dropout is cleaner, easier to explain, and already a production technique. **Stick with it.**

---

## Final Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                          │
│  yfinance + India VIX → OHLCV + Technical Indicators    │
│  → Feature Engineering → Walk-Forward Splits             │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           STAGE 1: PREDICTION ENGINE                     │
│  GRU + Multi-Head Attention + MC Dropout                 │
│  Input: 60-90 day window of features                     │
│  Output: P(UP) direction + confidence interval           │
│  Loss: Binary Cross-Entropy                              │
│  Trained per walk-forward window                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 2: RL TRADING AGENT                       │
│  MaskablePPO (sb3-contrib)                               │
│  Observation: prediction, confidence, portfolio state,   │
│               VIX, technical indicators                  │
│  Actions: BUY / HOLD / SELL (with masking)               │
│  Reward: Sortino-shaped (penalise downside)              │
│  Costs: Transaction fees + slippage deducted per trade   │
│  Confidence gate: forced HOLD if confidence < threshold  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              EVALUATION & DASHBOARD                      │
│  Walk-forward aggregated metrics:                        │
│  Cumulative Return, Sortino, Sharpe, Max Drawdown,       │
│  Win Rate, Calmar, Profit Factor                         │
│  Three-way comparison: Buy-Hold vs LSTM-only vs RL       │
│  Attention heatmaps, equity curves, trade log            │
│  Streamlit dashboard with 3 tabs                         │
└─────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
nifty50-deep-rl-trader/
│
├── README.md                     # Project overview, setup, results summary
├── requirements.txt              # All dependencies pinned
├── config.yaml                   # Central config (hyperparams, stock list, date ranges)
│
├── data/
│   ├── raw/                      # Downloaded OHLCV + VIX CSVs
│   ├── processed/                # Feature-engineered, split datasets
│   └── download.py               # yfinance download script
│
├── features/
│   ├── technical_indicators.py   # RSI, MACD, Bollinger, ATR, etc.
│   ├── vix.py                    # India VIX feature integration
│   └── pipeline.py               # Full feature engineering pipeline
│
├── models/
│   ├── gru_attention.py          # GRU + Multi-Head Attention + MC Dropout model
│   ├── train_predictor.py        # Training loop for Stage 1
│   └── inference.py              # MC Dropout inference (N forward passes)
│
├── rl/
│   ├── trading_env.py            # Custom Gymnasium environment
│   ├── reward.py                 # Sortino-shaped reward function
│   ├── action_masking.py         # Action mask logic
│   └── train_agent.py            # PPO/MaskablePPO training
│
├── evaluation/
│   ├── walk_forward.py           # Walk-forward validation orchestrator
│   ├── metrics.py                # Sortino, Sharpe, drawdown, etc.
│   ├── backtest.py               # Run 3-way comparison
│   └── report.py                 # Generate quantstats HTML tearsheet
│
├── dashboard/
│   └── app.py                    # Streamlit dashboard (3 tabs)
│
├── tuning/
│   └── optuna_search.py          # Hyperparameter optimization
│
├── notebooks/                    # Exploration / debugging only
│   └── eda.ipynb
│
└── results/
    ├── models/                   # Saved model weights
    ├── plots/                    # Equity curves, attention maps
    └── tearsheets/               # quantstats HTML reports
```

---

## Theory / Prerequisites You Should Brush Up On

### GRU (Gated Recurrent Unit)
A simplified LSTM with only two gates (reset and update) instead of three. Fewer parameters, faster training, same sequential learning capability. The key equation to understand:

- **Update gate (z):** How much of the previous hidden state to keep
- **Reset gate (r):** How much of the previous state to forget when computing the candidate
- Think of it as: LSTM without the separate cell state. The hidden state IS the memory.

### Multi-Head Attention
After the GRU processes the sequence, you have a hidden state for each timestep. Attention computes a weighted sum of these states:
1. Each hidden state is projected into Query (Q), Key (K), and Value (V) vectors
2. Attention weights = softmax(Q × K^T / √d)
3. Output = weighted sum of V using those weights
4. "Multi-head" = do this N times with different projections, concatenate results

This lets the model learn N different "types of relevance" simultaneously.

### MC Dropout (Monte Carlo Dropout)
Normally, dropout is disabled at inference time. MC Dropout keeps it enabled. You run the same input through the model T times (e.g., 50), each time different neurons are dropped, giving slightly different outputs. The mean of these outputs is your prediction; the standard deviation is your uncertainty. This is a Bayesian approximation — it's sampling from the posterior distribution of the model's weights.

### PPO (Proximal Policy Optimization)
The RL algorithm your agent uses. Key concepts:
- **Policy:** The agent's strategy (a neural network that maps observations → action probabilities)
- **Value function:** Estimates how good the current state is
- **Clipping:** PPO prevents the policy from changing too much in one update (the "proximal" part). This makes training stable.
- You don't need to implement PPO from scratch — Stable-Baselines3 handles it. You need to understand what the hyperparameters mean.

### Sortino Ratio
```
Sortino = (Portfolio Return - Risk-Free Rate) / Downside Deviation
```
Where Downside Deviation = standard deviation of ONLY negative returns. Unlike Sharpe, it doesn't penalise upside volatility. A Sortino > 2 is considered good for a trading strategy.

### Walk-Forward Validation
The time-series equivalent of k-fold cross-validation. You can't randomly shuffle time series data (that would be look-ahead bias). Instead, you roll the training window forward through time, always testing on unseen future data.

---

## What Will Look Best on Your Resume

The resume bullet points should emphasize:

1. **Two-stage architecture** (prediction → decision) — shows systems thinking
2. **Walk-forward validated** — shows you understand proper evaluation
3. **Sortino-optimised RL agent** — shows you understand risk, not just returns
4. **Uncertainty quantification via MC Dropout** — production ML technique
5. **Action masking + confidence gating** — shows you handle edge cases
6. **Attention visualisation** — explainability, critical for fintech
7. **Transaction cost modelling** — shows realism

### Suggested Resume Entry (Draft)

```
NIFTY 50 Deep RL Trading System
Sole Developer · github.com/SidK2003/nifty50-deep-rl

• Built a two-stage trading system: a GRU + Multi-Head Attention model predicts
  daily stock direction with MC Dropout uncertainty, and a Sortino-optimised PPO
  agent decides whether to act on each prediction.

• Engineered a custom OpenAI Gymnasium environment with transaction cost
  modelling, action space masking, and confidence-gated execution — the agent
  learns to abstain when the predictor is uncertain.

• Validated using walk-forward analysis across multiple market regimes;
  benchmarked against buy-and-hold on 5 NIFTY 50 stocks with full risk
  reporting (Sortino, max drawdown, Calmar ratio).

• Built a Streamlit dashboard with live signal inference, backtest equity
  curves, and attention weight heatmaps for prediction explainability.

Stack: Python · PyTorch · Stable-Baselines3 · OpenAI Gymnasium · Optuna · Streamlit
```

---

## Updated Skills Section (Draft)

```
Languages     Python
ML / DL       PyTorch (GPU), Scikit-learn, Optuna
LLM / AI      Google Gemini API, RAG, Hugging Face Transformers, LangChain
RL            Stable-Baselines3, OpenAI Gymnasium, PPO
Vector DBs    ChromaDB, sentence-transformers
Backend/Web   FastAPI, Node.js, React
Data          Pandas, Supabase (PostgreSQL), yfinance, quantstats
Tools         Git / GitHub, Vercel, CUDA/cuDNN
Finance       Quantitative Finance, Risk Metrics, Algorithmic Trading
```

---

## Phased Implementation Roadmap

### Phase 1: Data & Features (Days 1-3)
- [ ] Set up project structure and `config.yaml`
- [ ] Download OHLCV data for 5 NIFTY 50 stocks via yfinance
- [ ] Download India VIX data
- [ ] Implement technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV)
- [ ] Build feature engineering pipeline with proper normalization
- [ ] Implement walk-forward data splitting logic

### Phase 2: Prediction Model (Days 4-8)
- [ ] Implement GRU + Multi-Head Attention architecture in PyTorch
- [ ] Add MC Dropout layer
- [ ] Train on first walk-forward window (direction classification)
- [ ] Implement MC Dropout inference (50 forward passes → mean direction + confidence)
- [ ] Validate attention weights visualisation works
- [ ] Run full walk-forward training loop

### Phase 3: RL Environment & Agent (Days 9-14)
- [ ] Build custom Gymnasium trading environment
- [ ] Implement transaction costs and slippage
- [ ] Implement Sortino-shaped reward function
- [ ] Implement action space masking (sb3-contrib MaskablePPO)
- [ ] Add confidence threshold gating
- [ ] Train PPO agent within walk-forward framework
- [ ] Debug and validate agent behaviour (does it learn to not overtrade?)

### Phase 4: Evaluation & Tuning (Days 15-18)
- [ ] Implement three-way comparison (buy-hold vs predictor-only vs RL)
- [ ] Generate full risk metrics (Sortino, Sharpe, drawdown, Calmar, win rate, profit factor)
- [ ] Run Optuna hyperparameter search (GPU-accelerated)
- [ ] Generate quantstats tearsheets
- [ ] Analyse results honestly — document where system underperforms

### Phase 5: Dashboard & Polish (Days 19-21)
- [ ] Build Streamlit dashboard with 3 tabs
- [ ] Tab 1: Today's signal (live inference)
- [ ] Tab 2: Backtest comparison (equity curves, metrics table)
- [ ] Tab 3: Attention heatmaps
- [ ] Write README with results, architecture diagram, setup instructions
- [ ] Clean up code, add docstrings

---

## Open Questions for You

> [!IMPORTANT]
> **Stock Selection:** Which 5 NIFTY 50 stocks do you want to use? My suggestion: **TCS** (IT), **HDFCBANK** (Banking), **HINDUNILVR** (FMCG), **SUNPHARMA** (Pharma), **RELIANCE** (Energy/Conglomerate). These give good sector diversification.

> [!IMPORTANT]
> **Timeframe:** 6 years of daily data is fine for an MVP, but do you want to use intraday data (e.g., 15-minute candles)? Intraday would give you ~50× more data points but adds complexity (market hours, gaps). My recommendation: **stick with daily** for the MVP.

> [!IMPORTANT]
> **Multi-Stock vs Single-Stock:** Do you want to train one model per stock, or one model for all 5? I'd suggest starting single-stock and experimenting with multi-stock as a follow-up. This way you have results either way.

> [!IMPORTANT]
> **Dashboard Priority:** The Streamlit dashboard is nice for demos, but it's the least technically impressive part. If you're short on time, prioritise the model and evaluation over the dashboard. A well-documented Jupyter notebook with clear results is better than a pretty dashboard with questionable models.

---

## Verification Plan

### Automated Tests
- Unit tests for feature engineering (correct indicator calculations)
- Unit tests for Gym environment (state transitions, reward computation, action masking)
- Walk-forward split validation (no data leakage between train/test)
- Model training smoke test (loss decreases over epochs)

### Manual Verification
- Visual inspection of attention heatmaps (do they focus on meaningful days?)
- Equity curve sanity check (does the RL agent beat buy-and-hold after costs?)
- Quantstats tearsheet review
- Trade log analysis (is the agent overtrading? Making impossible trades?)
