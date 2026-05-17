# NIFTY 50 Deep RL Trading System — Implementation Plan (v2)

> [!NOTE]
> **v2 Updates:** Incorporates all user feedback — 25 years of Upstox data, direction-only prediction, GRU+Attention locked for MVP, single-stock first, dashboard deferred, intraday data strategy added, additional GPU tips.

---

## Decisions Locked In

| Decision | Choice | Rationale |
|---|---|---|
| Data source | **Upstox API** (not yfinance) | Script already built at `Data/upstox.py` |
| Data volume | **~25 years daily** (~6,500 rows/stock) | 2000–2026 for most stocks |
| Prediction target | **Binary direction** (UP/DOWN) | Cleaner for RL, better confidence scores |
| Architecture (MVP) | **GRU + Multi-Head Attention + MC Dropout** | Even with 25 years of data — TFT deferred to post-MVP |
| Stock training | **Single-stock models** for MVP | Multi-stock deferred to post-MVP |
| Dashboard | **Deferred** — but keep interfaces/comments ready | Focus on backend/models first |
| Intraday data | Available from Jan 2022 — **use as post-MVP experiment** | See analysis below |

### Post-MVP Backlog
- TFT architecture (when all 50 NIFTY stocks are available)
- Multi-stock training with learned embeddings
- Streamlit dashboard (3 tabs)
- Intraday model variant

---

## Intraday Data Strategy

You have access to 1-min, 15-min, hourly, weekly data from Jan 2022 onwards via Upstox.

**For MVP: Stick with daily only.** Here's why:
- Mixing timeframes (25 years daily + 4 years intraday) creates a Frankenstein dataset — the model sees different granularity in different eras, which is a data leakage risk
- Intraday trading is a fundamentally different problem (different patterns, different costs, different holding periods)
- Daily data with 25 years gives you ~6,500 rows per stock — that's plenty for GRU+Attention

**Post-MVP opportunity:** Train a separate intraday model on 15-min candles from 2022–2026. That's ~4 years × ~375 trading days × ~25 candles/day ≈ **37,500 rows per stock**. That's enough to try TFT. You could then compare daily vs intraday performance — which is a great analysis to show.

---

## All 8 Improvements — Status

All accepted. Quick reference:

| # | Improvement | Status | Complexity |
|---|---|---|---|
| 1 | Transaction costs & slippage | ✅ Must have | Very low |
| 2 | Walk-forward validation | ✅ Must have | Medium |
| 3 | Predict direction only | ✅ Locked | Low |
| 4 | Confidence threshold gating | ✅ Accepted | Very low |
| 5 | Full risk metrics | ✅ Must have | Very low (quantstats) |
| 6 | Sortino reward shaping | ✅ Accepted | Low-Medium |
| 7 | India VIX feature | ✅ Accepted | Very low |
| 8 | Action space masking | ✅ Accepted | Medium |

---

## Walk-Forward Validation — Updated for 25 Years

With 25 years of data, walk-forward becomes much more powerful. Use a **rolling 10-year train / 2-year test** window:

```
Window 1:  Train 2000–2009, Test 2010–2011
Window 2:  Train 2002–2011, Test 2012–2013
Window 3:  Train 2004–2013, Test 2014–2015
Window 4:  Train 2006–2015, Test 2016–2017
Window 5:  Train 2008–2017, Test 2018–2019
Window 6:  Train 2010–2019, Test 2020–2021
Window 7:  Train 2012–2021, Test 2022–2023
Window 8:  Train 2014–2023, Test 2024–2025
```

This gives you **8 test windows** spanning 16 years of out-of-sample data across multiple market regimes (2008 crash, 2020 COVID crash, 2021 bull run, 2022 correction). That's extremely robust validation.

The exact window sizes (train length, test length, step size) become hyperparameters — Optuna can search over these too.

---

## GPU Utilization — RTX 4070 Super (12GB VRAM)

### Already Decided
- **GRU + Multi-Head Attention + MC Dropout** (not TFT for MVP)
- **Optuna hyperparameter search**
- **Longer sequences** (60–120 day windows, Optuna-tuned)

### Additional GPU Tips Worth Adding

**1. Mixed Precision Training (FP16) ✅**

This is free performance. PyTorch's `torch.amp` (Automatic Mixed Precision) runs most operations in 16-bit floats instead of 32-bit:
- **~1.5–2× faster training** on your 4070 Super (Tensor Cores)
- **~50% less VRAM** per batch → you can double your batch size
- Zero accuracy loss for this type of model

Implementation: 3 lines of code wrapping your training loop with `torch.amp.GradScaler` and `torch.amp.autocast`.

**2. Gradient Accumulation ✅**

If you want even larger effective batch sizes without running out of VRAM:
- Instead of updating weights every batch, accumulate gradients over N mini-batches, then update once
- Effective batch size = mini-batch × N
- Larger batches → more stable gradients → smoother training

Useful during Optuna search when testing large hidden sizes (256+) that eat more VRAM.

**3. Model Ensembling (Low-Effort, High-Impact) ✅**

Train 3–5 GRU+Attention models with different random seeds. At inference:
- Each model gives a direction prediction via MC Dropout
- Average the predictions across models
- Confidence = agreement between models × MC Dropout confidence

This is a simple way to boost accuracy without changing the architecture. The GPU makes training 5 models feasible (each takes maybe 2–5 minutes on daily data).

**Why this is resume-worthy:** You can say "ensemble of uncertainty-aware models" — which is how production ML systems actually work. Single models are for demos; ensembles are for production.

---

## Final Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                          │
│  Upstox API + India VIX → OHLCV + Technical Indicators  │
│  → Feature Engineering → Walk-Forward Splits (8 windows) │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           STAGE 1: PREDICTION ENGINE                     │
│  GRU + Multi-Head Attention + MC Dropout                 │
│  Input: 60-120 day window of features (Optuna-tuned)     │
│  Output: P(UP) direction + confidence interval           │
│  Loss: Binary Cross-Entropy                              │
│  Training: Mixed precision (FP16) + walk-forward         │
│  Optional: Ensemble of 3-5 models                        │
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
│              EVALUATION & VISUALISATION                   │
│  Walk-forward aggregated metrics (8 windows):            │
│  Cumulative Return, Sortino, Sharpe, Max Drawdown,       │
│  Win Rate, Calmar, Profit Factor                         │
│  Three-way comparison: Buy-Hold vs Predictor-only vs RL  │
│  All metrics visualised as graphs (matplotlib/plotly)     │
│  Attention heatmaps, equity curves, trade log            │
│  # TODO: Connect to Streamlit dashboard (post-MVP)       │
└─────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
RNN/                              # Your existing repo root
│
├── README.md
├── requirements.txt
├── config.yaml                   # Central config (hyperparams, stocks, dates)
├── .gitignore                    # Already exists
├── Product_Concept_RNN.md        # Already exists
│
├── Data/                         # Already exists
│   ├── upstox.py                 # Already exists — Upstox download script
│   ├── reliance_daily.csv        # Already exists
│   ├── raw/                      # Other stock CSVs + VIX CSV
│   └── processed/                # Feature-engineered datasets (generated)
│
├── features/
│   ├── technical_indicators.py   # RSI, MACD, Bollinger, ATR, OBV, etc.
│   ├── vix.py                    # India VIX merge logic
│   └── pipeline.py               # Full feature eng + normalization pipeline
│
├── models/
│   ├── gru_attention.py          # GRU + Multi-Head Attention + MC Dropout
│   ├── train_predictor.py        # Training loop (FP16, walk-forward)
│   └── inference.py              # MC Dropout inference (N forward passes)
│
├── rl/
│   ├── trading_env.py            # Custom Gymnasium environment
│   ├── reward.py                 # Sortino-shaped reward function
│   ├── action_masking.py         # Action mask logic + confidence gate
│   └── train_agent.py            # MaskablePPO training
│
├── evaluation/
│   ├── walk_forward.py           # Walk-forward orchestrator
│   ├── metrics.py                # Risk metrics computation
│   ├── backtest.py               # 3-way comparison runner
│   ├── visualise.py              # Equity curves, metric charts, attention maps
│   └── report.py                 # quantstats HTML tearsheet
│
├── tuning/
│   └── optuna_search.py          # Hyperparameter optimization
│
├── dashboard/                    # POST-MVP — kept as placeholder
│   └── app.py                    # TODO: Streamlit dashboard (3 tabs)
│
├── notebooks/
│   └── eda.ipynb                 # Exploration only
│
└── results/
    ├── models/                   # Saved weights (.pt files)
    ├── plots/                    # Generated charts (PNG/HTML)
    └── tearsheets/               # quantstats reports
```

---

## Data Format Reference

Your Upstox data has this schema (from `reliance_daily.csv`):

| Column | Type | Example |
|---|---|---|
| timestamp | datetime+tz | `2000-01-03 00:00:00+05:30` |
| open | float | 21.8 |
| high | float | 23.1 |
| low | float | 21.8 |
| close | float | 23.1 |
| volume | int | 48573772 |
| oi | int | 0 (open interest — always 0 for equity) |

**Stocks to download** (5 NIFTY 50 across sectors):

| Stock | Sector | Upstox Instrument Key |
|---|---|---|
| RELIANCE | Energy | `NSE_EQ\|INE002A01018` ✅ Already have |
| TCS | IT | `NSE_EQ\|INE467B01029` |
| HDFCBANK | Banking | `NSE_EQ\|INE040A01034` |
| HINDUNILVR | FMCG | `NSE_EQ\|INE030A01027` |
| SUNPHARMA | Pharma | `NSE_EQ\|INE044A01036` |
| India VIX | Macro | Need to source separately (NSE website or yfinance `^INDIAVIX`) |

> [!NOTE]
> India VIX data only goes back to ~2009 (that's when NSE started publishing it). For walk-forward windows before 2009, we'll fill VIX with a neutral value or exclude it. This is fine — the model learns to use it when available.

---

## Feature Engineering (I'll Handle This)

The feature pipeline will compute these from raw OHLCV:

### Price-Based
- **Returns:** Daily log returns (what we predict direction of)
- **Bollinger Bands:** 20-day SMA ± 2σ → %B position (where price sits in the band)
- **ATR (Average True Range):** 14-day volatility measure

### Momentum
- **RSI (Relative Strength Index):** 14-day, measures overbought/oversold
- **MACD:** 12/26/9 EMA crossover signal
- **Stochastic Oscillator:** %K and %D lines

### Volume
- **OBV (On-Balance Volume):** Cumulative volume flow
- **Volume SMA ratio:** Today's volume / 20-day average volume

### Macro
- **India VIX:** Daily implied volatility (from 2009 onwards)

### Normalization
- All features will be **z-score normalized** using a rolling window (not the full dataset — that would be look-ahead bias)
- Rolling window = training window only, applied to test data using training statistics

---

## Theory / Prerequisites

### GRU (Gated Recurrent Unit)
Simplified LSTM — two gates (reset, update) instead of three. Fewer parameters, faster training, same capability. The hidden state IS the memory (no separate cell state like LSTM).

### Multi-Head Attention
After GRU processes the sequence, attention computes weighted sums of hidden states:
1. Project each hidden state → Q, K, V vectors
2. Attention weights = softmax(Q × K^T / √d)
3. Output = weighted V
4. Multi-head = N independent attention computations concatenated

Each head learns different "types of relevance" (volume patterns, price reversals, trends).

### MC Dropout
Keep dropout enabled at inference. Run N forward passes (e.g., 50), get N slightly different outputs. Mean = prediction, std = uncertainty. Bayesian approximation — cheap and effective.

### PPO (Proximal Policy Optimization)
RL algorithm: policy network maps observations → action probabilities. Clipping prevents unstable updates. Stable-Baselines3 handles implementation — you configure hyperparameters.

### Sortino Ratio
`(Return - Risk-Free Rate) / Downside Deviation` — penalises only downside volatility. Sortino > 2 is good.

### Walk-Forward Validation
Time-series k-fold: roll training window forward, always test on unseen future. Proves robustness across market regimes.

---

## Resume Entry (Draft)

```
NIFTY 50 Deep RL Trading System
Sole Developer · github.com/SidK2003/nifty50-deep-rl

• Built a two-stage trading system: a GRU + Multi-Head Attention model predicts
  daily stock direction with MC Dropout uncertainty, and a Sortino-optimised PPO
  agent decides whether to act on each prediction.

• Engineered a custom OpenAI Gymnasium environment with transaction cost
  modelling, action space masking, and confidence-gated execution — the agent
  learns to abstain when the predictor is uncertain.

• Walk-forward validated across 8 market regime windows (2010–2025) on 5 NIFTY
  50 stocks; benchmarked against buy-and-hold with full risk reporting (Sortino,
  max drawdown, Calmar ratio).

Stack: Python · PyTorch (GPU) · Stable-Baselines3 · OpenAI Gymnasium · Optuna
```

### Updated Skills Section

```
Languages     Python
ML / DL       PyTorch (GPU), Scikit-learn, Optuna
LLM / AI      Google Gemini API, RAG, Hugging Face Transformers, LangChain
RL            Stable-Baselines3, OpenAI Gymnasium, PPO
Vector DBs    ChromaDB, sentence-transformers
Backend/Web   FastAPI, Node.js, React
Data          Pandas, Supabase (PostgreSQL), quantstats
Tools         Git / GitHub, Vercel, CUDA/cuDNN
Finance       Quantitative Finance, Risk Metrics, Algorithmic Trading
```

---

## Phased Implementation Roadmap

### Phase 1: Data & Features (Days 1-3)
- [ ] Set up project structure and `config.yaml`
- [ ] Download remaining 4 stocks + India VIX via Upstox / NSE
- [ ] Implement technical indicators (RSI, MACD, Bollinger, ATR, OBV, Stochastic)
- [ ] Build feature engineering pipeline with rolling z-score normalization
- [ ] Implement walk-forward data splitting (8 windows across 25 years)
- [ ] Add dashboard connection comments/interfaces throughout

### Phase 2: Prediction Model (Days 4-8)
- [ ] Implement GRU + Multi-Head Attention + MC Dropout in PyTorch
- [ ] Set up mixed precision training (FP16)
- [ ] Train on first walk-forward window (binary direction classification)
- [ ] Implement MC Dropout inference (50 forward passes → P(UP) + confidence)
- [ ] Validate attention weights extraction works
- [ ] Run full walk-forward training loop
- [ ] (Optional) Train ensemble of 3-5 models with different seeds

### Phase 3: RL Environment & Agent (Days 9-14)
- [ ] Build custom Gymnasium trading environment
- [ ] Implement transaction costs and slippage (Indian market rates)
- [ ] Implement Sortino-shaped reward function
- [ ] Implement action space masking (sb3-contrib MaskablePPO)
- [ ] Add confidence threshold gating
- [ ] Train PPO agent within walk-forward framework
- [ ] Debug and validate agent behaviour

### Phase 4: Evaluation & Tuning (Days 15-18)
- [ ] Implement three-way comparison (buy-hold vs predictor-only vs RL)
- [ ] Generate full risk metrics with quantstats
- [ ] **Visualise ALL metrics as graphs** (equity curves, drawdown charts, metric comparison bars, attention heatmaps)
- [ ] Run Optuna hyperparameter search (GPU-accelerated, FP16)
- [ ] Analyse and document results honestly

### Phase 5: Polish & README (Days 19-20)
- [ ] Write README with results, architecture diagram, setup instructions
- [ ] Clean up code, add docstrings
- [ ] Ensure all visualisation outputs saved to `results/plots/`
- [ ] Add `# TODO: Streamlit` comments where dashboard will connect

### Phase 6: Dashboard (POST-MVP)
- [ ] Streamlit Tab 1: Today's signal (live inference)
- [ ] Streamlit Tab 2: Backtest comparison (equity curves, metrics table)
- [ ] Streamlit Tab 3: Attention heatmaps

---

## Verification Plan

### Automated
- Unit tests: feature engineering, Gym environment (state transitions, rewards, masking)
- Walk-forward split validation (no data leakage)
- Model training smoke test (loss decreases)

### Visual / Manual
- **Attention heatmaps** — do they focus on meaningful days? (saved as PNG)
- **Equity curves** — does RL beat buy-and-hold after costs? (saved as PNG)
- **Metric comparison charts** — bar charts of Sortino/Sharpe/drawdown across strategies
- **Trade log scatter** — entry/exit points overlaid on price chart
- **Quantstats tearsheet** — full HTML report
- All visualisations saved to `results/plots/` for dashboard integration later
