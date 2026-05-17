# NIFTY 50 Deep RL Trading System — Final Implementation Plan

---

## Locked Decisions

| Decision | Choice |
|---|---|
| Language | Python |
| Data source | Upstox API — script at `Data/upstox.py` |
| Data granularity | **Daily only** (MVP) |
| Data volume | ~25 years per stock (~6,500 rows), 2000–2026 |
| Stocks | 5 single-stock models trained independently |
| Prediction target | **Binary direction** (UP=1, DOWN=0) — next-day close vs today's close |
| Stage 1 model | GRU + Multi-Head Attention + MC Dropout (PyTorch) |
| Stage 2 agent | MaskablePPO (sb3-contrib) |
| Validation | Walk-forward (8 rolling windows across 25 years) |
| Primary metric | Sortino Ratio |
| Dashboard | **Deferred** — keep interfaces/comments ready for later connection |

---

## Data

### Source & Format

Data is pre-downloaded via `Data/upstox.py` as CSVs. Schema:

| Column | Type | Notes |
|---|---|---|
| timestamp | datetime+tz | `2000-01-03 00:00:00+05:30` — strip timezone, use date only |
| open | float | |
| high | float | |
| low | float | |
| close | float | |
| volume | int | |
| oi | int | Always 0 for equity — **drop this column** |

### Stocks (5 NIFTY 50, cross-sector)

| Stock | Sector | CSV file |
|---|---|---|
| RELIANCE | Energy | `Data/reliance_daily.csv` ✅ exists |
| TCS | IT | `Data/tcs_daily.csv` — user will download |
| HDFCBANK | Banking | `Data/hdfcbank_daily.csv` — user will download |
| HINDUNILVR | FMCG | `Data/hindunilvr_daily.csv` — user will download |
| SUNPHARMA | Pharma | `Data/sunpharma_daily.csv` — user will download |

### India VIX

- Separate CSV: `Data/india_vix_daily.csv`
- VIX data only available from **~2009 onwards**
- For dates before 2009: fill VIX column with 0 or NaN and add a binary `vix_available` feature (0/1) so the model knows when VIX data is missing vs when VIX is actually low
- Merge with stock data on date (inner join on trading days)

---

## Feature Engineering

All features computed from raw OHLCV. Implemented in `features/` module.

### Target Variable
- `target = 1 if close[t+1] > close[t] else 0` — binary direction of next day's close
- This is the label for Stage 1 training

### Input Features

**Price-based:**
| Feature | Computation | Window |
|---|---|---|
| log_return | `log(close[t] / close[t-1])` | — |
| bollinger_pctb | `(close - SMA) / (2 * std)` | 20-day |
| atr | Average True Range | 14-day |

**Momentum:**
| Feature | Computation | Window |
|---|---|---|
| rsi | Relative Strength Index | 14-day |
| macd | EMA(12) - EMA(26) | — |
| macd_signal | EMA(9) of MACD | — |
| macd_histogram | MACD - signal | — |
| stoch_k | Stochastic %K | 14-day |
| stoch_d | SMA of %K | 3-day |

**Volume:**
| Feature | Computation | Window |
|---|---|---|
| obv | On-Balance Volume (cumulative) | — |
| volume_sma_ratio | `volume / SMA(volume, 20)` | 20-day |

**Macro:**
| Feature | Computation | Notes |
|---|---|---|
| india_vix | Raw India VIX value | From 2009 onwards |
| vix_available | `1 if VIX data exists for this date, else 0` | Binary flag |

### Normalization

> [!CAUTION]
> **CRITICAL — No Look-Ahead Bias in Normalization**
> 
> Every feature must be normalized using **only data available up to that point in time**. Never use future data for normalization.
> 
> - Use **rolling z-score**: `(x - rolling_mean) / rolling_std` with a window equal to the training window length
> - At walk-forward boundaries: compute mean/std from **training data only**, apply those statistics to test data
> - Do NOT use `sklearn.StandardScaler.fit()` on the full dataset — this leaks future statistics into the past

### NaN Handling
- First ~20-30 rows will have NaN from rolling indicators — **drop them**
- Verify no NaN remains after feature engineering before training

---

## Walk-Forward Validation

Rolling window scheme across 25 years. **10-year train, 2-year test, 2-year step.**

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

> [!CAUTION]
> **CRITICAL — Walk-Forward Bias Prevention**
> 
> - **No data from the test period may be used in training** — not for feature computation, normalization, hyperparameter tuning, or any other purpose
> - **Retrain the model from scratch for each window** — do not warm-start from the previous window's weights
> - **Features must be computed per-window**: rolling indicators at the train/test boundary must only use training data
> - The final metrics are **aggregated across all 8 test windows** — do not cherry-pick the best window

---

## Stage 1: Prediction Engine

### Architecture: GRU + Multi-Head Attention + MC Dropout

```
Input: (batch_size, seq_len, num_features)
   │
   ▼
GRU (bidirectional=False, num_layers=2, hidden_size=128)
   │ output: (batch_size, seq_len, hidden_size)
   ▼
Multi-Head Attention (num_heads=4, embed_dim=hidden_size)
   │ Query = Key = Value = GRU output (self-attention)
   │ output: (batch_size, seq_len, hidden_size)
   ▼
Attention-weighted pooling → (batch_size, hidden_size)
   │ (weighted sum of attention output using attention weights)
   ▼
Dropout (p=0.3) ← MC Dropout — stays ON during inference
   │
   ▼
Linear(hidden_size, 64) → ReLU → Dropout(0.3)
   │
   ▼
Linear(64, 1) → Sigmoid
   │
   ▼
Output: P(UP) ∈ [0, 1]
```

### Training
- **Loss:** Binary Cross-Entropy (`BCELoss`)
- **Optimizer:** AdamW with weight decay
- **Mixed precision:** `torch.amp.autocast('cuda')` + `GradScaler` for FP16 training
- **Gradient accumulation:** Accumulate over N=4 mini-batches if batch size is limited by VRAM
- **Sequence length:** Hyperparameter — search range [30, 60, 90, 120] days via Optuna
- **Early stopping:** Monitor validation loss, patience=10 epochs
- **Save:** Best model weights per walk-forward window to `results/models/`

### Attention Weight Extraction
- After the Multi-Head Attention layer, extract and save attention weights for each prediction
- Shape: `(num_heads, seq_len, seq_len)` — save as numpy arrays
- These will be used for attention heatmap visualisations
- Add a method `model.get_attention_weights(x)` that returns both prediction and attention weights

### MC Dropout Inference
- At inference, call `model.train()` to keep dropout active (NOT `model.eval()`)
- Run **T=50 forward passes** on the same input
- `predictions = [model(x) for _ in range(T)]`  → shape: (50,)
- `direction = mean(predictions) > 0.5`  → UP or DOWN
- `confidence = 1 - std(predictions)`  → high std = low confidence
- Also compute: `p_up = mean(predictions)` → raw probability of UP

### Model Ensembling (Optional Enhancement)
- Train 3–5 models with different random seeds on the same data
- At inference, each model runs MC Dropout independently
- Final prediction = average of all models' `p_up`
- Final confidence = combination of inter-model agreement + MC Dropout variance
- This is low-effort and high-impact — implement after single model works

---

## Stage 2: RL Trading Agent

### Custom Gymnasium Environment (`rl/trading_env.py`)

**Observation space** (what the agent sees at each step):
| Component | Type | Description |
|---|---|---|
| prediction | float [0, 1] | P(UP) from Stage 1 |
| confidence | float [0, 1] | MC Dropout confidence |
| position | int {-1, 0, 1} | -1=short (not used in MVP), 0=flat, 1=long |
| unrealised_pnl | float | Current position's unrealised P&L (normalized) |
| days_in_position | int | How long current position has been held |
| india_vix | float | Current VIX value (normalized) |
| recent_returns | float[5] | Last 5 days of portfolio returns |

**Action space:** Discrete(3) — `{0: HOLD, 1: BUY, 2: SELL}`

**Action Masking (MaskablePPO):**
- Cannot BUY if already holding (position == 1)
- Cannot SELL if not holding (position == 0)
- If `confidence < confidence_threshold`: mask BUY and SELL — force HOLD
- Implement `action_masks()` method returning `np.array([bool, bool, bool])`

**Transaction Costs:**
| Cost | Value | When Applied |
|---|---|---|
| Brokerage | 0.1% of trade value | Every BUY and SELL |
| STT/SEBI | 0.01% | Every SELL |
| Slippage | 0.05% adverse | Every BUY and SELL |
| **Total per round-trip** | **~0.31%** | |

Implementation: When executing BUY or SELL, deduct costs from portfolio value before computing reward.

**Reward Function (Sortino-shaped):**
```python
daily_return = (portfolio_value[t] - portfolio_value[t-1]) / portfolio_value[t-1]

if daily_return >= 0:
    reward = daily_return
else:
    reward = daily_return * 2.0  # double penalty for losses

# Small penalty for each trade to discourage overtrading
if action in [BUY, SELL]:
    reward -= 0.001
```

**Episode:** One episode = one walk-forward test window (e.g., 2 years of trading days ≈ 500 steps)

### Training
- **Algorithm:** `MaskablePPO` from `sb3-contrib`
- **Policy:** MLP (not shared with Stage 1 — separate network)
- **Total timesteps:** Search via Optuna, starting at 100,000–500,000
- **Hyperparameters to tune:** learning_rate, clip_range, ent_coef, n_steps, batch_size

> [!CAUTION]
> **CRITICAL — RL Training Bias Prevention**
> 
> - The RL agent must be trained on **training data only** within each walk-forward window
> - The Stage 1 predictions fed to the RL agent during training must come from **out-of-fold predictions** — i.e., use a validation split within the training window, not the model's predictions on its own training data (which would be overfit and unrealistically accurate)
> - During test evaluation, use the Stage 1 model's genuine predictions on unseen test data

---

## Evaluation & Visualisation

### Three-Way Comparison

For each walk-forward test window, run three strategies on the same test data:

1. **Buy-and-Hold:** Buy on day 1, hold until end. Baseline.
2. **Predictor-Only:** BUY when P(UP) > 0.5, SELL otherwise. No RL agent.
3. **Full System (LSTM+RL):** Stage 1 prediction → Stage 2 RL agent decision.

### Risk Metrics (per strategy, per window, and aggregated)

| Metric | Library |
|---|---|
| Cumulative Return | quantstats |
| Sharpe Ratio | quantstats |
| Sortino Ratio | quantstats (primary metric) |
| Maximum Drawdown | quantstats |
| Win Rate | custom (% profitable trades) |
| Calmar Ratio | quantstats |
| Profit Factor | custom (gross profits / gross losses) |
| Total Trades | custom |
| Avg Trade Duration | custom |

### Visualisations (all saved to `results/plots/`)

All metrics must be visualised as graphs/charts:

1. **Equity curves** — 3 strategies overlaid per stock per window (matplotlib/plotly)
2. **Drawdown chart** — underwater plot showing drawdown over time
3. **Metric comparison bar chart** — Sortino/Sharpe/Calmar side-by-side for 3 strategies
4. **Attention heatmaps** — which past days the model focuses on for sample predictions
5. **Trade scatter** — BUY/SELL markers overlaid on price chart
6. **Walk-forward summary** — aggregate metrics table across all 8 windows
7. **Confidence distribution** — histogram of MC Dropout confidence scores
8. **quantstats HTML tearsheet** — full report saved to `results/tearsheets/`

> [!NOTE]
> Add `# TODO: Streamlit integration` comments next to all visualisation functions. These will later be connected to the dashboard's 3 tabs.

---

## Hyperparameter Search (Optuna)

Run on GPU with FP16. Search space:

**Stage 1 (GRU+Attention):**
| Param | Range |
|---|---|
| learning_rate | [1e-4, 1e-2] log-uniform |
| hidden_size | {64, 128, 256} |
| num_layers | {1, 2, 3} |
| num_heads | {2, 4, 8} |
| dropout | [0.1, 0.4] |
| seq_length | {30, 60, 90, 120} |
| batch_size | {32, 64, 128} |

**Stage 2 (MaskablePPO):**
| Param | Range |
|---|---|
| learning_rate | [1e-5, 1e-3] log-uniform |
| clip_range | [0.1, 0.3] |
| ent_coef | [0.0, 0.05] |
| confidence_threshold | [0.5, 0.8] |
| n_steps | {512, 1024, 2048} |

**Objective:** Maximize Sortino Ratio on a held-out validation split within the first walk-forward training window.

---

## Bias & Pitfall Checklist

> [!CAUTION]
> **Every item below must be explicitly verified during implementation.**

| Pitfall | Prevention |
|---|---|
| **Look-ahead bias in features** | Rolling indicators use only past data. No future leakage. |
| **Look-ahead bias in normalization** | Z-score stats computed on training window only, applied to test. |
| **Look-ahead bias in walk-forward splits** | Test period never overlaps with or precedes training period. |
| **Survivorship bias** | All 5 stocks are current NIFTY 50 constituents — acknowledge this limitation in results. We're not claiming to pick stocks, just to time them. |
| **Overfitting to training data** | Early stopping on validation loss, dropout, walk-forward validation. |
| **RL trained on overfit predictions** | Use out-of-fold predictions from Stage 1 to train Stage 2. |
| **Impossible actions** | Action masking prevents BUY when already long, SELL when flat. |
| **Ignoring transaction costs** | 0.31% round-trip cost deducted on every trade. |
| **Ignoring slippage** | 0.05% adverse slippage on every trade execution. |
| **Data snooping** | Hyperparameter tuning done on validation split inside training window, never on test. |
| **NaN propagation** | Assert no NaN in features before training. Drop initial rows with insufficient history. |

---

## GPU Configuration (RTX 4070 Super, 12GB VRAM)

| Optimization | Implementation |
|---|---|
| **Mixed precision (FP16)** | `torch.amp.autocast('cuda')` + `GradScaler` — ~2× faster, ~50% less VRAM |
| **Gradient accumulation** | Accumulate over 4 mini-batches before `optimizer.step()` — larger effective batch |
| **CUDA device** | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| **Pin memory** | `DataLoader(pin_memory=True)` for faster CPU→GPU transfer |
| **Deterministic seeds** | Set `torch.manual_seed()`, `np.random.seed()`, `torch.backends.cudnn.deterministic=True` for reproducibility |

---

## Project Directory Structure

```
RNN/                              # Repo root (existing)
│
├── README.md
├── requirements.txt
├── config.yaml                   # All hyperparams, stock list, date ranges, cost params
├── .gitignore                    # Already exists
├── Product_Concept_RNN.md        # Already exists
│
├── Data/                         # Already exists
│   ├── .env                      # Upstox access token (gitignored)
│   ├── upstox.py                 # Already exists — download script
│   ├── reliance_daily.csv        # Already exists (~6,557 rows)
│   ├── tcs_daily.csv             # User will download
│   ├── hdfcbank_daily.csv        # User will download
│   ├── hindunilvr_daily.csv      # User will download
│   ├── sunpharma_daily.csv       # User will download
│   ├── india_vix_daily.csv       # User will download
│   └── processed/                # Generated by feature pipeline
│       ├── reliance_features.csv
│       ├── tcs_features.csv
│       └── ...
│
├── features/
│   ├── __init__.py
│   ├── technical_indicators.py   # RSI, MACD, Bollinger, ATR, OBV, Stochastic
│   ├── vix.py                    # India VIX merge + vix_available flag
│   └── pipeline.py               # Full pipeline: load CSV → compute features → normalize → save
│
├── models/
│   ├── __init__.py
│   ├── gru_attention.py          # GRU + Multi-Head Attention + MC Dropout (PyTorch nn.Module)
│   ├── train_predictor.py        # Training loop (FP16, early stopping, walk-forward)
│   └── inference.py              # MC Dropout inference (T=50 passes) + ensemble logic
│
├── rl/
│   ├── __init__.py
│   ├── trading_env.py            # Custom Gymnasium env (observation, action, reward, masking)
│   ├── reward.py                 # Sortino-shaped reward computation
│   └── train_agent.py            # MaskablePPO training + walk-forward integration
│
├── evaluation/
│   ├── __init__.py
│   ├── walk_forward.py           # Orchestrates full walk-forward loop (train → predict → RL → evaluate)
│   ├── metrics.py                # Sortino, Sharpe, drawdown, Calmar, win rate, profit factor
│   ├── backtest.py               # Runs 3-way comparison (buy-hold vs predictor-only vs full system)
│   ├── visualise.py              # All charts: equity curves, drawdowns, attention maps, trade scatter
│   └── report.py                 # quantstats HTML tearsheet generation
│
├── tuning/
│   ├── __init__.py
│   └── optuna_search.py          # Hyperparameter optimization for both Stage 1 and Stage 2
│
├── dashboard/                    # POST-MVP placeholder
│   └── app.py                    # TODO: Streamlit (3 tabs: signals, backtest, attention)
│
├── notebooks/
│   └── eda.ipynb                 # Exploration only
│
└── results/
    ├── models/                   # Saved .pt weights per stock per window
    ├── plots/                    # All generated PNGs/HTMLs
    └── tearsheets/               # quantstats HTML reports
```

---

## `config.yaml` Schema

```yaml
stocks:
  - name: RELIANCE
    file: Data/reliance_daily.csv
  - name: TCS
    file: Data/tcs_daily.csv
  - name: HDFCBANK
    file: Data/hdfcbank_daily.csv
  - name: HINDUNILVR
    file: Data/hindunilvr_daily.csv
  - name: SUNPHARMA
    file: Data/sunpharma_daily.csv

vix:
  file: Data/india_vix_daily.csv
  available_from: "2009-01-01"

walk_forward:
  train_years: 10
  test_years: 2
  step_years: 2

model:
  seq_length: 60          # Optuna-tunable
  hidden_size: 128        # Optuna-tunable
  num_layers: 2           # Optuna-tunable
  num_heads: 4            # Optuna-tunable
  dropout: 0.3            # Optuna-tunable
  learning_rate: 0.001    # Optuna-tunable
  batch_size: 64          # Optuna-tunable
  max_epochs: 100
  early_stopping_patience: 10
  mc_dropout_passes: 50
  ensemble_count: 3       # Number of models in ensemble (1 = no ensemble)

rl:
  algorithm: MaskablePPO
  total_timesteps: 200000
  learning_rate: 0.0003   # Optuna-tunable
  clip_range: 0.2         # Optuna-tunable
  ent_coef: 0.01          # Optuna-tunable
  n_steps: 1024           # Optuna-tunable
  confidence_threshold: 0.6  # Optuna-tunable

costs:
  brokerage_pct: 0.001    # 0.1%
  stt_pct: 0.0001         # 0.01% on sell
  slippage_pct: 0.0005    # 0.05%

gpu:
  mixed_precision: true
  gradient_accumulation_steps: 4
  pin_memory: true

seed: 42

results_dir: results/
```

---

## Phased Task Checklist

### Phase 1: Data & Features (Days 1–3)
- [x] Create `config.yaml`
- [x] Create `requirements.txt` (torch, stable-baselines3, sb3-contrib, gymnasium, optuna, quantstats, pandas, numpy, matplotlib, plotly, pyyaml)
- [x] Implement `features/technical_indicators.py` — all indicators listed above
- [x] Implement `features/vix.py` — VIX merge + `vix_available` flag
- [x] Implement `features/pipeline.py` — end-to-end: load CSV → features → rolling z-score normalization → target column → save to `Data/processed/`
- [ ] Implement walk-forward splitting logic (can live in `evaluation/walk_forward.py`)
- [ ] **Verify:** no NaN in output, no look-ahead bias in normalization, correct target alignment

### Phase 2: Prediction Model (Days 4–8)
- [ ] Implement `models/gru_attention.py` — `GRUAttentionModel(nn.Module)` with `get_attention_weights()` method
- [ ] Implement `models/train_predictor.py` — FP16 training loop, early stopping, gradient accumulation, model saving
- [ ] Implement `models/inference.py` — MC Dropout inference (T=50 passes), ensemble averaging
- [ ] Train on first walk-forward window, verify loss decreases, verify attention weights are extractable
- [ ] Run full walk-forward training loop for one stock (RELIANCE)
- [ ] **Verify:** model outputs are in [0, 1], confidence scores are meaningful, no NaN in outputs

### Phase 3: RL Environment & Agent (Days 9–14)
- [ ] Implement `rl/trading_env.py` — full Gymnasium env with observation/action/reward/masking
- [ ] Implement `rl/reward.py` — Sortino-shaped reward
- [ ] Implement `rl/train_agent.py` — MaskablePPO training with walk-forward integration
- [ ] **Critical:** ensure RL trains on out-of-fold Stage 1 predictions (not overfit training predictions)
- [ ] Train agent on first walk-forward window, verify it learns to not overtrade
- [ ] **Verify:** action masking works (no impossible trades in trade log), transaction costs are deducted

### Phase 4: Evaluation & Tuning (Days 15–18)
- [ ] Implement `evaluation/metrics.py` — all risk metrics
- [ ] Implement `evaluation/backtest.py` — 3-way comparison runner
- [ ] Implement `evaluation/visualise.py` — all 8 chart types listed above, saved to `results/plots/`
- [ ] Implement `evaluation/report.py` — quantstats tearsheet
- [ ] Run full pipeline for all 5 stocks across all 8 walk-forward windows
- [ ] Run Optuna search (GPU, FP16)
- [ ] Document results honestly — include where system underperforms

### Phase 5: Polish (Days 19–20)
- [ ] Write README with results summary, architecture diagram, setup instructions
- [ ] Clean up code, add docstrings
- [ ] Add `# TODO: Streamlit` comments at all visualisation touchpoints
- [ ] Verify all plots saved, all tearsheets generated
