# Phase 4: Evaluation, Backtesting & Visualization

## Current State

**Phases 1–3 are complete.** We have:
- ✅ Feature pipeline (`features/`) — 20+ features across 5 stocks
- ✅ Stage 1 predictor (`models/`) — GRU+Attention, trained on all 5 stocks across all walk-forward windows, Optuna-tuned
- ✅ Stage 2 RL agent (`rl/`) — MaskablePPO with OOF training pipeline, custom TradingEnv, Sortino reward

**What's missing:** There is no way to answer the central question: *"Does this system actually make money?"*

We need the `evaluation/` module to:
1. Orchestrate the full end-to-end pipeline across all windows
2. Run three competing strategies on the same test data
3. Compute standardized risk metrics
4. Generate visual evidence (equity curves, drawdowns, trade overlays)
5. Produce a final report

---

## What Phase 4 Must Build

### 4 New Files in `evaluation/`

| File | Purpose |
|---|---|
| `walk_forward.py` | Orchestrates: load data → Stage 1 inference → RL agent decision → log trades for each test window |
| `metrics.py` | Computes Sortino, Sharpe, Max Drawdown, Win Rate, Profit Factor, Calmar, trade statistics |
| `backtest.py` | Runs 3-way comparison (Buy-and-Hold vs Predictor-Only vs Full RL) on every test window |
| `visualise.py` | Generates equity curves, drawdown charts, trade scatter plots, metric bar charts |

> [!IMPORTANT]
> `report.py` from the original plan is **merged into `backtest.py`** — generating a quantstats HTML tearsheet is a single function call, not worth a separate module. This keeps the evaluation module focused at 4 files instead of 5.

---

## Detailed Design

### 1. `evaluation/walk_forward.py` — The Orchestrator

This is the most critical file. For **each stock × each walk-forward window**, it:

1. Loads the processed features CSV
2. Splits data by window boundaries (train/test)
3. Normalizes test data using the **saved training-window scaler** (from `results/STOCK/norm_stats/windowN_scaler.pkl`)
4. Loads the trained Stage 1 model checkpoint (`results/STOCK/models/windowN.pt`)
5. Runs MC Dropout inference (T=50) on the test data to produce `p_up` and `confidence`
6. Assembles a test DataFrame: `[date, close, india_vix, p_up, confidence]`
7. Passes this to `backtest.py` for the 3-way strategy comparison

**Key data flow:**
```
Processed CSV → split_by_window() → normalize (saved scaler) → MC Dropout → test_df
                                                                                |
                                                             backtest.run_all_strategies(test_df)
```

**Critical anti-bias check:** The scaler used for test normalization is the one fitted during training (saved as `.pkl`). We do NOT refit on test data.

**CLI:**
```powershell
python -m evaluation.walk_forward --stock RELIANCE          # All windows for one stock
python -m evaluation.walk_forward --stock RELIANCE --window 0  # One window
python -m evaluation.walk_forward                           # All stocks, all windows
```

**Output:** Saves `results/STOCK/backtest_results.json` — a structured JSON with per-window and aggregate metrics for all 3 strategies.

---

### 2. `evaluation/metrics.py` — Risk Metric Calculator

Pure functions, no side effects. Each takes an array of daily returns and outputs a scalar.

| Function | Formula / Logic |
|---|---|
| `sortino_ratio(returns, rf=0)` | `mean(excess) / downside_std` where downside = `min(excess, 0)` |
| `sharpe_ratio(returns, rf=0)` | `mean(excess) / std(returns)` |
| `max_drawdown(cumulative_returns)` | `max(peak - trough) / peak` |
| `calmar_ratio(returns, rf=0)` | `annualized_return / max_drawdown` |
| `win_rate(trade_returns)` | `count(positive trades) / total trades` |
| `profit_factor(trade_returns)` | `sum(winning trades) / abs(sum(losing trades))` |
| `compute_all_metrics(returns, trades)` | Master function returning a dict of all metrics |

**Annualization:** Indian markets have ~250 trading days/year. All annualized metrics use `sqrt(250)` or `250` as appropriate.

> [!NOTE]
> We do NOT use the `quantstats` library for metric computation. Its API is fragile and frequently changes. We compute metrics ourselves for reliability and reproducibility. We only use `quantstats` for its HTML tearsheet generation (which is a nice-to-have visualization).

---

### 3. `evaluation/backtest.py` — Three-Way Strategy Engine

For a given test DataFrame `[date, close, india_vix, p_up, confidence]`, this module runs three strategies and returns their equity curves + trade logs.

#### Strategy 1: Buy-and-Hold (Baseline)
```
Day 1: Buy at close price. Hold forever.
Equity[t] = close[t] / close[0]
```
No transaction costs (single buy, never sell during window).

#### Strategy 2: Predictor-Only
```
If p_up > 0.5 AND confidence > threshold:
    If flat → BUY
If p_up <= 0.5:
    If long → SELL
```
Transaction costs applied on every BUY/SELL. Uses `confidence_threshold` from `config.yaml`.
This strategy tests: "What if we naively followed Stage 1's predictions?"

#### Strategy 3: Full RL System
```
Load saved RL model (results/STOCK/models/rl_windowN.zip)
Feed test_df into TradingEnv
Let agent.predict(obs, action_masks=env.action_masks()) decide
```
Transaction costs are embedded in the TradingEnv.

**Each strategy outputs:**
```python
{
    "equity_curve": pd.Series,        # daily portfolio values indexed by date
    "daily_returns": np.ndarray,      # daily % returns
    "trade_log": List[dict],          # [{date, action, price, portfolio_value}, ...]
    "metrics": dict                   # from metrics.compute_all_metrics()
}
```

**quantstats tearsheet:** After running all three strategies, we generate a `quantstats.reports.html()` tearsheet comparing the RL strategy against Buy-and-Hold. Saved to `results/STOCK/tearsheets/windowN.html`.

---

### 4. `evaluation/visualise.py` — Chart Generator

All plots saved as PNGs to `results/STOCK/plots/`. Uses `matplotlib` (not plotly — simpler, faster, no browser dependency).

| Chart | Description |
|---|---|
| **Equity Curves** | 3 overlaid equity curves per window. X=date, Y=portfolio value. |
| **Drawdown Chart** | Underwater plot: `(equity - peak) / peak` for RL strategy. |
| **Metric Comparison Bars** | Side-by-side bar chart of Sortino, Sharpe, Max Drawdown for all 3 strategies. |
| **Trade Scatter** | Close price line chart with BUY (green ▲) and SELL (red ▼) markers from the RL agent. |
| **Aggregate Summary Table** | A single PNG table showing average metrics across all windows, per stock. |

Each function takes strategy results as input and saves a PNG. No state, no side effects.

---

## Design Decisions & Deviations from `final_implementation_plan.md`

### Changes from the Original Plan

| Original Plan | Phase 4 Decision | Reason |
|---|---|---|
| `report.py` as separate file | Merged into `backtest.py` | quantstats tearsheet is a single function call |
| 8 chart types including attention heatmaps | 5 chart types (dropped attention heatmaps, confidence distribution, walk-forward summary) | Attention heatmaps are Stage 1 diagnostics already in `models/evaluate.py`. Confidence distribution is already there. The walk-forward summary IS the aggregate table. |
| `quantstats` for metric computation | Custom `metrics.py` | `quantstats` API is unreliable; we control our own metric math |
| Plotly for some charts | Matplotlib only | Simpler, no browser dependency, consistent output |

### RL Model Loading for Backtesting

The backtest needs a trained RL model for each window. Currently we only have `rl_window0.zip` for RELIANCE. The walk-forward orchestrator must:
1. Check if `rl_windowN.zip` exists
2. If not, **train it on the fly** using `rl.train_agent.train_rl_agent()`
3. If yes, load and use it

This means the first full run for a stock will be slow (training RL for all windows), but subsequent runs are instant.

---

## Verification Plan

### Smoke Test
```powershell
python -m evaluation.walk_forward --stock RELIANCE --window 0
```
Expected output:
- Prints metrics for all 3 strategies
- Saves `results/RELIANCE/backtest_results.json`
- Saves equity curve PNG to `results/RELIANCE/plots/window0_equity.png`
- Saves trade scatter PNG to `results/RELIANCE/plots/window0_trades.png`

### Full Run
```powershell
python -m evaluation.walk_forward --stock RELIANCE
```
Expected: Runs all 8 windows, saves aggregate metrics, generates all charts.

### Sanity Checks
- Buy-and-Hold equity curve should match the actual stock price movement
- Predictor-Only should have more trades than RL (RL is more selective)
- No impossible trades in any strategy's trade log
- Transaction costs should be visible as small drops in equity on trade days
- All metric values should be finite (no NaN/Inf)

---

## Open Questions

> [!IMPORTANT]
> **RL training for all windows:** We currently only have `rl_window0.zip` for RELIANCE. Should we:
> - **(A)** Train RL for all 8 windows × 5 stocks before running evaluation (40 training jobs, ~3.5 minutes each = ~2.5 hours)
> - **(B)** Let `walk_forward.py` train them on-the-fly during the first evaluation run
> 
> **My recommendation:** Option B. It's simpler and the training is cached (once trained, never retrained). The first evaluation run will be slow, but all subsequent runs are instant.
