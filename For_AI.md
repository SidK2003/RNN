# For_AI: Onboarding Guide & Project Reference

**Notice to AI Assistant:** Read this file *first* to understand the NIFTY 50 Deep RL Trading System. It is the single source of truth for architecture, rules, and existing patterns. Do not blindly read all source files unless explicitly required. 
*Also, heavily refer to `extra/final_implementation_plan.md`, as it contains the definitive checklist and detailed hyperparameter parameters for each phase.*

---

## 1. Project Purpose & High-Level Architecture
This is a two-stage hybrid Deep Learning and Reinforcement Learning pipeline designed for daily equity trading of NIFTY 50 stocks. 
- **Stage 1 (Deep Learning):** A `GRU + Multi-Head Attention + MC Dropout` PyTorch model predicts the probability of a stock going UP over the next 5 trading days (using a ±0.5% neutral zone to filter noise) and provides a confidence score based on Monte Carlo Dropout uncertainty.
- **Stage 2 (Reinforcement Learning):** A `MaskablePPO` (sb3-contrib) agent takes the Stage 1 predictions and makes discrete trading decisions (`BUY`, `SELL`, `HOLD`), explicitly optimizing for the **Sortino Ratio** while navigating transaction costs and slippage.

Validation is performed via a strict **Walk-Forward** rolling window mechanism (10 years train, 2 years test) across ~25 years of data.

---

## 2. Folder/File Structure
```text
RNN/
├── config.yaml               # Central hyperparameter, path, and config registry
├── requirements.txt          # Python dependencies
├── For_AI.md                 # (This File) AI onboarding guide
├── CLAUDE.md                 # Claude Code guidance (commands, architecture, invariants)
├── SUGGESTIONS.md            # Senior-engineer review: prioritized improvements roadmap
├── extra/
│   ├── final_implementation_plan.md # The COMPLETE, detailed phase-by-phase implementation plan and specifications.
│   ├── phase_4_implementation_plan.md # Phase 4 (evaluation) design doc + deviations from original plan
│   ├── System_Theory_and_Design.md  # Theory explanation behind the chosen architecture
│   └── external_events.txt          # Future plan for macro/event features (NOT yet implemented)
├── Data/
│   ├── historical_data/      # Raw Upstox CSVs (DO NOT push to git) — all 5 stocks + VIX + NIFTY50 present
│   ├── processed/            # Feature-engineered CSVs — REGENERATED after each pipeline change
│   ├── upstox.py             # Script to download daily data from Upstox API
│   └── validate_data.py      # Script to ensure no NaNs/Infs in processed data
├── features/                 # Stage 0: Feature Engineering ✅ COMPLETE
│   ├── pipeline.py           # Main ETL orchestrator
│   ├── technical_indicators.py # RSI, MACD, Bollinger, ATR, stock_volatility_20d, etc.
│   ├── vix.py                # India VIX merge logic & gap handling
│   └── nifty.py              # NIFTY50 index merge — log returns, 5d returns, 20d volatility
├── models/                   # Stage 1: PyTorch GRU+Attention ✅ COMPLETE
│   ├── gru_attention.py      # GRU+MHA+MC Dropout nn.Module (return_logits mode for AMP)
│   ├── dataset.py            # StockSequenceDataset — sliding window with neutral-label filtering
│   ├── train_predictor.py    # Training loop: walk-forward, FP16, RobustScaler, dynamic features
│   ├── evaluate.py           # Post-training evaluation: accuracy, AUC, MCC, balanced accuracy, calibration
│   └── inference.py          # MC Dropout inference (T=50 passes) + model loading
├── rl/                       # Stage 2: Gymnasium Env & MaskablePPO ✅ COMPLETE
│   ├── reward.py             # Sortino-shaped asymmetric reward (losses penalized 2x)
│   ├── trading_env.py        # Custom Gymnasium env (11-dim obs, action masking, transaction costs)
│   └── train_agent.py        # OOF prediction generator + MaskablePPO training loop
├── evaluation/               # Phase 4: Backtesting & Visualization ✅ COMPLETE
│   ├── walk_forward.py       # End-to-end orchestrator: Stage 1 inference → RL decisions → metrics
│   ├── metrics.py            # Sortino, Sharpe, Max Drawdown, Calmar, Win Rate, Profit Factor
│   ├── backtest.py           # 3-way comparison: Buy-and-Hold vs Predictor-Only vs Full RL + quantstats tearsheet
│   └── visualise.py          # Equity curves, drawdown charts, trade scatter plots
├── tuning/                   # Optuna hyperparameter searches ✅ COMPLETE
└── dashboard/                # Post-MVP Streamlit UI (TODO)
```

---

## 3. Main Entry Points & Execution Flow
*(Stage 0, Stage 1, Stage 2, and Phase 4 Evaluation code are all complete. Post-MVP Streamlit dashboard is next.)*
1. **Activate Env:** `.\env\Scripts\Activate.ps1`
2. **Fetch Data:** `python Data/upstox.py`
3. **Process Features:** `python -m features.pipeline`
4. **Validate Data:** `python Data/validate_data.py`
5. **Train Stage 1 (single window):** `python -m models.train_predictor --stock RELIANCE --window 0`
6. **Train Stage 1 (all windows):** `python -m models.train_predictor --stock RELIANCE`
7. **Evaluate Stage 1:** `python -m models.evaluate --stock RELIANCE`
8. **HPO Search:** `python -m tuning.optuna_search`
9. **Train RL Agent (single window):** `python -m rl.train_agent --stock RELIANCE --window 0`
10. **Run Backtest:** `python -m evaluation.walk_forward --stock RELIANCE`
11. **Run Backtest All Stocks:** `python -m evaluation.walk_forward`

**Pipeline orchestration:** `evaluation/walk_forward.py` orchestrates the full end-to-end pipeline (Load data → Stage 1 inference → RL agent decisions → Metrics → Charts → JSON results). If a window's RL model is missing, it is trained on the fly and cached (`results/STOCK/models/rl_windowN.zip`).

---

## 4. Data Flow & State Management Patterns
- Data strictly flows forward. Look-ahead bias is strictly prevented.
- **`features/pipeline.py` intentionally does NOT normalize data.** Normalization (`RobustScaler`) MUST happen *dynamically inside the walk-forward training loop* using only statistics from the training window. Binary flags (`vix_available`) are never scaled.
- The target is a **5-day forward return with a ±0.5% neutral zone**: `1.0` (UP), `0.0` (DOWN), `-1.0` (NEUTRAL). Neutral-labeled sequences are filtered out during training but kept in the CSV for chronological continuity.
- The `forward_return` intermediate column is **never saved to the CSV** — only `target` survives. This prevents future-value leakage.
- **Feature selection is dynamic.** Features are inferred from CSV columns by excluding `{"date", "timestamp", "target"}`. The inferred feature list is saved inside every `.pt` checkpoint.
- **Walk-forward boundary handling:** The last `horizon` (5) rows of each train/test window have their targets set to `-1.0` (neutral) to prevent leakage across splits.

---

## 5. API Contracts & External Services
- **Upstox API:** Used exclusively in `Data/upstox.py` to pull historical OHLCV data. Requires a `.env` file with `ACCESS_TOKEN`. No live trading API is connected yet.

---

## 6. Environment Variables, Config, and Setup
- **Environment:** Use the local virtual environment (`env/`).
- **Dependencies:** First, install PyTorch with CUDA 12.6: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. Then, install the remaining dependencies via `pip install -r requirements.txt`.
- **Config:** `config.yaml` is the ultimate source of truth. All scripts should read parameters (like `seq_length`, `batch_size`, paths) from here. Do not hardcode parameters in Python files.
- **Secrets:** `Data/.env` requires `ACCESS_TOKEN="your_upstox_token"` (if downloading new data).

---

## 7. Build, Run, Test, and Deploy Commands
- **Activate Env (Windows):** `.\env\Scripts\Activate.ps1`
- **Run Feature Pipeline:** `python -m features.pipeline`
- **Validate Data:** `python Data/validate_data.py`
- **Train Stage 1:** `python -m models.train_predictor --stock RELIANCE`
- **Train Stage 1 (single window):** `python -m models.train_predictor --stock RELIANCE --window 0`
- **Evaluate Stage 1:** `python -m models.evaluate --stock RELIANCE`
- **Optuna HPO:** `python -m tuning.optuna_search`
- **Train RL Agent:** `python -m rl.train_agent --stock RELIANCE --window 0`
- **Full Backtest:** `python -m evaluation.walk_forward --stock RELIANCE`
- **Full Backtest All Stocks:** `python -m evaluation.walk_forward`

---

## 8. Coding Conventions, Rules & Style Preferences
- **Detailed Comments:** The user strongly prefers **detailed and accurate comments everywhere in the code**. Every major block of logic inside a function should have an inline comment explaining *what* it does and *why*.
- **Look-Ahead Bias Prevention:** You must obsessively guard against look-ahead bias. Never use future data to scale, normalize, or compute indicators for past data.
- **Error Handling:** Fail fast. See `features/pipeline.py` NaN checks. If data is corrupted, raise a `ValueError` immediately rather than letting models train on garbage.
- **Type Hinting:** Use standard Python type hints (`df: pd.DataFrame`, `-> pd.Series`) for all function signatures.
- **Proactive Documentation:** You must update `For_AI.md`, `extra/System_Theory_and_Design.md`, and `extra/final_implementation_plan.md` at any point in time if and whenever needed to reflect the current state of the project.
- **Staged Execution:** If an implementation plan is large (multiple files or complex logic), ALWAYS execute it in stages rather than trying to do everything at once. Break the work into logical chunks, complete each chunk fully (write code, test, verify), and then move to the next. This prevents overwhelming the context window and avoids hitting output token limits.

---

## 9. Repeated Patterns
- **`df.copy()`:** Functions that modify DataFrames (like indicator additions) always start with `result = df.copy()` to avoid `SettingWithCopyWarning` and mutating global state.
- **Config Loading:** Scripts load `config.yaml` using the `load_config()` utility pattern at the start of their execution.

---

## 10. Known Limitations, Edge Cases & TODOs
- **VIX Data Gap:** India VIX data only exists post-2009. We fill pre-2009 VIX with `0.0` and use a `vix_available` binary flag so the network learns to ignore it instead of hallucinating a zero-volatility regime.
- **TCS Data Gap:** TCS IPO'd in 2004, so it has fewer walk-forward windows than Reliance. Scripts must dynamically calculate valid walk-forward windows based on actual data availability.
- **Neutral Target Bias:** With a 5-day ±0.5% target, "clear" labels skew ~54-56% UP due to structural market drift. Evaluation must compare against the **majority-class baseline**, not 50%. MCC and balanced accuracy are the primary metrics.
- **TODOs (in priority order):** 
  1. ~~Build `features/` module~~ ✅ Done
  2. ~~Run `validate_data.py` to confirm data integrity~~ ✅ Done
  3. ~~Build `models/gru_attention.py`~~ ✅ Done
  4. ~~Build `models/train_predictor.py`~~ ✅ Done
  5. ~~Build `models/inference.py`~~ ✅ Done
  6. ~~Build `models/evaluate.py`~~ ✅ Done
  7. ~~Stage 1 Data Quality Upgrade (Phase 2.5)~~ ✅ Done
  8. ~~Optuna hyperparameter search (`tuning/optuna_search.py`)~~ ✅ Done
  9. ~~Build `rl/trading_env.py` — Gymnasium env with action masking~~ ✅ Done
  10. ~~Build `rl/reward.py` + `rl/train_agent.py`~~ ✅ Done
  11. ~~Build `evaluation/walk_forward.py` — end-to-end orchestrator~~ ✅ Done
  12. ~~Build `evaluation/metrics.py` — risk metric functions~~ ✅ Done
  13. ~~Build `evaluation/backtest.py` — 3-way strategy comparison + quantstats tearsheet~~ ✅ Done
  14. ~~Build `evaluation/visualise.py` — equity curves, drawdown, trade scatter~~ ✅ Done
  15. Post-MVP Streamlit dashboard ← CURRENT
