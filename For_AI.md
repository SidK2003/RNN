# For_AI: Onboarding Guide & Project Reference

**Notice to AI Assistant:** Read this file *first* to understand the NIFTY 50 Deep RL Trading System. It is the single source of truth for architecture, rules, and existing patterns. Do not blindly read all source files unless explicitly required. 
*Also, heavily refer to `extra/final_implementation_plan.md`, as it contains the definitive checklist and detailed hyperparameter parameters for each phase.*

---

## 1. Project Purpose & High-Level Architecture
This is a two-stage hybrid Deep Learning and Reinforcement Learning pipeline designed for daily equity trading of NIFTY 50 stocks. 
- **Stage 1 (Deep Learning):** A `GRU + Multi-Head Attention + MC Dropout` PyTorch model predicts the probability of a stock going UP the next day and provides a confidence score based on Monte Carlo Dropout uncertainty.
- **Stage 2 (Reinforcement Learning):** A `MaskablePPO` (sb3-contrib) agent takes the Stage 1 predictions and makes discrete trading decisions (`BUY`, `SELL`, `HOLD`), explicitly optimizing for the **Sortino Ratio** while navigating transaction costs and slippage.

Validation is performed via a strict **Walk-Forward** rolling window mechanism (10 years train, 2 years test) across ~25 years of data.

---

## 2. Folder/File Structure
```text
RNN/
├── config.yaml               # Central hyperparameter, path, and config registry
├── requirements.txt          # Python dependencies
├── extra/
│   └── final_implementation_plan.md # The COMPLETE, detailed phase-by-phase implementation plan and specifications.
├── System_Theory_and_Design.md # Theory explanation behind the chosen architecture
├── For_AI.md                 # (This File) AI onboarding guide
├── Data/
│   ├── historical_data/      # Raw Upstox CSVs (DO NOT push to git)
│   ├── processed/            # Feature-engineered CSVs ready for DL models
│   ├── upstox.py             # Script to download daily data from Upstox API
│   └── validate_data.py      # Script to ensure no NaNs/Infs in processed data
├── features/                 # Stage 0: Feature Engineering
│   ├── pipeline.py           # Main ETL orchestrator
│   ├── technical_indicators.py # RSI, MACD, Bollinger, ATR, etc.
│   └── vix.py                # India VIX merge logic & gap handling
├── models/                   # Stage 1: PyTorch GRU+Attention (TODO)
├── rl/                       # Stage 2: Gymnasium Env & MaskablePPO (TODO)
├── evaluation/               # Metrics, walk-forward orchestrator, visualizations (TODO)
├── tuning/                   # Optuna hyperparameter searches (TODO)
└── dashboard/                # Post-MVP Streamlit UI (TODO)
```

---

## 3. Main Entry Points & Execution Flow
*(Currently only Stage 0 is fully implemented)*
1. **Fetch Data:** `python Data/upstox.py`
2. **Process Features:** `python -m features.pipeline` (Reads raw CSVs, adds features & targets, saves to `Data/processed/`)
3. **Validate Data:** `python Data/validate_data.py` (Ensures data integrity before Stage 1)

**Future Flow:** `evaluation/walk_forward.py` will orchestrate the full pipeline (Train Stage 1 -> Generate Out-of-Fold Predictions -> Train Stage 2 -> Test).

---

## 4. Data Flow & State Management Patterns
- Data strictly flows forward. Look-ahead bias is strictly prevented.
- **`features/pipeline.py` intentionally does NOT normalize data.** Normalization (z-scores) MUST happen *dynamically inside the walk-forward training loop* using only statistics from the training window.
- The target is **binary classification** (`1.0` if tomorrow's close > today's close, `0.0` otherwise).

---

## 5. API Contracts & External Services
- **Upstox API:** Used exclusively in `Data/upstox.py` to pull historical OHLCV data. Requires a `.env` file with `ACCESS_TOKEN`. No live trading API is connected yet.

---

## 6. Environment Variables, Config, and Setup
- **Environment:** Use the local virtual environment (`env/`).
- **Dependencies:** Install via `pip install -r requirements.txt`.
- **Config:** `config.yaml` is the ultimate source of truth. All scripts should read parameters (like `seq_length`, `batch_size`, paths) from here. Do not hardcode parameters in Python files.
- **Secrets:** `Data/.env` requires `ACCESS_TOKEN="your_upstox_token"` (if downloading new data).

---

## 7. Build, Run, Test, and Deploy Commands
- **Activate Env (Windows):** `.\env\Scripts\Activate.ps1`
- **Run Pipeline:** `python -m features.pipeline`
- **Validate Data:** `python Data/validate_data.py`
*(Tests and deployment are pending further implementation)*

---

## 8. Coding Conventions, Rules & Style Preferences
- **Detailed Comments:** The user strongly prefers **detailed and accurate comments everywhere in the code**. Every major block of logic inside a function should have an inline comment explaining *what* it does and *why*.
- **Look-Ahead Bias Prevention:** You must obsessively guard against look-ahead bias. Never use future data to scale, normalize, or compute indicators for past data.
- **Error Handling:** Fail fast. See `features/pipeline.py` NaN checks. If data is corrupted, raise a `ValueError` immediately rather than letting models train on garbage.
- **Type Hinting:** Use standard Python type hints (`df: pd.DataFrame`, `-> pd.Series`) for all function signatures.

---

## 9. Repeated Patterns
- **`df.copy()`:** Functions that modify DataFrames (like indicator additions) always start with `result = df.copy()` to avoid `SettingWithCopyWarning` and mutating global state.
- **Config Loading:** Scripts load `config.yaml` using the `load_config()` utility pattern at the start of their execution.

---

## 10. Known Limitations, Edge Cases & TODOs
- **VIX Data Gap:** India VIX data only exists post-2009. We fill pre-2009 VIX with `0.0` and use a `vix_available` binary flag so the network learns to ignore it instead of hallucinating a zero-volatility regime.
- **TCS Data Gap:** TCS IPO'd in 2004, so it has fewer walk-forward windows than Reliance. Scripts must dynamically calculate valid walk-forward windows based on actual data availability.
- **TODOs:** 
  - Build `models/gru_attention.py`
  - Build `rl/trading_env.py`
  - Build the walk-forward orchestrator.
  - Setup Post-MVP Streamlit dashboard.
