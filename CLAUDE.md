# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-stage hybrid DL + RL pipeline for daily trading of 5 NIFTY 50 stocks (RELIANCE, TCS, HDFCBANK, HINDUNILVR, SUNPHARMA) over ~25 years of data:

- **Stage 1 (models/):** GRU + Multi-Head Attention + MC Dropout (PyTorch) predicts the probability of a stock going UP over the next 5 trading days (`p_up`) plus an uncertainty-based `confidence` score (T=50 MC Dropout passes).
- **Stage 2 (rl/):** MaskablePPO (sb3-contrib) consumes Stage 1 predictions and makes BUY/SELL/HOLD decisions in a custom Gymnasium env with transaction costs, optimizing a Sortino-shaped reward.
- **Evaluation (evaluation/):** Walk-forward backtest comparing Buy-and-Hold vs Predictor-Only vs Full RL on every test window.

**Read `For_AI.md` first** — it is the onboarding dossier and must be kept up to date. `extra/final_implementation_plan.md` holds the definitive phase checklist; `extra/phase_4_implementation_plan.md` documents the evaluation module design and its deviations from the original plan.

## Commands

```powershell
.\env\Scripts\Activate.ps1                                  # Activate venv (Windows)

python Data/upstox.py                                       # Download raw data (needs Data/.env with ACCESS_TOKEN)
python -m features.pipeline                                 # Feature engineering → Data/processed/*_features.csv
python Data/validate_data.py                                # Assert no NaN/Inf in processed data

python -m models.train_predictor --stock RELIANCE           # Train Stage 1, all walk-forward windows
python -m models.train_predictor --stock RELIANCE --window 0  # Single window (fast smoke test)
python -m models.evaluate --stock RELIANCE                  # Stage 1 metrics: AUC, MCC, balanced acc, calibration
python -m tuning.optuna_search                              # Hyperparameter search

python -m rl.train_agent --stock RELIANCE --window 0        # Train Stage 2 RL agent for one window

python -m evaluation.walk_forward --stock RELIANCE --window 0  # End-to-end backtest, one window
python -m evaluation.walk_forward --stock RELIANCE          # All windows for one stock
python -m evaluation.walk_forward                           # All stocks, all windows
```

There is no test suite. Verification = running the relevant pipeline stage on `--stock RELIANCE --window 0` and checking console metrics + outputs under `results/`.

Dependencies: install PyTorch with CUDA first (`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`), then `pip install -r requirements.txt`.

## Architecture & Data Flow

```
Data/upstox.py → Data/historical_data/*.csv (raw, gitignored)
    → features/pipeline.py → Data/processed/<stock>_features.csv   (NOT normalized; regenerate after any pipeline change)
    → models/train_predictor.py → results/<STOCK>/models/windowN.pt + norm_stats/windowN_scaler.pkl
    → rl/train_agent.py (OOF predictions → MaskablePPO) → results/<STOCK>/models/rl_windowN.zip
    → evaluation/walk_forward.py → backtest_results.json, plots/, tearsheets/
```

Cross-file contracts to know before editing:

- **`models/train_predictor.py` is the shared utility hub.** `load_config`, `get_feature_cols`, `get_scale_cols`, `compute_walk_forward_windows`, `split_by_window`, `normalize_features`, `set_seed`, `get_device` are imported by both `rl/train_agent.py` and `evaluation/walk_forward.py`. Changing their signatures breaks three modules.
- **Prediction DataFrame contract:** `[date, close, india_vix, p_up, confidence]` is the interface between Stage 1 inference, `rl/trading_env.TradingEnv`, and `evaluation/backtest.py`.
- **Checkpoints are self-describing:** every `.pt` stores its `feature_cols` list; inference reads features from the checkpoint, never from a hardcoded list. Scalers are saved per window as `.pkl` and reused at evaluation time.
- **TradingEnv:** 11-dim observation (`p_up, confidence, position, unrealised_pnl, days_in_position, vix_norm, 5×recent_returns`), `Discrete(3)` action space with `action_masks()` (long-only: flat→{HOLD,BUY}, long→{HOLD,SELL}), transaction costs from `config.yaml`. Episodes start at step 5 (needs return history).
- **RL training uses Out-Of-Fold predictions:** `rl/train_agent.py` splits each train window 70/30 chronologically, trains a throwaway Stage 1 model on the 70%, predicts on the 30%, and trains the RL agent on those predictions — the agent never sees in-sample Stage 1 outputs.
- **`evaluation/walk_forward.py` trains missing RL models on the fly** and caches them (first run slow, reruns instant).

## Critical Invariants (do not violate)

- **Look-ahead bias is the cardinal sin.** `features/pipeline.py` intentionally does NOT normalize. RobustScaler is fit *inside* the walk-forward loop on training-window data only; evaluation loads the saved scaler and never refits on test data. Never use future data to scale, normalize, or compute indicators for the past.
- **`forward_return` never reaches disk.** Only the encoded `target` column survives in processed CSVs.
- **Target encoding:** 5-day forward return with ±0.5% neutral zone → `1.0` UP / `0.0` DOWN / `-1.0` NEUTRAL. Neutral sequences are filtered during *training* (`filter_neutrals=True`) but kept for *inference/RL* (`filter_neutrals=False` — the env needs contiguous daily predictions). The last `horizon` (5) rows of each window split are forced to `-1.0` to prevent leakage across boundaries.
- **Dynamic feature selection:** features = all CSV columns minus `{date, timestamp, target}`. Binary flags (`vix_available`) are never scaled. Don't hardcode feature lists.
- **`config.yaml` is the single source of truth** for hyperparameters, paths, costs, and walk-forward settings. Never hardcode parameters in Python files. Current model hyperparameters are Optuna-tuned — don't change them casually.
- **Stage 1 evaluation baseline is the majority class (~54-56% UP), not 50%.** MCC and balanced accuracy are the primary classification metrics; Sortino is the primary trading metric.
- **Walk-forward windows vary per stock** (TCS IPO'd 2004; VIX exists only post-2009 → `vix_available` flag). Always compute windows dynamically from actual data.

## Conventions

- **Detailed comments everywhere** — the user strongly prefers inline comments explaining *what* and *why* for every major logic block. Match the existing plain-language comment style.
- Fail fast: raise `ValueError` immediately on corrupt/NaN data rather than training on garbage.
- Type hints on all function signatures; functions that modify DataFrames start with `result = df.copy()`.
- Windows console: scripts call `sys.stdout.reconfigure(encoding="utf-8")` — keep this in new entry points.
- **Proactively update `For_AI.md`** (and `extra/` plan docs) whenever the project state changes.
- For large implementations, execute in stages: complete and verify each chunk before starting the next.
