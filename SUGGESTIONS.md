# Senior Engineer Review — Making This Project Resume-Worthy

**Reviewer lens:** Senior AI + Quant Engineer. The goal here is not miraculous returns — it's making the project *defensible in an interview*. A quant or ML interviewer will probe two things: (1) did you avoid the classic backtesting sins, and (2) do you engineer like a professional. Strong results are a bonus; rigorous methodology and honest reporting are the actual resume material.

The good news: the bones are genuinely solid. Walk-forward validation, OOF predictions for the RL agent, saved per-window scalers, neutral-zone labels, majority-class baselines, transaction costs — most hobby projects have *none* of these. The suggestions below are ordered by impact-per-effort.

---

## Tier 1 — Do These First (highest resume impact, low-to-medium effort)

### 1.1 Write a real README.md (there isn't one)
The repo has `For_AI.md` (for AI assistants) but no human-facing README. **This is the single highest-impact item** — recruiters and interviewers see the README before anything else. It should contain:

- A one-paragraph pitch and an architecture diagram (Mermaid renders natively on GitHub):
  `Raw OHLCV → Feature Engineering → GRU+Attention+MC Dropout → p_up + confidence → MaskablePPO → Trades → Walk-Forward Backtest`
- A **results table**: per-stock aggregate Sortino / Sharpe / MaxDD / Return for all 3 strategies (Buy-Hold vs Predictor-Only vs Full RL). Pull straight from `backtest_results.json`.
- 2–3 of your best charts (equity curve overlay, aggregate summary table) committed to a `docs/img/` folder.
- A "Methodology & Bias Controls" section listing what you guarded against (look-ahead, leakage at window boundaries, scaler refitting, neutral-zone labels). Interviewers eat this up.
- A short "Limitations" section (see 1.2). Honesty reads as seniority.
- Quickstart commands.

### 1.2 Document known biases honestly — especially survivorship bias
The universe is 5 hand-picked stocks that are *today's* mega-caps (RELIANCE, TCS, HDFCBANK, HINDUNILVR, SUNPHARMA). Backtesting 25 years on companies you already know survived and thrived inflates every strategy, including buy-and-hold. You don't need to fix it (point-in-time universe data is expensive) — you need to **name it**. Add a Limitations section covering:

- Survivorship bias (stocks chosen with hindsight).
- Long-only, single-stock-at-a-time, no portfolio effects.
- Daily close-to-close fills (no intraday slippage modeling beyond the flat %).
- No corporate-action handling beyond what Upstox adjusts.

An interviewer who hears you volunteer these unprompted will trust everything else you say.

### 1.3 Add a test suite (pytest) — the #1 engineering signal
There are currently zero tests. For a project whose core claim is "no look-ahead bias," tests are how you *prove* the claim. Priority order:

1. **The gold-standard leakage test:** run `features/pipeline.py` on a full series, then on the same series truncated by N days; assert all overlapping feature rows are *identical*. If any feature changes when future data is removed, you have leakage. This single test is an interview story by itself.
2. **Metrics unit tests:** `evaluation/metrics.py` functions are pure — test against hand-computed values (known returns array → known Sortino/Sharpe/MaxDD). Include edge cases: all-positive returns, empty trades, single trade.
3. **TradingEnv invariants:** portfolio accounting (buy→hold→sell with known prices yields the exact expected portfolio value including costs), action masks never permit SELL while flat or BUY while long, episode terminates at the right step.
4. **Target encoding test:** synthetic price series where you know the 5-day forward returns → assert correct 1.0/0.0/-1.0 labels and that the last `horizon` rows are neutralized.

~1 day of work, transforms the project's credibility.

### 1.4 Add CI (GitHub Actions) + linting
A `ci.yml` that runs `ruff check` + `pytest` on push (CPU-only; skip GPU tests with markers). Add a `ruff` config to `pyproject.toml`. Green badge in the README. Trivial effort, outsized "this person ships professionally" signal.

### 1.5 Statistical honesty: report variance, not point estimates
Two cheap upgrades that scream rigor:

- **Multiple RL seeds:** PPO results from a single seed are noise. Train each window's agent with 3–5 seeds and report mean ± std of Sortino in the aggregate table. (Yes, it multiplies training time — do it for one showcase stock if needed.)
- **Bootstrap confidence intervals:** resample daily returns (block bootstrap, ~20-day blocks to preserve autocorrelation) and report a 95% CI on Sortino/Sharpe per strategy. ~50 lines in `evaluation/metrics.py`. Saying "Full-RL Sortino 0.9 [CI: 0.4–1.3] vs Buy-Hold 0.6 [CI: 0.2–1.0]" instead of "my Sortino is 0.9" is the difference between a student project and quant work.

---

## Tier 2 — Methodological Fixes (things an interviewer might catch)

### 2.1 The RL reward shaping is at risk of reward hacking — audit it
In `rl/reward.py`, the `action_bonus` of **+1.0** for a confident BUY is the same magnitude as a +1% daily portfolio return (`daily_return * 100`). Daily moves average well under 1%, so the shaping bonus can *dominate* the actual P&L signal — the agent may be learning "follow the predictor to farm bonuses" rather than "maximize risk-adjusted return." The current design makes Full-RL structurally similar to Predictor-Only with extra steps.

Recommended:
- **Log reward components separately** (base / opportunity-cost / bonus) in TensorBoard via the existing `LoggingCallback` to see what fraction of total reward is shaping. If bonuses dominate, that's a problem.
- **Anneal the bonus to zero** over training (e.g., linear decay over the first 50% of timesteps). It exists to overcome cost-aversion during exploration; it shouldn't shape the final policy.
- Mention "potential-based reward shaping" in your docs as the principled framing (Ng et al., 1999) — even partial adherence is a great talking point.

### 2.2 Fix the VIX normalization constant
`rl/trading_env.py` normalizes VIX as `row["india_vix"] / 50.0` with a comment claiming a "historical median of ~50". India VIX's long-run median is ~15–17 (it spiked to ~80 only in 2008 and ~2020). Dividing by 50 compresses the feature into ~0.2–0.4 for almost all observations, muting the regime signal. Either divide by ~20, or better, use a rolling z-score / percentile rank computed from training data only. Small fix, but a quant reviewer reading the env will notice the comment is factually wrong.

### 2.3 Calibrate `p_up` and prove `confidence` means something
You feed raw sigmoid outputs to the RL agent. Two additions:

- **Reliability diagram + Brier score** per window in `models/evaluate.py` outputs (you already track calibration — surface it as a chart in the README).
- **Temperature scaling** fit on a validation slice (single scalar parameter, ~20 lines) before predictions reach the RL agent. "I calibrated the classifier before downstream consumption" is a senior-level move.
- One scatter plot: MC-Dropout confidence vs. realized accuracy (binned). If higher confidence → higher accuracy, your uncertainty estimate works — *that's a headline chart*. If it doesn't, report that honestly and discuss.

### 2.4 Benchmark against the NIFTY 50 index, not just the stock itself
`backtest.py` compares against buy-and-hold of *the same stock*. Add a 4th reference curve: buy-and-hold of the NIFTY 50 index (data already exists at `Data/historical_data/nifty50_daily.csv`). "Did the system beat just buying the index?" is the question a practitioner actually asks.

### 2.5 Transaction-cost sensitivity analysis
One loop, big credibility: rerun the backtest with 0×, 0.5×, 1×, 2× the configured costs and plot Sortino vs. cost multiplier per strategy. Shows you understand that cost assumptions make or break daily-frequency strategies — and reveals whether the RL agent's edge (if any) survives realistic friction.

### 2.6 Verify the ensemble is actually wired up
`config.yaml` has `ensemble_count: 3`, but make sure `evaluation/walk_forward.py` actually loads/averages an ensemble at inference (it currently loads a single `windowN.pt`). Either implement ensemble inference end-to-end or set the config to 1 — a config key that silently does nothing in the eval path is the kind of inconsistency reviewers find.

---

## Tier 3 — Code Quality Cleanups (quick wins)

1. **Dead code in `evaluation/backtest.py`** (`run_predictor_only`, ~lines 163–166): a convoluted inline `trade_returns.append(...)` computation whose result is discarded — `_compute_trade_returns()` is the real source. Delete it.
2. **Remove `_compare_metrics.py`** from the repo root (scratch file) or move it into `extra/`.
3. **Pin dependencies for reproducibility:** keep `requirements.txt` loose, but add a `requirements.lock` (`pip freeze`) so results are reproducible. Mention the seed policy in the README.
4. **`Data/` vs `data/` casing:** the folder on disk is lowercase `data/` while code/docs reference `Data/`. Works on Windows, breaks on Linux/CI. Standardize on one casing (this matters once CI from 1.4 exists).
5. **Consolidate `load_config`:** defined in both `features/pipeline.py` and `models/train_predictor.py`. Move shared utilities (`load_config`, seeding, device) into a small `common/` module so `rl/` and `evaluation/` stop importing training internals just for config loading.

---

## Tier 4 — Stretch Features (genuine differentiators, pick 1–2)

### 4.1 Position sizing via confidence (best bang-for-buck extension)
Extend the action space from `Discrete(3)` to sized entries (e.g., BUY_25 / BUY_50 / BUY_100% of capital), letting the agent express conviction. This converts the project from "binary signal follower" to "risk allocator" — a much stronger story, and MaskablePPO handles it with minimal env changes.

### 4.2 Regime breakdown analysis
Slice backtest results by VIX regime (low/mid/high terciles) and by calendar year. One chart: "the RL agent adds most value in high-VIX regimes" (or doesn't — either is publishable in a README). Cheap to compute from existing trade logs.

### 4.3 Finish the Streamlit dashboard (already planned, currently a 2-line stub)
Three tabs as planned: live signals, backtest explorer, attention heatmaps. A deployed dashboard link on the resume (Streamlit Community Cloud is free) lets non-technical reviewers *see* the project in 10 seconds. The attention-weights tab (`get_attention_weights()` already exists in `models/gru_attention.py`) doubles as your interpretability story.

### 4.4 Name-drop the literature in your docs
You're already doing embargo-style boundary neutralization and walk-forward analysis. Reference the formal concepts in `extra/System_Theory_and_Design.md`: purged walk-forward / embargo (López de Prado, *Advances in Financial Machine Learning*), Deflated Sharpe Ratio, probability calibration (Guo et al., 2017). Knowing the canonical names for what you built is exactly what separates "I followed a tutorial" from "I understand the field."

---

## What NOT to spend time on

- **Chasing accuracy.** 52–55% directional accuracy with a real edge after costs is realistic; 60%+ on daily equities means you have a bug (almost always leakage). Your MCC-vs-majority-baseline framing is already correct — keep it.
- **More stocks / more data.** Five liquid names across 25 years is plenty for the story. Breadth adds compute, not credibility.
- **Live trading integration.** Paper-trade hooks sound cool but add operational risk and zero interview value compared to Tier 1 items.
- **Swapping architectures (Transformers, Mamba, etc.).** The GRU+Attention choice is defensible for daily data with ~6k samples per window; an architecture-of-the-week swap would *reduce* the signal that you make reasoned choices.

---

## Suggested resume bullets (after Tier 1 + a couple of Tier 2 items)

> - Built a two-stage ML trading system (PyTorch GRU+Attention with MC-Dropout uncertainty → MaskablePPO execution agent) evaluated over 25 years of NIFTY 50 data via strict walk-forward validation with per-window normalization and embargoed label boundaries to eliminate look-ahead bias.
> - Engineered a 3-way backtesting framework (buy-and-hold / signal-only / RL) with transaction-cost modeling, block-bootstrap confidence intervals, and cost-sensitivity analysis; validated the pipeline with automated leakage tests in CI.
> - Quantified model uncertainty via 50-pass Monte Carlo Dropout and demonstrated confidence-accuracy correlation, using calibrated probabilities as state inputs to the RL agent.

Note these bullets emphasize *methodology* (walk-forward, embargo, bootstrap CIs, leakage tests) over returns. That's deliberate — returns claims invite skepticism; rigor claims invite questions you can actually answer well.
