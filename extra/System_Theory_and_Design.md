# NIFTY 50 Deep RL Trading System: Theory & System Design

This document serves as a comprehensive summary of the project's conceptual foundation, the architectural choices, and the specific implementations completed so far. It explains *what* we are building and, more importantly, *why* we are building it this way (in simple words).

---

## 1. System Architecture Overview

The system is a two-stage hybrid Deep Learning and Reinforcement Learning pipeline designed for daily equity trading. 

Instead of training a single monolithic RL agent to look at raw prices and make trades (which often fails due to noise and non-stationarity), we separate the problem into two distinct stages:

1. **Stage 1 (The Analyst):** A Deep Learning model (GRU + Attention) that looks at historical data and predicts the *probability of the stock going UP tomorrow*. It also outputs a *confidence score*.
2. **Stage 2 (The Trader):** A Reinforcement Learning agent (MaskablePPO) that takes the Analyst's predictions, factors in transaction costs, portfolio state, and market volatility, and decides whether to BUY, SELL, or HOLD.

---

## 2. Completed Implementation: Data & Feature Engineering (Stage 0)

We have successfully built the entire feature engineering pipeline (`features/pipeline.py`), which processes raw OHLCV data into a state ready for Stage 1.

### 2.1 The Data
We are using daily OHLCV (Open, High, Low, Close, Volume) data from Upstox for 5 major cross-sector NIFTY 50 stocks (RELIANCE, TCS, HDFCBANK, HINDUNILVR, SUNPHARMA) alongside the India VIX index. The data spans from 2000 (or IPO date) to 2026.

### 2.2 Feature Selection & Theory
Financial markets are highly noisy and non-stationary. Raw prices cannot be fed directly into a neural network. We compute features that capture different market dynamics:

*   **Price Action & Volatility:**
    *   **Log Returns:** $\ln(P_t / P_{t-1})$. Converts exponential compounding into additive, stationary data.
    *   **Bollinger %B:** Measures price relative to recent volatility. Shows if a stock is statistically overbought/oversold.
    *   **ATR (Average True Range):** A pure measure of market volatility.
*   **Momentum:**
    *   **RSI (Relative Strength Index):** Measures the speed and change of price movements.
    *   **MACD (Moving Average Convergence Divergence):** Captures trend strength by comparing short-term and long-term momentum.
    *   **Stochastic Oscillator:** Compares a particular closing price to a range of its prices over a certain period.
*   **Volume (Conviction):**
    *   **OBV (On-Balance Volume):** Adds volume on up-days, subtracts on down-days. Captures institutional accumulation/distribution.
    *   **Volume SMA Ratio:** Flags unusual volume spikes.
*   **Macro Regime:**
    *   **India VIX:** Captures broader market fear/complacency.
    *   *Implementation Trick:* India VIX data only exists post-2009. Instead of dropping pre-2009 stock data or filling it with fake numbers, we fill it with `0.0` and add a `vix_available` binary flag (0 or 1). This allows the neural network to learn: *"Ignore the VIX value when vix_available=0."*

### 2.3 The Target Variable
We frame Stage 1 as a **Binary Classification** problem:
`target = 1 if Close(tomorrow) > Close(today) else 0`

**Theory:** Predicting exact future prices (regression) in finance is notoriously difficult because a stock moving from 100 to 101 vs 100 to 105 are both "UP" moves, but the latter creates huge regression errors. Predicting *direction* is much cleaner, bounds the loss function [0,1], and provides a clear probability $P(UP)$.

### 2.4 Bias Prevention (CRITICAL)
A major reason AI trading bots fail in live trading is **Look-Ahead Bias** (using data from the future).
*   **Indicator Calculation:** All indicators use `rolling()` windows. At day $T$, the features only "see" day $T$ and before.
*   **Normalization:** We deliberately *did not* normalize the dataset in the pipeline. If we normalized the whole dataset at once, the mean/std from 2024 would leak into the data of 2010. Normalization will happen *during training* on a per-window basis.

---

## 3. Future Implementation: Validation & Evaluation

### 3.1 Walk-Forward Validation
Financial regimes change. A model trained in a 2010 bull market might fail in a 2020 crash. 
Instead of a simple Train/Test split, we use **Walk-Forward Validation**:
*   Train on 10 years (e.g., 2000-2009).
*   Test on the next 2 years (2010-2011).
*   Step forward 2 years and repeat (Train 2002-2011, Test 2012-2013).

This gives us an honest assessment of how the model would have performed if deployed live, retraining periodically.

---

## 4. Completed Implementation: Stage 1 (Deep Learning Predictor)

We have successfully built the Stage 1 Neural Network and its training pipeline. This lives inside the `models/` directory.

### 4.1 PyTorch Dataset (`models/dataset.py`)
Neural networks cannot process raw spreadsheets. We built a "sliding window" mechanism that chops the data into 60-day sequences. For example, it looks at Days 1-60 to predict Day 61. It then slides forward to look at Days 2-61 to predict Day 62.

### 4.2 GRU + Multi-Head Attention (`models/gru_attention.py`)
This is the "Brain" of the Analyst.
*   **GRU (Gated Recurrent Unit):** Processes the 60-day sequences day-by-day, building a "memory" of the trend. It's faster and less prone to overfitting than older LSTM models.
*   **Multi-Head Attention:** Instead of just relying on the GRU's final memory state, this mechanism allows the AI to look back over the full 60 days and assign "importance weights" to specific events (like a huge volume spike 40 days ago).

### 4.3 Training Pipeline (`models/train_predictor.py`)
This is the "Gym" where the AI learns. It automatically handles:
*   **Walk-Forward Training:** Slices data into rolling 10-year train / 2-year test windows.
*   **Anti-Bias Normalization:** Converts raw numbers into standard scales (Z-Scores) using *only* the training data, ensuring the AI never "looks into the future."
*   **High-Speed AMP:** Uses the RTX 4070's Tensor Cores (Automatic Mixed Precision - FP16) to train the model twice as fast while using half the VRAM.
*   **Early Stopping:** Prevents the AI from "memorizing" the test answers by automatically halting training if the test score stops improving.

### 4.4 MC Dropout Inference (`models/inference.py`)
Standard neural networks are notoriously overconfident. They might output $P(UP) = 0.9$ even if they have never seen a pattern before.
*   **Theory:** By keeping "Dropout" (randomly turning off artificial brain cells) *turned on* during inference and running the data through the network 50 times, we get 50 slightly different predictions.
*   If the predictions are tightly clustered (0.80, 0.81, 0.79), the model has high **Confidence**.
*   If the predictions are scattered (0.20, 0.90, 0.50), the model is uncertain. We mathematically measure this spread (Standard Deviation) and pass it to the Stage 2 RL Agent, so it knows when to avoid trading during highly uncertain regimes.

---

## 5. Completed Implementation: Stage 2 (RL Agent)

We have successfully built the Stage 2 Reinforcement Learning agent, its environment, and the training orchestrator. This lives inside the `rl/` directory.

### 5.1 Trading Environment (`rl/trading_env.py`)
This is the custom Gymnasium environment where the RL agent interacts with the market.
*   **Observation Space (11-dim):** The agent receives the probability of an UP move (`p_up`), the `confidence` score from Stage 1, current `position`, unrealized P&L, normalized days in position, normalized India VIX, and a 5-day history of portfolio returns.
*   **Action Space & Masking:** The agent can output HOLD (0), BUY (1), or SELL (2). We use **Maskable PPO** to mathematically zero out illegal actions (e.g., BUYing when already long, or SELLing when flat). This prevents the agent from wasting time exploring invalid trades.
*   **Transaction Costs:** We deduct **brokerage (0.1%)**, **STT/SEBI charges (0.01%)**, and **slippage (0.05%)** during every trade directly from the portfolio. This naturally discourages overtrading without needing an artificial static penalty.

### 5.2 The Reward Function (`rl/reward.py`)
Traditional RL bots optimize for maximum profit, which often leads to catastrophic drawdowns. We shape the reward to optimize for risk-adjusted returns (Sortino-style).
*   We penalize negative returns (losses) 2x more heavily than we reward positive returns.
*   The reward directly reflects the portfolio's step-by-step percentage return (including transaction costs).

### 5.3 OOF Training Pipeline (`rl/train_agent.py`)
A massive risk in two-stage RL systems is training the RL agent on the same data the DL model trained on. The DL model is highly confident on its training data, causing the RL agent to learn unrealistic expectations.
*   **Out-Of-Fold (OOF) Generation:** We split every walk-forward training window 80/20. We train a temporary Stage 1 model on the 80%, predict on the 20%, and then train the RL agent *only* on that unseen 20%. This ensures the RL agent learns how to trade on realistic, imperfect DL predictions.

---

## Summary of Work Done

1.  **Environment Setup:** Virtual environment created, GPU-enabled PyTorch, Stable-Baselines3, Gymnasium, and Optuna installed.
2.  **Configuration:** `config.yaml` established as the central source of truth for all hyperparameters and paths.
3.  **Data Extraction:** Scripts built and executed to pull 25 years of daily data for 5 major NIFTY 50 stocks + India VIX.
4.  **Feature Pipeline:** `features/` module built. Code successfully parses OHLCV, computes 11 technical indicators, handles the VIX gap, computes the binary target, drops warm-up NaNs, and saves clean files.
5.  **Validation Check:** `Data/validate_data.py` built to ensure 0 NaNs, 0 Infs, continuous dates, and correct schema. All 5 stocks passed.
6.  **Stage 1 DL Predictor:** PyTorch dataset, GRU+Attention model, Walk-Forward training loop, and MC Dropout inference pipeline fully implemented. All 5 stocks fully trained and evaluated after Optuna tuning.
7.  **Stage 2 RL Agent:** MaskablePPO agent, custom Gymnasium environment (`TradingEnv`), asymmetric Sortino reward function, and OOF (Out-Of-Fold) training pipeline built and successfully tested.
8.  **Stage 3 Evaluation Pipeline:** Walk-forward orchestrator (`evaluation/walk_forward.py`), 3-way backtester, risk metrics calculator, and full visualisation suite successfully built. End-to-end integration tested.

We are now in Phase 5: Polish & Streamlit Dashboard.

---

## 7. Evaluation, Backtesting & Visualization (Phase 4)

This phase answers the central question: *"Does this system actually make money?"*

### 7.1 Walk-Forward Orchestrator (`evaluation/walk_forward.py`)
The orchestrator ties the entire pipeline together end-to-end. For each stock × each walk-forward window, it:
1. Loads the processed features CSV and splits by window boundaries
2. Normalizes the test data using the **saved training-window scaler** (never refit on test data)
3. Loads the Stage 1 model and runs MC Dropout inference (T=50) to produce `p_up` and `confidence`
4. Assembles a test DataFrame and passes it to the backtester for strategy comparison
5. If no RL model exists for a window, it trains one on-the-fly using `rl.train_agent`

### 7.2 Three-Way Backtesting (`evaluation/backtest.py`)
For each test window, we run three strategies on the same data:
*   **Buy-and-Hold (Baseline):** Buy on day 1, hold until end. No transaction costs.
*   **Predictor-Only:** Naively follow Stage 1's predictions — BUY when `p_up > 0.5`, SELL otherwise. Applies transaction costs on every trade.
*   **Full RL System:** The trained MaskablePPO agent makes all decisions. Transaction costs are embedded in the environment.

Each strategy outputs an equity curve, daily returns, a trade log, and a metrics dictionary.

### 7.3 Risk Metrics (`evaluation/metrics.py`)
Pure metric functions with no side effects:
*   **Sortino Ratio** — Primary metric. Penalizes downside risk only.
*   **Sharpe Ratio** — Risk-adjusted return using total volatility.
*   **Maximum Drawdown** — Worst peak-to-trough decline.
*   **Calmar Ratio** — Annualized return divided by max drawdown.
*   **Win Rate** — Percentage of profitable trades.
*   **Profit Factor** — Gross profits divided by gross losses.

All annualized metrics use 250 trading days/year (Indian market convention).

### 7.4 Visualization (`evaluation/visualise.py`)
All charts saved as PNGs to `results/STOCK/plots/`:
*   **Equity Curves** — 3 strategies overlaid per window
*   **Drawdown Chart** — Underwater plot for the RL strategy
*   **Metric Comparison Bars** — Side-by-side Sortino/Sharpe/MaxDD for all 3 strategies
*   **Trade Scatter** — BUY/SELL markers overlaid on the stock's close price
*   **Aggregate Summary Table** — Average metrics across all windows per stock

---

## 8. Future Implementation: Polish & Streamlit Dashboard (Phase 5)

This is the final phase to make the project presentable and interactive.

### 8.1 Streamlit Dashboard (`dashboard/app.py`)
An interactive dashboard to explore the results:
*   **Backtest Viewer** — Dropdowns for Stock and Window to view equity curves and trade scatter plots interactively.
*   **Metrics Scorecard** — Display the JSON metric results in clean UI tables.
*   **Signal Explorer** — View Stage 1 DL predictions (p_up, confidence) vs actual price movements.

### 8.2 Polish
*   Comprehensive README.md with architecture diagrams and setup instructions.
*   Code cleanup and docstring verification.
*   Final end-to-end execution across all 5 stocks for all windows to generate the ultimate report.

---

## 6. How to Run the System (Commands)

(IMP: This section always needs to be updated whenever the code is updated with new features, model, phases, etc.)

To execute different parts of the NIFTY 50 Deep RL Trading System, use the following commands from the root directory (`d:\2_Antigravity\RNN`):

### 6.1 Setup & Environment
Activate the Python virtual environment (PowerShell):
```powershell
.\env\Scripts\Activate.ps1
```

### 6.2 Data Processing & Validation
Generate features from raw data:
```powershell
python -m features.pipeline
```
Validate the integrity of the processed data:
```powershell
python -m data.validate_data
```

### 6.3 Stage 1: Deep Learning Predictor
Train the GRU+Attention model for a specific stock (e.g., RELIANCE):
```powershell
python -m models.train_predictor --stock RELIANCE
```
Run hyperparameter optimization via Optuna:
```powershell
python -m tuning.optuna_search
```
Evaluate the trained Stage 1 models (calculates accuracy, MCC, etc.):
```powershell
python -m models.evaluate --stock RELIANCE
```

### 6.4 Stage 2: Reinforcement Learning Agent
Train the RL Agent using Out-Of-Fold predictions for a specific stock and window (e.g., RELIANCE, Window 0):
```powershell
python -m rl.train_agent --stock RELIANCE --window 0
```

### 6.5 Evaluation & Backtesting (Phase 4 — Upcoming)
Run the full walk-forward evaluation for a stock:
```powershell
python -m evaluation.walk_forward --stock RELIANCE
```
Run evaluation for a specific window:
```powershell
python -m evaluation.walk_forward --stock RELIANCE --window 0
```
Run evaluation for all stocks:
```powershell
python -m evaluation.walk_forward
```
