# NIFTY 50 Deep RL Trading System: Theory & System Design

This document serves as a comprehensive summary of the project's conceptual foundation, the architectural choices, and the specific implementations completed so far. It explains *what* we are building and, more importantly, *why* we are building it this way.

---

## 1. System Architecture Overview

The system is a two-stage hybrid Deep Learning and Reinforcement Learning pipeline designed for daily equity trading. 

Instead of training a single monolithic RL agent to look at raw prices and make trades (which often fails due to noise and non-stationarity), we separate the problem into two distinct stages:

1. **Stage 1 (The Analyst):** A Deep Learning model (GRU + Attention) that looks at historical data and predicts the *probability of the stock going UP tomorrow*. It also outputs a *confidence score*.
2. **Stage 2 (The Trader):** A Reinforcement Learning agent (MaskablePPO) that takes the Analyst's predictions, factors in transaction costs, portfolio state, and market volatility, and decides whether to BUY, SELL, or HOLD.

---

## 2. Completed Implementation: Data & Feature Engineering

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

## 4. Future Implementation: Stage 1 (Deep Learning Predictor)

### 4.1 GRU (Gated Recurrent Unit)
We use a GRU over an LSTM because it has fewer parameters, making it faster to train and less prone to overfitting on financial data, while still successfully capturing time-series dependencies.

### 4.2 Multi-Head Attention
A pure GRU forces all past information into a single fixed-size hidden state (the "bottleneck" problem). By adding Multi-Head Attention, the model can look back over the entire 60-day sequence and "pay attention" directly to critical past events (e.g., a massive volume spike 45 days ago) when making today's prediction.

### 4.3 Monte Carlo (MC) Dropout for Uncertainty
Standard neural networks are notoriously overconfident. They might output $P(UP) = 0.9$ even if they have never seen a pattern before.
*   **Theory:** By keeping Dropout *turned on* during inference and running the input through the network 50 times, we get 50 slightly different predictions.
*   If the predictions are tightly clustered, the model has high **Confidence**.
*   If the predictions are scattered, the model is uncertain. The RL agent can use this confidence score to avoid trading during highly uncertain regimes.

---

## 5. Future Implementation: Stage 2 (RL Agent)

### 5.1 Maskable PPO
We use Reinforcement Learning to optimize the actual execution of trades. We use **Maskable PPO** (Proximal Policy Optimization).
*   **Action Masking:** An agent shouldn't try to BUY if it already holds the stock, or SELL if it has nothing. Masking mathematically zeroes out the probabilities of these illegal actions, forcing the agent to only explore valid trades, drastically speeding up learning.

### 5.2 The Reward Function (Sortino Optimized)
Traditional RL bots optimize for maximum profit, which often leads to catastrophic drawdowns.
We shape the reward to optimize for risk-adjusted returns:
*   We penalize negative returns (losses) more heavily than we reward positive returns.
*   We deduct **transaction costs (0.1%)**, **STT/SEBI charges**, and **slippage (0.05%)** on *every single trade step* within the environment. If the RL agent trades too much, fees will destroy its P&L. It learns to hold for the best opportunities.

---

## Summary of Work Done

1.  **Environment Setup:** Virtual environment created, GPU-enabled PyTorch, Stable-Baselines3, Gymnasium, and Optuna installed.
2.  **Configuration:** `config.yaml` established as the central source of truth for all hyperparameters and paths.
3.  **Data Extraction:** Scripts built and executed to pull 25 years of daily data for 5 major NIFTY 50 stocks + India VIX.
4.  **Feature Pipeline:** `features/` module built. Code successfully parses OHLCV, computes 11 technical indicators, handles the VIX gap, computes the binary target, drops warm-up NaNs, and saves clean files.
5.  **Validation Check:** `Data/validate_data.py` built to ensure 0 NaNs, 0 Infs, continuous dates, and correct schema. All 5 stocks passed.

We are now perfectly positioned to begin coding the PyTorch neural networks in Stage 1.
