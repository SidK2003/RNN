# NIFTY 50 Deep RL Trading System
### Product Concept Document
**Author:** Siddharth Khodke | **Version:** 1.0

---

## The One-Line Pitch
A two-stage AI system where an LSTM predicts where a stock is going, and a Reinforcement Learning agent decides whether that prediction is worth acting on — trained and evaluated entirely on NIFTY 50 stocks.

---

## The Problem

Basic stock prediction projects answer the wrong question. Knowing that a stock *might* go up tomorrow doesn't tell you whether to buy, how confident to be, or when to exit. Most ML trading projects stop at prediction and never answer: *"So what do you actually do with that?"*

This project answers it.

---

## The Solution

Two models working in sequence:

**Stage 1 — LSTM Prediction Engine**
Takes historical price data and technical indicators as input. Outputs two things: a predicted price direction for the next day, and a confidence score for that prediction. The confidence score is the key upgrade — it tells the agent how much to trust what it's seeing.

**Stage 2 — PPO Reinforcement Learning Agent**
Takes the LSTM's prediction, the confidence score, current portfolio state, and market context as its inputs. Learns through thousands of simulated trading sessions to decide: BUY, HOLD, or SELL. The agent learns not just what the LSTM says — but *when* the LSTM is worth listening to.

---

## Why Two Stages

A single model that predicts price and also decides trades conflates two different problems. Separating them makes each model better at its specific job, and makes the system explainable — you can pinpoint whether a bad trade came from a wrong prediction or a poor decision on a correct prediction. That's not something a single end-to-end model gives you.

---

## Scope

- **Universe:** 5 NIFTY 50 stocks across different sectors (IT, Banking, FMCG, Pharma, Energy)
- **Data:** 6 years of daily OHLCV data via yfinance — 5 years training, 1 year held-out testing
- **Mode:** Backtesting only for MVP. No live trading, no real money.
- **Benchmark:** Strategy is evaluated against NIFTY 50 buy-and-hold

---

## Key Features

**Confidence-Aware Prediction**
The LSTM doesn't just output a direction — it outputs how sure it is. This is done via Monte Carlo Dropout: the model runs inference many times with randomness enabled and the spread of outputs becomes the confidence measure. High spread = low confidence. This is a technique used in production ML systems, not just academia.

**Custom Trading Environment**
The RL agent trains inside a custom-built simulation of the stock market modelled as an OpenAI Gym environment. Every parameter of that simulation is a design decision: what information the agent sees, what actions it can take, and critically — how it is rewarded. The reward function penalises volatility and over-trading, not just raw profit. This teaches the agent to make *good* decisions, not just lucky ones.

**Three-Way Evaluation**
The final deliverable isn't a model — it's a comparison. The held-out test year is replayed under three strategies: buy-and-hold, LSTM-only, and LSTM + RL Agent. Each is evaluated on cumulative return, Sharpe ratio, maximum drawdown, and win rate. Whether the RL agent wins or loses is less important than the rigour of the comparison.

**Attention Visualisation**
The LSTM uses an attention mechanism that shows, for any given prediction, which past days it weighted most heavily. This is surfaced in the dashboard — you can see *why* the model predicted what it predicted, not just what it predicted. This is the explainability angle that matters in fintech.

**Streamlit Dashboard**
Three tabs: live signal inference for today's date, the backtest comparison charts, and the attention weight visualisation. Makes it a product you can show, not a notebook you have to explain.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | yfinance, pandas |
| Prediction Model | PyTorch (LSTM + Attention + MC Dropout) |
| RL Agent | Stable-Baselines3 (PPO algorithm) |
| RL Environment | OpenAI Gymnasium (custom environment) |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |

---

## What This Demonstrates

- You can frame a problem correctly, not just apply a model to it
- You understand the difference between prediction and decision-making
- You know how to build a custom RL environment — not just use a pre-built one
- You can measure your own work honestly against a benchmark
- You understand uncertainty quantification (MC Dropout) — a production ML concept most students never touch

---

## What This Is Not

This is not a get-rich-quick trading bot. It is a research and engineering project that demonstrates how modern AI techniques — sequence modelling, uncertainty estimation, and reinforcement learning — can be combined and evaluated on a real-world domain. The honest reporting of results, including where the system underperforms, is part of what makes it credible.

---

## Resume Narrative

This project completes a three-project story on the resume:
- **Crux** → LLM-powered full-stack product
- **RAG System** → NLP retrieval pipeline
- **This project** → Deep learning + Reinforcement Learning on financial data

Each project uses a different paradigm of AI. Together they show range — something most candidates with three similar projects cannot claim.

---

*Concept version: May 2025*
