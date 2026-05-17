# Stage 1 Diagnostic Analysis (Iteration 2) & Refinement Plan

## The Diagnosis: Optimization Fixed, Generalization Failed

I ran a side-by-side comparison of the baseline metrics vs. the new metrics after applying our fixes (LR Scheduler, Xavier Init, Gradient Clipping, lower LR).

Here is the high-level summary of changes:

| Metric | Baseline | Current | What this means |
|---|---|---|---|
| **Best Epoch** | ~2 epochs | ~12 epochs | **SUCCESS:** The model is no longer instantly blowing up its gradients on epoch 1. It is actually learning on the training set. |
| **Prediction Spread (Std)** | 0.02 | 0.06 | **SUCCESS:** The model is no longer completely collapsing its predictions to 0.50. It is making distinct choices. |
| **Test Accuracy** | ~50.5% | ~50.0% | **FAILURE:** The model still fails to generalize. It is guessing randomly on unseen data. |
| **ROC-AUC** | ~0.510 | ~0.510 | **FAILURE:** The model has no rank-ordering power on unseen data. |

### Root Cause Analysis: Over-parameterization
Because the *optimization* is now working properly, the failure to generalize points to one specific issue: **Massive Overfitting through Over-parameterization**.

Currently, the model has:
- 128 hidden units
- 2 layers
- 4 attention heads
- 60 days of sequence lookback
- **Total Parameters:** ~230,000

We are trying to train a 230,000 parameter network on a training dataset of roughly **2,000 rows** per walk-forward window. Financial data has extremely low signal-to-noise ratio. A model this large will easily memorize the noise in those 2,000 rows (scoring perfectly on the training set) and fail completely on the test set.

## Proposed Fixes (Priority Order)

To fix this, we must drastically reduce the capacity of the model and turn up regularization. We need to force the model to be "dumb" enough that it can only learn broad, generalizable patterns, rather than memorizing the exact sequence of prices.

### Fix 1: Drastic Architecture Reduction
**File:** `config.yaml`
We will reduce the model's parameter count by ~90% so it cannot memorize the data.
- `hidden_size`: **128 → 32**
- `num_layers`: **2 → 1**
- `num_heads`: **4 → 2**

### Fix 2: Shorter Sequence Lookback
**File:** `config.yaml`
Stock market data is heavily auto-correlated for the first few days, but a 60-day lookback introduces a massive amount of noise for predicting the *very next day*.
- `seq_length`: **60 → 20** (Approx 1 trading month)

### Fix 3: Extreme Regularization (Dropout & Weight Decay)
**Files:** `config.yaml` and `models/train_predictor.py`
We will increase the penalties for overfitting.
- **Dropout:** Increase from `0.3` to `0.5` in `config.yaml`. This forces the network to rely on robust, redundant features.
- **Weight Decay (L2 Penalty):** Increase `weight_decay` in the AdamW optimizer inside `train_predictor.py` from `1e-5` to `1e-3`. This prevents the network from assigning large weights to random noise.

---

## Expected Impact

By shrinking the model from ~230k parameters to ~15k parameters and increasing regularization, the model will struggle much more on the *training set*, but whatever it does learn is significantly more likely to translate to the *test set*. 

If this iteration achieves > 52-53% accuracy and AUC > 0.53 on the test set, we will have a viable Alpha signal for the Stage 2 RL Agent.

## User Review Required

Does this aggressive downsizing plan sound good to you? Once you approve, I will make the edits, and you can fire off the retraining!
