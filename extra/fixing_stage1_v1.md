# Stage 1 Diagnostic Analysis & Improvement Plan

## The Diagnosis: What the Metrics Tell Us

### Cross-Stock Summary

| Stock | Mean Accuracy | Mean AUC | Best Epoch = 0 | Avg Prediction Std |
|---|---|---|---|---|
| RELIANCE | 49.7% | 0.507 | 38% of windows | 0.027 |
| TCS | 49.6% | 0.523 | 50% of windows | 0.049 |
| HDFCBANK | 50.2% | 0.501 | 0% of windows | 0.029 |
| HINDUNILVR | 50.5% | 0.515 | 29% of windows | 0.023 |
| SUNPHARMA | 51.3% | 0.514 | 43% of windows | 0.034 |

> [!CAUTION]
> **All 5 stocks are performing at near-random (50%) accuracy.** A coin flip would achieve 50%. The models are essentially not learning any useful patterns.

---

### Root Cause Analysis

I've identified **5 specific problems** from the metrics:

#### 🔴 Problem 1: Predictions are Collapsed Around 0.50

The prediction histogram tells the real story. For most windows, **75-98% of predictions fall between 0.45 and 0.55**. The model is barely moving its output away from the center.

```
RELIANCE Window 1: 92.8% of predictions in [0.45, 0.55]
HDFCBANK Window 6: 90.0% of predictions in [0.45, 0.55]  
HINDUNILVR Window 3: 98.2% of predictions in [0.45, 0.55]
```

**What this means:** The model has learned to "play it safe" by predicting ~0.50 for everything. The sigmoid output is barely activated — the logits are near zero.

**Why it happens:** The learning rate (0.001) is too high for this architecture. The model takes wild gradient steps in the first few epochs, the validation loss jumps up, and early stopping kills training before the model can find useful patterns.

---

#### 🔴 Problem 2: Training Dies Immediately (Best Epoch = 0)

Across all stocks, **30-50% of windows have best_epoch = 0**, meaning the very first untrained weights were the "best" the model ever achieved. The model immediately starts overfitting from epoch 1.

```
RELIANCE: 3/8 windows → best_epoch = 0
TCS:      3/6 windows → best_epoch = 0
SUNPHARMA: 3/7 windows → best_epoch = 0
```

**What this means:** The first gradient update is already too aggressive — it overshoots any useful signal and lands in a worse spot. Every subsequent epoch makes it worse.

---

#### 🔴 Problem 3: Confidence is Meaningless

Every single sample across all stocks has confidence > 0.9. The confidence scores are not calibrated at all.

```
All stocks: 100% of predictions have confidence > 0.9
But accuracy at confidence > 0.9: 45-55% (random)
```

**What this means:** The MC Dropout variance is tiny (std ~ 0.02-0.05), which makes `confidence = 1 - std` always > 0.90. This happens because the model outputs are so tightly clustered near 0.50 that even with dropout perturbation, there's almost no variance.

**Fix:** Once predictions are properly spread across [0, 1], MC Dropout will naturally produce more meaningful confidence ranges. This is a symptom, not a root cause.

---

#### 🔴 Problem 4: No Learning Rate Decay

The current setup uses a flat learning rate of 0.001 for the entire training run. There is no scheduler (like ReduceLROnPlateau) to reduce the learning rate when the model stops improving. This means:

1. Early epochs: LR too high → overshoots
2. Mid epochs: LR still too high → oscillates around minima
3. Early stopping triggers before the model can converge

---

#### 🔴 Problem 5: Weight Initialization

PyTorch's default initialization (Kaiming uniform) may not be optimal for GRU + Attention architectures. Poor initialization combined with a high learning rate creates the "instant overfit" pattern we see.

---

## Proposed Fixes (Priority Order)

All fixes are modifications to existing files. No new files needed.

### Fix 1: Add ReduceLROnPlateau Scheduler

**File:** `models/train_predictor.py`

Add a learning rate scheduler that automatically reduces the LR by half when validation loss plateaus. This is the single most important fix.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)
# Called at end of each epoch:
scheduler.step(avg_val_loss)
```

**Why this helps:** Instead of the model taking wild jumps and immediately overshooting, the LR will drop to 0.0005 → 0.00025 → 0.000125 as training progresses, allowing the model to settle into finer patterns.

---

### Fix 2: Lower Initial Learning Rate

**File:** `config.yaml`

Change `learning_rate: 0.001` → `learning_rate: 0.0001`

**Why:** 0.001 is standard for NLP/vision tasks with millions of parameters. Our model has only **230K parameters** and financial data is extremely noisy. A 10x smaller LR gives the model room to learn gradually instead of overshooting.

---

### Fix 3: Xavier Weight Initialization

**File:** `models/gru_attention.py`

Add explicit Xavier/Glorot initialization for the GRU and Linear layers. This ensures activations and gradients maintain a healthy scale from the very first forward pass.

```python
def _init_weights(self):
    for name, param in self.named_parameters():
        if 'weight_ih' in name or 'weight_hh' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.xavier_uniform_(self.fc2.weight)
```

---

### Fix 4: Gradient Clipping

**File:** `models/train_predictor.py`

Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before the optimizer step. This prevents catastrophic gradient explosions that can destroy the model's weights in a single batch.

---

### Fix 5: Learning Rate Warmup

**File:** `models/train_predictor.py`

For the first 3 epochs, linearly ramp the LR from ~0 to the target LR (0.0001). This lets the model "feel out" the loss landscape before taking full-sized steps.

Implementation: Use a combined scheduler — LinearLR warmup for 3 epochs, then ReduceLROnPlateau.

---

## Expected Impact

| Metric | Current | Expected After Fixes |
|---|---|---|
| Mean Accuracy | ~50% (random) | 52-55% (meaningful signal) |
| Best Epoch = 0 | 30-50% of windows | < 10% |
| Prediction Std | 0.02-0.05 | 0.10-0.20 |
| Confidence calibration | Useless (all > 0.9) | Graduated bins with accuracy correlation |

> [!NOTE]
> 52-55% accuracy on daily stock direction may sound low, but in quant finance even **51-52% with proper confidence calibration** is profitable when combined with good risk management (which is exactly what our Stage 2 RL agent does). The key is not raw accuracy but having *calibrated confidence* — knowing WHEN the model is right.

---

## Files Modified

| File | Changes |
|---|---|
| `config.yaml` | `learning_rate: 0.001` → `0.0001` |
| `models/gru_attention.py` | Add `_init_weights()` method with Xavier initialization |
| `models/train_predictor.py` | Add ReduceLROnPlateau scheduler, gradient clipping, LR warmup |

## Verification Plan

After applying fixes:
1. Retrain all 5 stocks
2. Re-run `python -m models.evaluate`
3. Compare `metrics.json` against current baseline
4. Key success criteria:
   - Prediction std > 0.10 (spread away from 0.50)
   - Best epoch > 0 for at least 80% of windows
   - Accuracy > 51% averaged across all stocks
