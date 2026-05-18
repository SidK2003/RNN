# Stage 1 Data Quality Upgrade — Implementation Plan

Improve Stage 1 predictions by upgrading data quality (target, features, scaling) instead of changing model architecture. Applies all 5 guardrails from the analysis.

## User Review Required

> [!IMPORTANT]
> This plan touches the feature pipeline, which means **all processed CSVs must be regenerated** and **all existing model checkpoints become invalid**. Old results should be archived or deleted before retraining.

> [!IMPORTANT]
> The `forward_return` column computed during pipeline is **never saved to disk** — it is used only as an intermediate to compute the target, then dropped. This is the primary leakage safeguard (Guardrail 1).

## Proposed Changes

### Config

#### [MODIFY] [config.yaml](file:///d:/2_Antigravity/RNN/config.yaml)
- Add a new `target:` section:
  ```yaml
  target:
    horizon: 5            # predict 5-day forward return
    neutral_zone: 0.005   # ±0.5% — label as neutral / drop from training
  ```
- Add NIFTY50 file reference under `nifty50:` (already has the key, just confirm path is correct).

---

### Feature Pipeline

#### [NEW] [nifty.py](file:///d:/2_Antigravity/RNN/features/nifty.py)
- `load_nifty(csv_path) -> pd.DataFrame` — loads NIFTY50 daily CSV, cleans it (same pattern as `vix.py`).
- `merge_nifty(stock_df, nifty_df) -> pd.DataFrame` — left-join on date, adds:
  - `nifty_log_return` — `np.log(close / close.shift(1))` of NIFTY (log return, not percentage return — Guardrail 4).
  - `nifty_log_return_5d` — `np.log(close / close.shift(5))`.
  - `nifty_volatility_20d` — 20-day rolling std of `nifty_log_return`.
- For dates before NIFTY data starts: fill with `0.0` (same as VIX pattern).

#### [MODIFY] [pipeline.py](file:///d:/2_Antigravity/RNN/features/pipeline.py)
Changes to `run_pipeline()`:
1. **Load NIFTY** — call `load_nifty()` once, then `merge_nifty()` per stock (same as VIX).
2. **Add relative features** after merge:
   - `relative_strength` = `log_return - nifty_log_return` (both log returns — Guardrail 4).
   - `relative_volatility` = `stock_volatility_20d / nifty_volatility_20d` (where `stock_volatility_20d` = 20-day rolling std of `log_return`).
   - `vix_change` = daily percentage change of `india_vix` (where `vix_available == 1`; else `0.0`).
3. **Replace `add_target()`** — new logic:
   - Compute `forward_return = (close.shift(-horizon) - close) / close` using `config["target"]["horizon"]`.
   - Label: `1.0` if `> +neutral_zone`, `0.0` if `< -neutral_zone`, `-1.0` if between.
   - **Drop `forward_return` column from the DataFrame before saving** (Guardrail 1). Only `target` survives.
   - Last `horizon` rows get `NaN` target (no future data) → dropped by NaN cleaner.
4. **Add `stock_volatility_20d`** to `technical_indicators.py` or inline in pipeline — 20-day rolling std of `log_return`.

#### [MODIFY] [technical_indicators.py](file:///d:/2_Antigravity/RNN/features/technical_indicators.py)
- Add `stock_volatility_20d(log_return_series, window=20)` function — simple `rolling(20).std()`.

---

### Dataset

#### [MODIFY] [dataset.py](file:///d:/2_Antigravity/RNN/models/dataset.py)
- Add neutral-label filtering to `StockSequenceDataset.__init__()`:
  - After computing `self.num_sequences`, build `self.valid_indices` — a list of indices where `targets[idx + seq_length - 1] != -1.0`.
  - `__len__()` returns `len(self.valid_indices)`.
  - `__getitem__(idx)` maps through `self.valid_indices[idx]` to the original position.
- Sequences remain contiguous (chronological integrity preserved). Only the index sampling skips neutrals.

---

### Training & Evaluation

#### [MODIFY] [train_predictor.py](file:///d:/2_Antigravity/RNN/models/train_predictor.py)
1. **Dynamic feature selection** (replaces hardcoded `FEATURE_COLS`):
   - `EXCLUDE_COLS = {"date", "timestamp", "target"}` — forward_return never exists in the CSV so no need to exclude it.
   - `BINARY_FLAGS = {"vix_available"}` — excluded from scaling (Guardrail 5).
   - Features inferred: `feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]`.
   - `scale_cols = [c for c in feature_cols if c not in BINARY_FLAGS]`.
   - Save `feature_cols` list inside every `.pt` checkpoint for downstream consistency.
2. **RobustScaler** (replaces z-score normalization):
   - `from sklearn.preprocessing import RobustScaler`.
   - `scaler.fit(train_df[scale_cols])` — fit on train only.
   - `train_df[scale_cols] = scaler.transform(train_df[scale_cols])`, same for test.
   - Binary flags (`vix_available`) are **not scaled** (Guardrail 5).
   - Save scaler (pickle or median/IQR arrays) alongside checkpoint for inference consistency.
3. **Boundary handling** (Guardrail 2):
   - After `split_by_window()`, set `train_df.iloc[-horizon:, target_col_idx] = -1.0` and `test_df.iloc[-horizon:, target_col_idx] = -1.0` where `horizon = config["target"]["horizon"]`.
   - The `StockSequenceDataset` neutral-filter then naturally excludes these boundary rows.

#### [MODIFY] [evaluate.py](file:///d:/2_Antigravity/RNN/models/evaluate.py)
1. **Import changes**: Stop importing `FEATURE_COLS` / `NORMALIZE_COLS` from train_predictor. Instead, load `feature_cols` from the checkpoint being evaluated.
2. **Same boundary handling** as train_predictor — set last `horizon` rows of test to `-1.0`.
3. **Add metrics** (Guardrail — evaluation baseline):
   - `majority_class_baseline` — accuracy of always predicting the most common class.
   - `balanced_accuracy` — `sklearn.metrics.balanced_accuracy_score`.
   - `mcc` — `sklearn.metrics.matthews_corrcoef`.
   - `accuracy_lift` — `model_accuracy - majority_baseline`.
4. Add these to both per-window results and aggregates.

---

### Validation

#### [MODIFY] [validate_data.py](file:///d:/2_Antigravity/RNN/Data/validate_data.py)
1. **Schema check** — replace hardcoded `expected_cols` with a dynamic approach: require a minimum set (`date`, `target`, `close`, etc.) and accept any additional columns (since features are now dynamic).
2. **Target check** — change from `[0.0, 1.0]` to `[-1.0, 0.0, 1.0]` to allow the neutral label.
3. **Add neutral % report** — print what percentage of rows are neutral after the upgrade.

---

## Verification Plan

### Automated Tests
1. **Pipeline re-run**: `python -m features.pipeline` — must complete with 0 NaN/Inf.
2. **Validation**: `python Data/validate_data.py` — must pass with updated checks.
3. **Sanity check**: Verify `forward_return` does NOT appear in any processed CSV column headers.
4. **Train smoke test**: `python -m models.train_predictor --stock RELIANCE --window 0` — must train without errors.
5. **Evaluate smoke test**: `python -m models.evaluate --stock RELIANCE` — must produce `metrics.json` with the new metrics (MCC, balanced accuracy, majority baseline).

### Manual Verification
- Inspect a processed CSV to confirm: no `forward_return` column, `target` values are `{-1.0, 0.0, 1.0}`, new NIFTY/relative columns are present.
- Check that the neutral % is roughly 20-35% (as expected for ±0.5% on a 5-day horizon).
- Compare new evaluation metrics against v2 results to verify improvement (or at least no regression in signal quality).
