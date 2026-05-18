"""
Comprehensive post-training evaluation for Stage 1 models.

In simple words: After the AI finishes training, this script is the "Report Card Generator".
It loads every saved model checkpoint, runs it on the out-of-sample test data, and computes
a huge set of metrics that tell us EXACTLY what's going right and wrong.

Metrics computed:
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Balanced Accuracy
  - Baseline: Majority-class accuracy (what you'd get by always predicting the most common class)
  - Overfitting: Train-Val loss gap, best epoch analysis
  - Prediction Distribution: Are predictions collapsed around 0.5? (bad sign)
  - Confidence Calibration: When the model says "I'm 80% confident", is it actually right 80% of the time?

Usage:
    python -m models.evaluate --stock RELIANCE
    python -m models.evaluate --stock TCS
    python -m models.evaluate                    # Evaluates all stocks

Output: Saves results/STOCKNAME/metrics.json
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from models.gru_attention import GRUAttentionModel
from models.dataset import StockSequenceDataset
from models.inference import mc_dropout_predict
from models.train_predictor import (
    load_config,
    get_feature_cols,
    get_scale_cols,
    compute_walk_forward_windows,
    split_by_window,
    normalize_features,
    set_seed,
    get_device,
)

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================
# CORE EVALUATION LOGIC
# =============================================

def evaluate_window(
    config: dict,
    stock_name: str,
    window_idx: int,
    window: dict,
    df: pd.DataFrame,
    device: torch.device,
) -> Optional[Dict]:
    """
    Evaluate a single trained model on its corresponding test window.

    This is where the magic happens. For each walk-forward window:
    1. We load the trained model checkpoint
    2. Run it on the OUT-OF-SAMPLE test data (data it has NEVER seen)
    3. Compute classification metrics (accuracy, F1, AUC, etc.)
    4. Run MC Dropout for confidence analysis
    5. Analyze the prediction distribution
    """
    model_cfg = config["model"]
    seq_length = model_cfg["seq_length"]
    horizon = config["target"]["horizon"]

    # --- Check if checkpoint exists ---
    checkpoint_path = os.path.join(
        config["results_dir"], stock_name, "models", f"window{window_idx}.pt"
    )
    if not os.path.exists(checkpoint_path):
        print(f"    SKIP: No checkpoint found at {checkpoint_path}")
        return None

    # --- Load checkpoint first to get feature_cols ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get feature_cols from checkpoint (saved during training)
    # This guarantees we use the exact same features the model was trained with.
    feature_cols = checkpoint.get("feature_cols", None)
    if feature_cols is None:
        # Fallback for old checkpoints: infer from data
        print(f"    WARNING: No feature_cols in checkpoint, inferring from data")
        feature_cols = get_feature_cols(df)
    
    scale_cols = checkpoint.get("scale_cols", get_scale_cols(feature_cols))
    num_features = len(feature_cols)

    # --- Split and normalize data (with boundary handling) ---
    train_df, test_df = split_by_window(df, window, horizon=horizon)

    if len(test_df) < seq_length + 1:
        print(f"    SKIP: Test set too small ({len(test_df)} rows)")
        return None

    # Normalize using ONLY training statistics (same as during training)
    train_norm, test_norm, _ = normalize_features(train_df, test_df, scale_cols)

    # --- Create test sequences (filter_neutrals=True to exclude boundary targets) ---
    test_features = test_norm[feature_cols].values.astype(np.float32)
    test_targets = test_norm["target"].values.astype(np.float32)
    test_dataset = StockSequenceDataset(test_features, test_targets, seq_length, filter_neutrals=True)

    if len(test_dataset) == 0:
        print(f"    SKIP: No valid test sequences after filtering neutrals")
        return None

    # We use a single large batch to process all test sequences at once (faster on GPU)
    all_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    all_y = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))]).cpu().numpy()

    # --- Load model ---
    model = GRUAttentionModel(
        num_features=num_features,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # --- 1. Standard Classification Metrics ---
    # Run in eval mode (dropout OFF) for deterministic predictions
    model.eval()
    with torch.no_grad():
        raw_probs = model(all_x).cpu().numpy()  # Sigmoid probabilities

    predictions_binary = (raw_probs > 0.5).astype(float)

    test_metrics = compute_classification_metrics(all_y, raw_probs, predictions_binary)

    # --- 2. MC Dropout Confidence Analysis ---
    # Use 30 passes — good balance between speed and statistical reliability.
    # 50 is ideal for production but 30 is sufficient for diagnostics.
    mc_passes = 30
    mc_result = mc_dropout_predict(model, all_x, num_passes=mc_passes)

    mc_probs = mc_result["p_up"].cpu().numpy()
    mc_confidence = mc_result["confidence"].cpu().numpy()
    mc_direction = mc_result["direction"].cpu().numpy()

    # MC Dropout classification metrics (these are more realistic than eval mode)
    mc_metrics = compute_classification_metrics(all_y, mc_probs, mc_direction)

    # --- 3. Prediction Distribution Analysis ---
    distribution = compute_distribution_analysis(mc_probs)

    # --- 4. Confidence Calibration ---
    confidence_analysis = compute_confidence_calibration(all_y, mc_probs, mc_confidence)

    # --- 5. Training History (from checkpoint) ---
    training_info = {
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_val_loss": float(checkpoint.get("val_loss", -1)),
        "best_train_loss": float(checkpoint.get("train_loss", -1)),
        "total_epochs": int(checkpoint.get("total_epochs", -1)),
        # Overfitting indicator: if train loss << val loss, the model memorized the training data
        "train_val_gap": float(checkpoint.get("train_loss", 0) - checkpoint.get("val_loss", 0)),
    }

    # Include full loss curves if available (for plotting later)
    if "train_losses" in checkpoint:
        training_info["train_loss_curve"] = [float(x) for x in checkpoint["train_losses"]]
        training_info["val_loss_curve"] = [float(x) for x in checkpoint["val_losses"]]

    return {
        "window_idx": window_idx,
        "train_period": f"{window['train_start'].year}-{window['train_end'].year}",
        "test_period": f"{window['test_start'].year}-{window['test_end'].year}",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_sequences": len(test_dataset),
        "num_features": num_features,
        "training": training_info,
        "eval_mode_metrics": test_metrics,
        "mc_dropout_metrics": mc_metrics,
        "prediction_distribution": distribution,
        "confidence_analysis": confidence_analysis,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute all standard classification metrics.
    y_true: actual labels (0 or 1)
    y_probs: predicted probabilities (0.0 to 1.0)
    y_pred: predicted labels (0 or 1)
    """
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ROC-AUC needs at least 2 classes in y_true
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = 0.5  # Fallback if only one class present

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "majority_class_baseline": float(max(y_true.mean(), 1 - y_true.mean())),
        "accuracy_lift": float(accuracy_score(y_true, y_pred) - max(y_true.mean(), 1 - y_true.mean())),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
        "class_balance": {
            "actual_up_pct": float(y_true.mean()),
            "predicted_up_pct": float(y_pred.mean()),
        },
    }


def compute_distribution_analysis(probs: np.ndarray) -> Dict:
    """
    Analyze the distribution of predicted probabilities.

    If all predictions are clustered near 0.50, the model hasn't learned
    to distinguish UP from DOWN — it's essentially random.
    A healthy model should have predictions spread across the [0, 1] range.
    """
    return {
        "mean": float(probs.mean()),
        "std": float(probs.std()),
        "min": float(probs.min()),
        "max": float(probs.max()),
        "median": float(np.median(probs)),
        # What % of predictions fall in the "no-man's land" near 0.50?
        "pct_in_uncertain_zone_0.45_0.55": float(
            np.mean((probs >= 0.45) & (probs <= 0.55))
        ),
        "pct_in_uncertain_zone_0.40_0.60": float(
            np.mean((probs >= 0.40) & (probs <= 0.60))
        ),
        # Histogram: how many predictions fall in each 10% bucket
        "histogram": {
            "0.0-0.1": int(np.sum(probs < 0.1)),
            "0.1-0.2": int(np.sum((probs >= 0.1) & (probs < 0.2))),
            "0.2-0.3": int(np.sum((probs >= 0.2) & (probs < 0.3))),
            "0.3-0.4": int(np.sum((probs >= 0.3) & (probs < 0.4))),
            "0.4-0.5": int(np.sum((probs >= 0.4) & (probs < 0.5))),
            "0.5-0.6": int(np.sum((probs >= 0.5) & (probs < 0.6))),
            "0.6-0.7": int(np.sum((probs >= 0.6) & (probs < 0.7))),
            "0.7-0.8": int(np.sum((probs >= 0.7) & (probs < 0.8))),
            "0.8-0.9": int(np.sum((probs >= 0.8) & (probs < 0.9))),
            "0.9-1.0": int(np.sum(probs >= 0.9)),
        },
    }


def compute_confidence_calibration(
    y_true: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
) -> Dict:
    """
    The KEY diagnostic: Does higher confidence actually mean higher accuracy?

    We bin predictions by confidence level and check if accuracy rises
    with confidence. If it doesn't, the model's uncertainty estimates
    are meaningless and the RL agent can't trust them.
    """
    # Direction predictions from probabilities
    directions = (probs > 0.5).astype(float)
    correct = (directions == y_true).astype(float)

    bins = []
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for i, threshold in enumerate(thresholds):
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else 1.01

        # Samples whose confidence falls in this bin
        mask = (confidence >= threshold) & (confidence < upper)
        count = int(mask.sum())

        if count > 0:
            bin_accuracy = float(correct[mask].mean())
            bin_avg_confidence = float(confidence[mask].mean())
        else:
            bin_accuracy = None
            bin_avg_confidence = None

        bins.append({
            "range": f"{threshold:.1f}-{upper:.2f}",
            "count": count,
            "coverage_pct": float(count / len(y_true)),
            "accuracy": bin_accuracy,
            "avg_confidence": bin_avg_confidence,
        })

    # Also compute cumulative thresholds: "accuracy when confidence >= X"
    cumulative = []
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= threshold
        count = int(mask.sum())
        if count > 0:
            cum_accuracy = float(correct[mask].mean())
        else:
            cum_accuracy = None

        cumulative.append({
            "min_confidence": threshold,
            "count": count,
            "coverage_pct": float(count / len(y_true)),
            "accuracy": cum_accuracy,
        })

    return {
        "bins": bins,
        "cumulative_thresholds": cumulative,
        "overall_accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
    }


# =============================================
# AGGREGATION
# =============================================

def compute_aggregates(window_results: List[Dict]) -> Dict:
    """
    Compute summary statistics across all walk-forward windows.
    This gives us the big picture: how does the model perform ON AVERAGE?
    """
    if not window_results:
        return {}

    # Extract per-window values for the MC Dropout metrics (more realistic than eval mode)
    accuracies = [w["mc_dropout_metrics"]["accuracy"] for w in window_results]
    balanced_accs = [w["mc_dropout_metrics"]["balanced_accuracy"] for w in window_results]
    mccs = [w["mc_dropout_metrics"]["mcc"] for w in window_results]
    aucs = [w["mc_dropout_metrics"]["roc_auc"] for w in window_results]
    f1s = [w["mc_dropout_metrics"]["f1"] for w in window_results]
    baselines = [w["mc_dropout_metrics"]["majority_class_baseline"] for w in window_results]
    lifts = [w["mc_dropout_metrics"]["accuracy_lift"] for w in window_results]
    gaps = [w["training"]["train_val_gap"] for w in window_results]
    best_epochs = [w["training"]["best_epoch"] for w in window_results]
    total_epochs = [w["training"]["total_epochs"] for w in window_results]

    # Find best and worst windows
    best_idx = int(np.argmax(accuracies))
    worst_idx = int(np.argmin(accuracies))

    return {
        "num_windows": len(window_results),
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
        },
        "balanced_accuracy": {
            "mean": float(np.mean(balanced_accs)),
            "std": float(np.std(balanced_accs)),
        },
        "mcc": {
            "mean": float(np.mean(mccs)),
            "std": float(np.std(mccs)),
        },
        "roc_auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "min": float(np.min(aucs)),
            "max": float(np.max(aucs)),
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
        },
        "majority_baseline": {
            "mean": float(np.mean(baselines)),
        },
        "accuracy_lift": {
            "mean": float(np.mean(lifts)),
            "std": float(np.std(lifts)),
        },
        "overfitting": {
            "mean_train_val_gap": float(np.mean(gaps)),
            "mean_best_epoch": float(np.mean(best_epochs)),
            "mean_total_epochs": float(np.mean(total_epochs)),
            # If best_epoch is always 0, the model never learned anything useful
            "pct_windows_best_epoch_0": float(np.mean([e == 0 for e in best_epochs])),
        },
        "best_window": {
            "idx": int(window_results[best_idx]["window_idx"]),
            "period": window_results[best_idx]["test_period"],
            "accuracy": float(accuracies[best_idx]),
        },
        "worst_window": {
            "idx": int(window_results[worst_idx]["window_idx"]),
            "period": window_results[worst_idx]["test_period"],
            "accuracy": float(accuracies[worst_idx]),
        },
    }


# =============================================
# MAIN ORCHESTRATOR
# =============================================

def evaluate_stock(stock_name: str, config: dict) -> Dict:
    """
    Evaluate all walk-forward windows for a stock and save a comprehensive metrics.json.
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {stock_name}")
    print(f"{'=' * 70}")

    set_seed(config["seed"])
    device = get_device()

    # Load processed data
    processed_path = os.path.join("Data", "processed", f"{stock_name.lower()}_features.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} rows")

    # Compute windows
    windows = compute_walk_forward_windows(df, config)
    print(f"  Walk-forward windows: {len(windows)}")

    # Evaluate each window
    window_results = []
    for i, window in enumerate(windows):
        print(f"\n  --- Window {i}: "
              f"Train {window['train_start'].year}-{window['train_end'].year}, "
              f"Test {window['test_start'].year}-{window['test_end'].year} ---")

        result = evaluate_window(config, stock_name, i, window, df, device)
        if result is not None:
            window_results.append(result)

            # Print a quick summary for this window
            mc = result["mc_dropout_metrics"]
            tr = result["training"]
            dist = result["prediction_distribution"]
            print(f"    Accuracy: {mc['accuracy']:.4f} | "
                  f"F1: {mc['f1']:.4f} | "
                  f"AUC: {mc['roc_auc']:.4f} | "
                  f"Best Epoch: {tr['best_epoch']} | "
                  f"Pred Std: {dist['std']:.4f}")

    # Aggregate metrics
    aggregates = compute_aggregates(window_results)

    # Build final report
    report = {
        "stock": stock_name,
        "evaluated_at": datetime.now().isoformat(),
        "config_snapshot": {
            "seq_length": config["model"]["seq_length"],
            "hidden_size": config["model"]["hidden_size"],
            "num_layers": config["model"]["num_layers"],
            "num_heads": config["model"]["num_heads"],
            "dropout": config["model"]["dropout"],
            "learning_rate": config["model"]["learning_rate"],
            "batch_size": config["model"]["batch_size"],
        },
        "aggregate": aggregates,
        "windows": window_results,
    }

    # Save to per-stock folder
    output_dir = os.path.join(config["results_dir"], stock_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE: {stock_name}")
    print(f"{'=' * 70}")
    if aggregates:
        print(f"  Mean Accuracy:     {aggregates['accuracy']['mean']:.4f} ± {aggregates['accuracy']['std']:.4f}")
        print(f"  Majority Baseline: {aggregates['majority_baseline']['mean']:.4f}")
        print(f"  Accuracy Lift:     {aggregates['accuracy_lift']['mean']:+.4f}")
        print(f"  Balanced Accuracy: {aggregates['balanced_accuracy']['mean']:.4f} ± {aggregates['balanced_accuracy']['std']:.4f}")
        print(f"  MCC:               {aggregates['mcc']['mean']:.4f} ± {aggregates['mcc']['std']:.4f}")
        print(f"  Mean ROC-AUC:      {aggregates['roc_auc']['mean']:.4f} ± {aggregates['roc_auc']['std']:.4f}")
        print(f"  Mean F1:           {aggregates['f1']['mean']:.4f} ± {aggregates['f1']['std']:.4f}")
        print(f"  Overfitting Gap:   {aggregates['overfitting']['mean_train_val_gap']:.6f}")
        print(f"  Best Window:       {aggregates['best_window']['period']} ({aggregates['best_window']['accuracy']:.4f})")
        print(f"  Worst Window:      {aggregates['worst_window']['period']} ({aggregates['worst_window']['accuracy']:.4f})")
    print(f"  Saved to: {output_path}")

    return report


# =============================================
# CLI ENTRY POINT
# =============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Stage 1 models")
    parser.add_argument("--stock", type=str, default=None,
                        help="Stock name (default: evaluate all stocks in config)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stock:
        # Evaluate a single stock
        evaluate_stock(args.stock, config)
    else:
        # Evaluate ALL stocks from config
        for stock_cfg in config["stocks"]:
            stock_name = stock_cfg["name"]
            try:
                evaluate_stock(stock_name, config)
            except FileNotFoundError as e:
                print(f"  Skipping {stock_name}: {e}")
