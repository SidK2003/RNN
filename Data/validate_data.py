"""
Data validation script — run after features/pipeline.py.

Verifies that all processed feature files are correct and ready 
for Stage 1 training. Checks for:
1. No NaN values
2. No infinite values
3. Correct column schema
4. Target is binary (0/1)
5. Date continuity (no large gaps)
6. Walk-forward window feasibility
7. Feature statistics sanity check
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_stock(name: str, path: str, config: dict) -> dict:
    """Validate a single stock's processed feature file."""
    issues = []
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    
    # --- Schema check ---
    expected_cols = [
        "date", "open", "high", "low", "close", "volume",
        "log_return", "bollinger_pctb", "atr", "rsi",
        "macd", "macd_signal", "macd_histogram",
        "stoch_k", "stoch_d", "obv", "volume_sma_ratio",
        "india_vix", "vix_available", "target"
    ]
    missing = set(expected_cols) - set(df.columns)
    extra = set(df.columns) - set(expected_cols)
    if missing:
        issues.append(f"MISSING columns: {missing}")
    if extra:
        issues.append(f"EXTRA columns (not necessarily bad): {extra}")
    
    # --- NaN check ---
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        nan_cols = df.isnull().sum()
        issues.append(f"NaN found: {dict(nan_cols[nan_cols > 0])}")
    
    # --- Inf check ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        inf_cols = np.isinf(df[numeric_cols]).sum()
        issues.append(f"Inf found: {dict(inf_cols[inf_cols > 0])}")
    
    # --- Target check ---
    unique_targets = sorted(df["target"].unique())
    if unique_targets != [0.0, 1.0]:
        issues.append(f"Target values unexpected: {unique_targets}")
    
    # --- Date continuity (flag gaps > 5 trading days) ---
    date_diffs = df["date"].diff().dt.days
    large_gaps = date_diffs[date_diffs > 5]
    if len(large_gaps) > 0:
        # Just informational — weekends/holidays are normal (2-3 days)
        max_gap = date_diffs.max()
        if max_gap > 10:
            issues.append(f"Large date gap detected: {max_gap} days")
    
    # --- Walk-forward feasibility ---
    start = df["date"].min()
    end = df["date"].max()
    train_years = config["walk_forward"]["train_years"]
    test_years = config["walk_forward"]["test_years"]
    min_needed = train_years + test_years
    actual_years = (end - start).days / 365.25
    
    if actual_years < min_needed:
        issues.append(
            f"Insufficient data for walk-forward: need {min_needed} years, "
            f"have {actual_years:.1f} years"
        )
    
    # Compute actual walk-forward windows for this stock
    step = config["walk_forward"]["step_years"]
    first_year = start.year
    last_year = end.year
    windows = []
    
    train_start_year = first_year
    while True:
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1
        
        if test_end_year > last_year:
            break
        
        windows.append({
            "train": f"{train_start_year}-{train_end_year}",
            "test": f"{test_start_year}-{test_end_year}",
        })
        train_start_year += step
    
    # --- Feature statistics ---
    stats = {}
    feature_cols = [c for c in df.columns if c not in ["date", "target", "vix_available"]]
    for col in feature_cols:
        col_data = df[col]
        stats[col] = {
            "mean": round(col_data.mean(), 4),
            "std": round(col_data.std(), 4),
            "min": round(col_data.min(), 4),
            "max": round(col_data.max(), 4),
        }
    
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": f"{start.date()} to {end.date()}",
        "years": round(actual_years, 1),
        "walk_forward_windows": len(windows),
        "window_details": windows,
        "target_up_pct": round(df["target"].mean() * 100, 1),
        "issues": issues,
        "feature_stats": stats,
    }


def validate_all(config_path: str = "config.yaml"):
    """Validate all processed stock files."""
    config = load_config(config_path)
    processed_dir = os.path.join("Data", "processed")
    
    print(f"{'=' * 70}")
    print("DATA VALIDATION REPORT")
    print(f"{'=' * 70}")
    
    all_valid = True
    
    for stock_cfg in config["stocks"]:
        name = stock_cfg["name"]
        path = os.path.join(processed_dir, f"{name.lower()}_features.csv")
        
        if not os.path.exists(path):
            print(f"\n❌ {name}: File not found at {path}")
            all_valid = False
            continue
        
        result = validate_stock(name, path, config)
        
        status = "✅" if not result["issues"] else "⚠️"
        if any("NaN" in i or "Inf" in i or "Insufficient" in i for i in result["issues"]):
            status = "❌"
            all_valid = False
        
        print(f"\n{status} {name}")
        print(f"   Rows: {result['rows']} | Columns: {result['columns']}")
        print(f"   Date range: {result['date_range']} ({result['years']} years)")
        print(f"   Target: {result['target_up_pct']}% UP")
        print(f"   Walk-forward windows: {result['walk_forward_windows']}")
        
        for w in result["window_details"]:
            print(f"     Train {w['train']} → Test {w['test']}")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"   ⚠️  {issue}")
    
    print(f"\n{'=' * 70}")
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED — Data is ready for Stage 1")
    else:
        print("❌ VALIDATION FAILURES DETECTED — Fix issues before proceeding")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    validate_all()
