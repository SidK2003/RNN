import json, os

stocks = ["RELIANCE", "TCS", "HDFCBANK", "HINDUNILVR", "SUNPHARMA"]

print(f"{'='*80}")
print(f"{'STOCK':<12} | {'METRIC':<15} | {'OLD (Baseline)':<15} | {'NEW (Current)':<15} | {'CHANGE'}")
print(f"{'='*80}")

for stock in stocks:
    old_path = f"results/baseline_metrics/{stock}_metrics.json"
    new_path = f"results/{stock}/metrics.json"
    
    if not os.path.exists(old_path) or not os.path.exists(new_path):
        continue
        
    old_data = json.load(open(old_path))
    new_data = json.load(open(new_path))
    
    old_agg = old_data["aggregate"]
    new_agg = new_data["aggregate"]
    
    # Calculate avg prediction std
    old_stds = [w["prediction_distribution"]["std"] for w in old_data["windows"]]
    old_avg_std = sum(old_stds) / len(old_stds) if old_stds else 0
    
    new_stds = [w["prediction_distribution"]["std"] for w in new_data["windows"]]
    new_avg_std = sum(new_stds) / len(new_stds) if new_stds else 0
    
    metrics = [
        ("Accuracy", old_agg["accuracy"]["mean"], new_agg["accuracy"]["mean"]),
        ("ROC-AUC", old_agg["roc_auc"]["mean"], new_agg["roc_auc"]["mean"]),
        ("F1 Score", old_agg["f1"]["mean"], new_agg["f1"]["mean"]),
        ("Avg Best Epoch", old_agg["overfitting"]["mean_best_epoch"], new_agg["overfitting"]["mean_best_epoch"]),
        ("Pred Std", old_avg_std, new_avg_std),
    ]
    
    for i, (name, old_val, new_val) in enumerate(metrics):
        diff = new_val - old_val
        stock_name = stock if i == 0 else ""
        diff_str = f"{diff:+.4f}" if name != "Avg Best Epoch" else f"{diff:+.1f}"
        
        if name == "Avg Best Epoch":
            print(f"{stock_name:<12} | {name:<15} | {old_val:<15.1f} | {new_val:<15.1f} | {diff_str}")
        else:
            print(f"{stock_name:<12} | {name:<15} | {old_val:<15.4f} | {new_val:<15.4f} | {diff_str}")
            
    print(f"{'-'*80}")
