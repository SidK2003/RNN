"""
Visualization module for the evaluation pipeline.

In simple words: This module draws the charts that tell the story.
Numbers are precise but boring — charts make it immediately obvious
whether the system is working or not.

All functions take strategy results as input and save PNGs.
No state, no side effects beyond file I/O.

Charts generated:
1. Equity Curves — 3 strategies overlaid (the money chart)
2. Drawdown Chart — underwater plot for RL strategy (the pain chart)
3. Trade Scatter — BUY/SELL markers on price chart (the decision chart)
4. Metric Comparison Bars — side-by-side strategy comparison (the scorecard)
5. Aggregate Summary Table — cross-window average metrics (the report card)

All plots use matplotlib for simplicity and consistent PNG output.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI windows, just save PNGs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================
# CHART STYLE CONFIGURATION
# =============================================

# Use a clean, professional style for all charts
plt.style.use("seaborn-v0_8-darkgrid")

# Color palette: visually distinct, colorblind-friendly
COLORS = {
    "buy_and_hold": "#4A90D9",     # Blue — calm, baseline
    "predictor_only": "#F5A623",   # Orange — intermediate
    "full_rl": "#7ED321",          # Green — the system we're testing
    "drawdown": "#D0021B",         # Red — for pain/loss charts
    "buy_marker": "#27AE60",       # Green triangle for BUY
    "sell_marker": "#E74C3C",      # Red triangle for SELL
}

# Strategy display names for legends
STRATEGY_NAMES = {
    "buy_and_hold": "Buy-and-Hold",
    "predictor_only": "Predictor-Only",
    "full_rl": "Full RL System",
}


# =============================================
# CHART 1: EQUITY CURVES
# =============================================

def plot_equity_curves(
    results: Dict[str, Dict],
    stock_name: str,
    window_idx: int,
    save_dir: str,
) -> str:
    """
    Plot 3 equity curves overlaid on the same chart.
    
    This is THE most important chart. It shows how each strategy's
    portfolio value evolved over time on the exact same test data.
    
    Args:
        results: Dict from backtest.run_all_strategies()
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        save_dir: Directory to save the PNG
        
    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for key, color in [("buy_and_hold", COLORS["buy_and_hold"]),
                       ("predictor_only", COLORS["predictor_only"]),
                       ("full_rl", COLORS["full_rl"])]:
        if key not in results:
            continue
        
        res = results[key]
        equity = res["equity_curve"]
        dates = res["dates"][:len(equity)]
        
        # Convert dates to proper datetime for matplotlib
        dates_dt = pd.to_datetime(dates)
        
        ax.plot(dates_dt, equity, label=STRATEGY_NAMES.get(key, key),
                color=color, linewidth=1.5, alpha=0.9)
    
    # Formatting
    ax.set_title(f"{stock_name} — Window {window_idx} — Equity Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value (starting = 1.0)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Breakeven")
    
    # Format x-axis dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"window{window_idx}_equity.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Saved: {save_path}")
    return save_path


# =============================================
# CHART 2: DRAWDOWN CHART
# =============================================

def plot_drawdown(
    results: Dict[str, Dict],
    stock_name: str,
    window_idx: int,
    save_dir: str,
    strategy_key: str = "full_rl",
) -> str:
    """
    Plot the "underwater" chart — how far below the peak the portfolio fell at each point.
    
    This visualizes the PAIN of holding the strategy. Deep red valleys = bad periods.
    Useful for understanding risk tolerance: "Could I stomach a 15% drawdown?"
    
    Args:
        results: Dict from backtest.run_all_strategies()
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        save_dir: Directory to save the PNG
        strategy_key: Which strategy to plot (default: full_rl)
        
    Returns:
        Path to the saved PNG file.
    """
    if strategy_key not in results:
        return ""
    
    res = results[strategy_key]
    equity = res["equity_curve"]
    dates = pd.to_datetime(res["dates"][:len(equity)])
    
    # Calculate drawdown at each point
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100  # as percentage
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Fill the area under the drawdown curve with red
    ax.fill_between(dates, drawdown, 0, color=COLORS["drawdown"], alpha=0.3)
    ax.plot(dates, drawdown, color=COLORS["drawdown"], linewidth=1.0, alpha=0.8)
    
    # Formatting
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key)
    ax.set_title(f"{stock_name} — Window {window_idx} — {strategy_name} Drawdown", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_ylim(top=2)  # Small positive margin above zero
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"window{window_idx}_drawdown.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Saved: {save_path}")
    return save_path


# =============================================
# CHART 3: TRADE SCATTER
# =============================================

def plot_trade_scatter(
    results: Dict[str, Dict],
    test_df: pd.DataFrame,
    stock_name: str,
    window_idx: int,
    save_dir: str,
    strategy_key: str = "full_rl",
) -> str:
    """
    Plot the stock's close price with BUY (▲) and SELL (▼) markers.
    
    This shows WHEN the agent chose to trade. Good agents buy at local
    dips and sell at local peaks. Bad agents trade randomly or never.
    
    Args:
        results: Dict from backtest.run_all_strategies()
        test_df: Original test DataFrame with close prices
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        save_dir: Directory to save the PNG
        strategy_key: Which strategy to plot trades for
        
    Returns:
        Path to the saved PNG file.
    """
    if strategy_key not in results:
        return ""
    
    res = results[strategy_key]
    trade_log = res["trade_log"]
    
    # Plot the stock price as the background
    dates = pd.to_datetime(test_df["date"].values)
    close_prices = test_df["close"].values
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, close_prices, color="#555555", linewidth=1.0, alpha=0.7, label="Close Price")
    
    # Overlay BUY and SELL markers
    for trade in trade_log:
        trade_date = pd.to_datetime(trade["date"])
        trade_price = trade["price"]
        
        if trade["action"] == "BUY":
            ax.scatter(trade_date, trade_price, marker="^", color=COLORS["buy_marker"],
                      s=100, zorder=5, edgecolors="black", linewidths=0.5)
        elif trade["action"] == "SELL":
            ax.scatter(trade_date, trade_price, marker="v", color=COLORS["sell_marker"],
                      s=100, zorder=5, edgecolors="black", linewidths=0.5)
    
    # Create legend entries manually
    ax.scatter([], [], marker="^", color=COLORS["buy_marker"], s=80, label="BUY", edgecolors="black")
    ax.scatter([], [], marker="v", color=COLORS["sell_marker"], s=80, label="SELL", edgecolors="black")
    
    # Formatting
    strategy_name = STRATEGY_NAMES.get(strategy_key, strategy_key)
    ax.set_title(f"{stock_name} — Window {window_idx} — {strategy_name} Trades",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price (₹)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"window{window_idx}_trades.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Saved: {save_path}")
    return save_path


# =============================================
# CHART 4: METRIC COMPARISON BARS
# =============================================

def plot_metric_bars(
    results: Dict[str, Dict],
    stock_name: str,
    window_idx: int,
    save_dir: str,
) -> str:
    """
    Side-by-side bar chart comparing key metrics across all 3 strategies.
    
    Shows Sortino, Sharpe, Total Return, and Max Drawdown as grouped bars.
    Easy to see at a glance which strategy won on which dimension.
    
    Args:
        results: Dict from backtest.run_all_strategies()
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        save_dir: Directory to save the PNG
        
    Returns:
        Path to the saved PNG file.
    """
    # Metrics to compare
    metric_keys = ["sortino_ratio", "sharpe_ratio", "total_return_pct", "max_drawdown_pct"]
    metric_labels = ["Sortino", "Sharpe", "Return (%)", "Max DD (%)"]
    
    # Extract values for each strategy
    strategies = ["buy_and_hold", "predictor_only", "full_rl"]
    strategy_labels = [STRATEGY_NAMES[s] for s in strategies if s in results]
    strategy_keys = [s for s in strategies if s in results]
    
    n_metrics = len(metric_keys)
    n_strategies = len(strategy_keys)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    
    bar_width = 0.25
    x = np.arange(n_strategies)
    colors = [COLORS[s] for s in strategy_keys]
    
    for idx, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]
        values = [results[s]["metrics"].get(metric_key, 0) for s in strategy_keys]
        
        bars = ax.bar(x, values, width=bar_width * 2.5, color=colors, alpha=0.85, edgecolor="white")
        
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("-", "\n") for s in strategy_labels], fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    
    fig.suptitle(f"{stock_name} — Window {window_idx} — Strategy Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"window{window_idx}_metrics.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Saved: {save_path}")
    return save_path


# =============================================
# CHART 5: AGGREGATE SUMMARY TABLE
# =============================================

def plot_aggregate_table(
    all_window_results: List[Dict],
    stock_name: str,
    save_dir: str,
) -> str:
    """
    Generate a summary PNG table showing average metrics across all walk-forward windows.
    
    This is the final "report card" — one table per stock showing how each strategy
    performed ON AVERAGE across all test periods.
    
    Args:
        all_window_results: List of dicts, each from run_all_strategies() for one window.
        stock_name: e.g., "RELIANCE"
        save_dir: Directory to save the PNG.
        
    Returns:
        Path to the saved PNG file.
    """
    if not all_window_results:
        return ""
    
    # Collect metrics across windows for each strategy
    strategies = ["buy_and_hold", "predictor_only", "full_rl"]
    metric_keys = ["total_return_pct", "annualized_return_pct", "sortino_ratio", 
                   "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate_pct"]
    metric_labels = ["Total Ret%", "Ann. Ret%", "Sortino", "Sharpe", 
                     "MaxDD%", "Trades", "WinRate%"]
    
    # Build the table data
    table_data = []
    for strategy in strategies:
        row = [STRATEGY_NAMES[strategy]]
        for mk in metric_keys:
            # Collect this metric across all windows where this strategy exists
            values = [wr[strategy]["metrics"].get(mk, 0) 
                     for wr in all_window_results if strategy in wr]
            if values:
                avg = np.mean(values)
                row.append(f"{avg:.2f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create the figure with just a table (no axes)
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    
    col_labels = ["Strategy"] + metric_labels
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color the header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    # Color strategy name cells by their strategy color
    strategy_colors_list = [COLORS["buy_and_hold"], COLORS["predictor_only"], COLORS["full_rl"]]
    for i in range(len(strategies)):
        table[i + 1, 0].set_facecolor(strategy_colors_list[i])
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")
    
    fig.suptitle(f"{stock_name} — Aggregate Metrics (Avg Across {len(all_window_results)} Windows)",
                 fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"aggregate_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Saved: {save_path}")
    return save_path


# =============================================
# MASTER FUNCTION
# =============================================
# TODO: Streamlit integration — connect these functions to dashboard tabs

def generate_all_charts(
    results: Dict[str, Dict],
    test_df: pd.DataFrame,
    stock_name: str,
    window_idx: int,
    save_dir: str,
) -> List[str]:
    """
    Generate all charts for a single window evaluation.
    
    Called by walk_forward.py after running all strategies on a window.
    
    Args:
        results: Dict from backtest.run_all_strategies()
        test_df: Original test DataFrame
        stock_name: e.g., "RELIANCE"
        window_idx: Walk-forward window index
        save_dir: Base directory for plots
        
    Returns:
        List of paths to saved PNGs.
    """
    saved_paths = []
    
    # 1. Equity curves
    path = plot_equity_curves(results, stock_name, window_idx, save_dir)
    saved_paths.append(path)
    
    # 2. Drawdown chart (for RL strategy)
    path = plot_drawdown(results, stock_name, window_idx, save_dir)
    if path:
        saved_paths.append(path)
    
    # 3. Trade scatter (for RL strategy)
    path = plot_trade_scatter(results, test_df, stock_name, window_idx, save_dir)
    if path:
        saved_paths.append(path)
    
    # 4. Metric comparison bars
    path = plot_metric_bars(results, stock_name, window_idx, save_dir)
    saved_paths.append(path)
    
    return saved_paths
