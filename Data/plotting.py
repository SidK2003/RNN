import pandas as pd
import plotly.graph_objects as go

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("Data/historical_data/sunpharma_daily.csv")

# =====================================
# PREPROCESS
# =====================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Ensure numeric
df["close"] = pd.to_numeric(df["close"], errors="coerce")

df = df.dropna(subset=["timestamp", "close"])

# =====================================
# CALCULATE PERCENTAGE CHANGE
# =====================================

df["pct_change"] = df["close"].pct_change() * 100

# =====================================
# DETECT LARGE MOVES
# =====================================

THRESHOLD = 15

large_moves = df[
    (df["pct_change"] > THRESHOLD) |
    (df["pct_change"] < -THRESHOLD)
]

print("\n===== LARGE MOVES =====\n")

print(
    large_moves[
        ["timestamp", "close", "pct_change"]
    ]
)

# =====================================
# CREATE INTERACTIVE PLOT
# =====================================

fig = go.Figure()

# Main close price line
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["close"],
        mode="lines",
        name="Close Price"
    )
)

# Highlight suspicious jump points
fig.add_trace(
    go.Scatter(
        x=large_moves["timestamp"],
        y=large_moves["close"],
        mode="markers",
        name="Large Jumps",
        marker=dict(
            size=10,
            symbol="circle"
        ),
        text=[
            f"{pct:.2f}%"
            for pct in large_moves["pct_change"]
        ],
        hovertemplate=
        "<b>Date:</b> %{x}<br>" +
        "<b>Close:</b> %{y}<br>" +
        "<b>Change:</b> %{text}<extra></extra>"
    )
)

# =====================================
# LAYOUT
# =====================================

fig.update_layout(
    title="sunpharma",
    template="plotly_dark",
    hovermode="x unified",
    width=1400,
    height=700,
    dragmode="pan",
    xaxis_rangeslider_visible=True
)

fig.update_xaxes(
    showgrid=True
)

fig.update_yaxes(
    showgrid=True
)

# =====================================
# SHOW PLOT
# =====================================

fig.show()