import pandas as pd

# Load CSV
df = pd.read_csv("Data/historical_data/reliance_daily.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort properly
df = df.sort_values("timestamp")

# Percentage change
df["pct_change"] = df["close"].pct_change() * 100

# Detect suspicious jumps
large_moves = df[
    (df["pct_change"] > 40) |
    (df["pct_change"] < -40)
]

# Print results
print(
    large_moves[
        ["timestamp", "close", "pct_change"]
    ]
)