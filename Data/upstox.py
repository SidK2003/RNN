import os
import requests
import pandas as pd

from dotenv import load_dotenv
from urllib.parse import quote
from datetime import datetime, timedelta

# -----------------------------------
# LOAD ENV
# -----------------------------------

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# -----------------------------------
# CONFIG
# -----------------------------------

instrument_key = "NSE_EQ|INE002A01018"
encoded_key = quote(instrument_key, safe="")

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

base_url = "https://api.upstox.com/v3/historical-candle"

# -----------------------------------
# DATE RANGES
# -----------------------------------

start_date = datetime(2000, 1, 1)
end_date = datetime(2026, 5, 17)

all_candles = []

current_start = start_date

while current_start < end_date:

    # fetch 1 year at a time
    current_end = min(
        current_start + timedelta(days=364),
        end_date
    )

    from_date = current_start.strftime("%Y-%m-%d")
    to_date = current_end.strftime("%Y-%m-%d")

    url = (
        f"{base_url}/"
        f"{encoded_key}/days/1/"
        f"{to_date}/{from_date}"
    )

    print(f"Fetching: {from_date} -> {to_date}")

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("FAILED:", response.text)
        current_start = current_end + timedelta(days=1)
        continue

    data = response.json()

    candles = data["data"]["candles"]

    all_candles.extend(candles)

    current_start = current_end + timedelta(days=1)

# -----------------------------------
# DATAFRAME
# -----------------------------------

df = pd.DataFrame(
    all_candles,
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oi"
    ]
)

# remove duplicates
df.drop_duplicates(inplace=True)

# convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# oldest first
df = df.sort_values("timestamp").reset_index(drop=True)

print(df.head())
print(df.tail())

# -----------------------------------
# SAVE CSV
# -----------------------------------

df.to_csv("nifty50_daily.csv", index=False)

print("\nSaved successfully.")
print("Total rows:", len(df))