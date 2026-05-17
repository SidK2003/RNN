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

if not ACCESS_TOKEN:
    raise ValueError("ACCESS_TOKEN not found in .env")

# -----------------------------------
# CONFIG
# -----------------------------------

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

base_url = "https://api.upstox.com/v3/historical-candle"

# -----------------------------------
# STOCKS / INDICES
# -----------------------------------

assets = {
    "NIFTY50": "NSE_INDEX|Nifty 50",
    "INDIA_VIX": "NSE_INDEX|India VIX",

    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
}

# -----------------------------------
# DATE RANGE
# -----------------------------------

start_date = datetime(2000, 1, 1)
end_date = datetime.now()

# -----------------------------------
# OUTPUT DIRECTORY
# -----------------------------------

output_dir = "historical_data"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# FETCH FUNCTION
# -----------------------------------

def fetch_historical_data(asset_name, instrument_key):

    print(f"\n{'=' * 60}")
    print(f"FETCHING: {asset_name}")
    print(f"{'=' * 60}")

    encoded_key = quote(instrument_key, safe="")

    all_candles = []

    current_start = start_date

    while current_start < end_date:

        # 1-year chunks
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

        print(f"[{asset_name}] {from_date} -> {to_date}")

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"FAILED: {response.text}")

                current_start = current_end + timedelta(days=1)
                continue

            data = response.json()

            candles = data["data"]["candles"]

            all_candles.extend(candles)

        except Exception as e:
            print(f"ERROR: {e}")

        current_start = current_end + timedelta(days=1)

    # -----------------------------------
    # CREATE DATAFRAME
    # -----------------------------------

    if not all_candles:
        print(f"No data found for {asset_name}")
        return

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

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove timezone
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Sort oldest -> newest
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add asset column
    df["asset"] = asset_name

    # -----------------------------------
    # SAVE CSV
    # -----------------------------------

    filename = f"{asset_name.lower()}_daily.csv"

    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)

    print(f"\nSaved: {filepath}")
    print(f"Rows: {len(df)}")

    print(df.head())
    print(df.tail())


# -----------------------------------
# MAIN LOOP
# -----------------------------------

for asset_name, instrument_key in assets.items():

    fetch_historical_data(
        asset_name,
        instrument_key
    )

print("\nALL DOWNLOADS COMPLETED")