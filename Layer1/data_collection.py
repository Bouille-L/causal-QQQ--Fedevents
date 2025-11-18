import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# === CONFIG ===
# NOTE: The data supplier API restricts loading large date ranges in a single request.
# Therefore, we load data in 2-year spans from 2011 to 2025 and combine them to cover the full period.
token = "96dc57a3-71fc-463e-8ba6-8c6714e24b39"
ticker = "QQQ"
start_date = datetime(2011, 6, 17)
end_date = datetime(2025, 6, 16)  # Fixed endpoint to save API calls

# === DATA STORAGE ===
all_data = []

# === LOOP THROUGH EACH TRADE DATE ===
date = start_date
while date <= end_date:
    trade_date = date.strftime("%Y-%m-%d")
    print(f"Fetching {ticker} for {trade_date}...")

    url = "https://api.orats.io/datav2/hist/strikes"
    params = {
        "token": token,
        "ticker": ticker,
        "tradeDate": trade_date
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            json_data = response.json().get("data", [])
            if json_data:
                all_data.extend(json_data)
            else:
                print(f"No data for {trade_date}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception on {trade_date}: {e}")

    date += timedelta(days=1)

# === CONVERT TO DATAFRAME ===
df = pd.DataFrame(all_data)

# === Fix Data Types for Compatibility with Parquet ===
if not df.empty:
    # Force numeric columns to correct type, ignore errors
    numeric_cols = ['residualRate', 'delta', 'gamma', 'theta', 'vega', 'rho', 'phi', 'driftlessTheta']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # === SAVE TO PARQUET FILE ===
    filename = "qqq_optiondata_8.parquet"
    df.to_parquet(filename, index=False)
    print(f"\n✅ Done! Saved {len(df)} rows to {filename}")
    print(f"📁 File path: {os.path.abspath(filename)}")
else:
    print("⚠️ No data collected. Nothing was saved.")

