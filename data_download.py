import yfinance as yf
import numpy as np
import pandas as pd
import os

# --- Data Download Script ---
# This script downloads daily data for the 'WYNN' ticker and saves it
# to a CSV file, which is the starting point for the pipeline.

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Direct and clean download from scratch
print("Downloading WYNN data (2010-2025)...")
data = yf.download(
    tickers="WYNN",
    start="2010-01-01",
    end="2025-10-17", # Note: yfinance 'end' is exclusive
    interval="1d",
    auto_adjust=False, # We want Open, High, Low, Close, Adj Close, Volume
    progress=False,
)

# Flatten if a MultiIndex is returned (due to the ticker)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Reset index to get 'Date' as a column
data = data.reset_index()

# Reorder columns and ensure standard names
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Save clean data
output_path = 'data/wynn_daily_15y.csv'
data.to_csv(output_path, index=False)

print(data.head())
print(f"Data downloaded: {len(data)} records between {data['Date'].min()} and {data['Date'].max()}")
print(f"Saved to {output_path}")