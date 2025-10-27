import yfinance as yf
import numpy as np
import pandas as pd
import os
import config # Import the master configuration file

# --- Data Download Script ---
# This script downloads daily data for the ticker defined in config.py
# and saves it to the CSV path defined in config.py.

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Get parameters from config
TICKER = config.TICKER
START_DATE = config.START_DATE
END_DATE = config.END_DATE
OUTPUT_PATH = config.DATA_CSV_PATH

# Direct and clean download from scratch
print(f"Downloading {TICKER} data ({START_DATE} to {END_DATE})...")
data = yf.download(
    tickers=TICKER,
    start=START_DATE,
    end=END_DATE, 
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
data.to_csv(OUTPUT_PATH, index=False)

print(data.head())
print(f"Data downloaded: {len(data)} records between {data['Date'].min()} and {data['Date'].max()}")
print(f"Saved to {OUTPUT_PATH}")