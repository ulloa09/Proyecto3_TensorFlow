import numpy as np
import pandas as pd
import pandas_ta as ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a set of technical features from OHLCV data using pandas_ta.
    Mixes momentum, volatility, and volume indicators.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns:
        pd.DataFrame: Original DataFrame expanded with new feature columns.
    """

    df = df.copy()

    # ==== MOMENTUM ====
    # RSI (short and medium term)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_28"] = ta.rsi(df["Close"], length=28)

    # MACD (two configurations)
    macd_fast = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    macd_slow = ta.macd(df["Close"], fast=19, slow=39, signal=9)
    df["macd_12_26"] = macd_fast["MACD_12_26_9"]
    df["macd_signal_12_26"] = macd_fast["MACDs_12_26_9"]
    df["macd_19_39"] = macd_slow["MACD_19_39_9"]
    df["macd_signal_19_39"] = macd_slow["MACDs_19_39_9"]

    # Stochastic %K (two scales)
    stoch_fast = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    stoch_slow = ta.stoch(df["High"], df["Low"], df["Close"], k=28, d=3)
    df["stoch_k_14"] = stoch_fast["STOCHk_14_3_3"]
    df["stoch_k_28"] = stoch_slow["STOCHk_28_3_3"]

    # ADX (two windows)
    df["adx_14"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]
    df["adx_28"] = ta.adx(df["High"], df["Low"], df["Close"], length=28)["ADX_28"]

    # Momentum and Rate of Change
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["roc_20"] = ta.roc(df["Close"], length=20)

    # Williams %R
    df["williams_r_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)


    
    # ==== VOLATILITY ====
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_28"] = ta.atr(df["High"], df["Low"], df["Close"], length=28)

    # Bollinger Bands (width and %B)
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["bb_width"] = (bbands["BBU_20_2.0_2.0"] - bbands["BBL_20_2.0_2.0"]) / bbands["BBM_20_2.0_2.0"]
    df["bb_percent_b"] = (df["Close"] - bbands["BBL_20_2.0_2.0"]) / (bbands["BBU_20_2.0_2.0"] - bbands["BBL_20_2.0_2.0"])

    # Std Dev and Donchian
    df["std_20"] = df["Close"].rolling(20).std()
    df["donchian_width"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["tr_norm"] = df["atr_14"] / df["Close"] # Normalized True Range


    # ==== VOLUME ====
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["mfi_14"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["cmf_20"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_zscore_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["vol_spike_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()

    # --- VWAP deviation (without modifying main index) ---
    # pandas_ta's VWAP requires a DatetimeIndex.
    df_temp = df.copy()

    # If 'Date' column exists, convert and set as index only in the copy
    if "Date" in df_temp.columns:
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])
        df_temp = df_temp.sort_values("Date")  # Sort by date
        df_temp = df_temp.set_index("Date")

    # Calculate VWAP using the copy with DatetimeIndex
    vwap_series = ta.vwap(df_temp["High"], df_temp["Low"], df_temp["Close"], df_temp["Volume"])

    # If VWAP returns something valid, align it to the original index
    if vwap_series is not None:
        vwap_series = vwap_series.reset_index(drop=True)
        # Calculate deviation from VWAP
        df["vwap_dev"] = (df["Close"] - vwap_series) / vwap_series
    else:
        print("⚠️ VWAP could not be calculated (returned None). Setting vwap_dev to None.")
        df["vwap_dev"] = None

    # --- Final Cleanup ---
    print(f"Total Data prior to feature NA drop: {len(df)}")
    # Drop rows with NaNs created by indicators (e.g., initial SMA periods)
    df = df.dropna()

    print(f"Total NA values after Drop: {df.isna().sum().sum()}")
    print(f"Total Data after feature NA drop: {len(df)}")
    return df