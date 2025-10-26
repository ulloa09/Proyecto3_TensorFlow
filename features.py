import numpy as np
import pandas as pd
import pandas_ta as ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera 25 features técnicas a partir de datos OHLCV.
    Mezcla indicadores de momento, volatilidad y volumen.
    """

    df = df.copy()

    # ==== MOMENTO ====
    # RSI (corto y medio plazo)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_28"] = ta.rsi(df["Close"], length=28)

    # MACD (dos configuraciones)
    macd_fast = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    macd_slow = ta.macd(df["Close"], fast=19, slow=39, signal=9)
    df["macd_12_26"] = macd_fast["MACD_12_26_9"]
    df["macd_signal_12_26"] = macd_fast["MACDs_12_26_9"]
    df["macd_19_39"] = macd_slow["MACD_19_39_9"]
    df["macd_signal_19_39"] = macd_slow["MACDs_19_39_9"]

    # Stochastic %K (dos escalas)
    stoch_fast = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    stoch_slow = ta.stoch(df["High"], df["Low"], df["Close"], k=28, d=3)
    df["stoch_k_14"] = stoch_fast["STOCHk_14_3_3"]
    df["stoch_k_28"] = stoch_slow["STOCHk_28_3_3"]

    # ADX (dos ventanas)
    df["adx_14"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]
    df["adx_28"] = ta.adx(df["High"], df["Low"], df["Close"], length=28)["ADX_28"]

    # Momentum y Rate of Change
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["roc_20"] = ta.roc(df["Close"], length=20)

    # Williams %R
    df["williams_r_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)


    
    # ==== VOLATILIDAD ====
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_28"] = ta.atr(df["High"], df["Low"], df["Close"], length=28)

    # Bollinger Bands (ancho y %B)
    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["bb_width"] = (bbands["BBU_20_2.0_2.0"] - bbands["BBL_20_2.0_2.0"]) / bbands["BBM_20_2.0_2.0"]
    df["bb_percent_b"] = (df["Close"] - bbands["BBL_20_2.0_2.0"]) / (bbands["BBU_20_2.0_2.0"] - bbands["BBL_20_2.0_2.0"])

    # Std y Donchian
    df["std_20"] = df["Close"].rolling(20).std()
    df["donchian_width"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["tr_norm"] = df["atr_14"] / df["Close"]


    # ==== VOLUMEN ====
    
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["mfi_14"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["cmf_20"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)

    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_zscore_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["vol_spike_ratio"] = df["Volume"] / df["Volume"].rolling(5).mean()

    # --- VWAP deviation (sin modificar índice principal) ---
    # Requisito de ahuevo de pandas_ta
    df_temp = df.copy()

    # Si hay columna 'Date', la convertimos y la usamos solo en la copia
    if "Date" in df_temp.columns:
  
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])
        df_temp = df_temp.sort_values("Date")  # ordenar por fecha
        df_temp = df_temp.set_index("Date")

    # Calculamos VWAP usando la copia con DatetimeIndex
    vwap_series = ta.vwap(df_temp["High"], df_temp["Low"], df_temp["Close"], df_temp["Volume"])

    # Si VWAP devuelve algo válido, lo alineamos al índice original
    if vwap_series is not None:
        vwap_series = vwap_series.reset_index(drop=True)
        df["vwap_dev"] = (df["Close"] - vwap_series) / vwap_series
    else:
        print("⚠️ VWAP no pudo calcularse (devuelve None).")
        df["vwap_dev"] = None




    # Eliminar filas con Nas
    print(f"Total de Datos previo a drop: {len(df)}")
    df = df.dropna()

    print(f"Datos NA después de Drop:\n{df.isna().sum().sum()}")
    print(f"Total de Datos después de drop: {len(df)}")
    return df