import numpy as np
import pandas as pd
import pandas_ta as ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ~27 features técnicas a partir de datos OHLCV.
    Mezcla indicadores de momento, volatilidad y volumen.
    Incluye VWAP dev y ejecuta dropna() al final.
    """
    df = df.copy()
    if df.empty:
        print("Advertencia: DataFrame de entrada vacío.")
        return df

    # Resetear índice si tiene duplicados o no es estándar
    if df.index.has_duplicates or not isinstance(df.index, pd.RangeIndex):
        print("Advertencia: Índice no estándar o duplicado, reseteando índice.")
        df = df.reset_index(drop=True)

    # Convertir columnas a numérico
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop inicial solo si OHLCV tiene NaNs
    initial_len = len(df)
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True, how='any')
    if df.empty:
        print(f"Advertencia: DataFrame vacío después de dropear NaNs en OHLCV.")
        return df
    if len(df) < initial_len:
         print(f"Filas eliminadas por NaNs en OHLCV: {initial_len - len(df)}")


    # ==== MOMENTO ====
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_28"] = ta.rsi(df["Close"], length=28)

    macd_fast = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    macd_slow = ta.macd(df["Close"], fast=19, slow=39, signal=9)
    # Asignar NaN si falla el cálculo
    df["macd_12_26"] = macd_fast["MACD_12_26_9"] if macd_fast is not None and "MACD_12_26_9" in macd_fast else np.nan
    df["macd_signal_12_26"] = macd_fast["MACDs_12_26_9"] if macd_fast is not None and "MACDs_12_26_9" in macd_fast else np.nan
    df["macd_19_39"] = macd_slow["MACD_19_39_9"] if macd_slow is not None and "MACD_19_39_9" in macd_slow else np.nan
    df["macd_signal_19_39"] = macd_slow["MACDs_19_39_9"] if macd_slow is not None and "MACDs_19_39_9" in macd_slow else np.nan

    stoch_fast = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3, smooth_k=3) # Usar smooth_k=3 es común
    stoch_slow = ta.stoch(df["High"], df["Low"], df["Close"], k=28, d=3, smooth_k=3)
    df["stoch_k_14"] = stoch_fast["STOCHk_14_3_3"] if stoch_fast is not None and "STOCHk_14_3_3" in stoch_fast else np.nan
    df["stoch_k_28"] = stoch_slow["STOCHk_28_3_3"] if stoch_slow is not None and "STOCHk_28_3_3" in stoch_slow else np.nan

    adx14 = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    adx28 = ta.adx(df["High"], df["Low"], df["Close"], length=28)
    df["adx_14"] = adx14["ADX_14"] if adx14 is not None and "ADX_14" in adx14 else np.nan
    df["adx_28"] = adx28["ADX_28"] if adx28 is not None and "ADX_28" in adx28 else np.nan

    df["momentum_10"] = df["Close"].diff(10) # Usar diff en lugar de shift
    df["roc_20"] = ta.roc(df["Close"], length=20)
    df["williams_r_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)

    # ==== VOLATILIDAD ====
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_28"] = ta.atr(df["High"], df["Low"], df["Close"], length=28)

    # Bollinger Bands (ancho y %B) - Usando nombres originales BBU_20_2.0_2.0 etc.
    bbands = None
    try: bbands = ta.bbands(df["Close"], length=20, std=2)
    except Exception as e: print(f"⚠️ Error calculando BBands: {e}")

    df["bb_width"] = np.nan
    df["bb_percent_b"] = np.nan
    if bbands is not None and isinstance(bbands, pd.DataFrame):
        # Intentar con el nombre de columna original que tenías
        bbu_col, bbm_col, bbl_col = "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0" #<- Probando nombres sin '_2.0' al final
        if all(col in bbands.columns for col in [bbu_col, bbm_col, bbl_col]):
            bbm_safe = bbands[bbm_col].replace(0, np.nan)
            bb_range = (bbands[bbu_col] - bbands[bbl_col]).replace(0, np.nan)
            if bbm_safe is not None: df["bb_width"] = (bbands[bbu_col] - bbands[bbl_col]) / bbm_safe
            if bb_range is not None: df["bb_percent_b"] = (df["Close"] - bbands[bbl_col]) / bb_range
        else:
             print(f"Advertencia: Columnas BBands ('{bbu_col}', etc.) no encontradas. Presentes: {list(bbands.columns)}")


    # Std y Donchian
    df["std_20"] = df["Close"].rolling(20).std()
    # Usar rolling min/max para replicar Donchian Width original
    df["donchian_width"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["tr_norm"] = df["atr_14"] / df["Close"].replace(0, np.nan)

    # ==== VOLUMEN ====
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["mfi_14"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["cmf_20"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20)
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    vol_std_20 = df["Volume"].rolling(20).std()
    df["vol_zscore_20"] = (df["Volume"] - df["vol_sma_20"]) / vol_std_20.replace(0, np.nan)
    vol_sma_5 = df["Volume"].rolling(5).mean()
    df["vol_spike_ratio"] = df["Volume"] / vol_sma_5.replace(0, np.nan)

    # --- VWAP deviation (Lógica Original) ---
    df["vwap_dev"] = np.nan # Inicializar
    df_temp = df.copy() # Copiar el df actual (puede tener NaNs de indicadores previos)

    # Asegurar Date y poner como índice
    if "Date" in df_temp.columns:
        df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors='coerce')
        # Dropear filas donde Date es NaT antes de set_index
        df_temp.dropna(subset=['Date'], inplace=True)
        if not df_temp.empty:
            try:
                df_temp = df_temp.set_index("Date").sort_index()
                # Calcular VWAP solo si df_temp no está vacío después de set_index
                if not df_temp.empty:
                    vwap_series_indexed = ta.vwap(df_temp["High"], df_temp["Low"], df_temp["Close"], df_temp["Volume"])

                    if vwap_series_indexed is not None and isinstance(vwap_series_indexed, pd.Series):
                        # Alinear al índice original del DataFrame 'df' (que debería ser RangeIndex)
                        # Reindexar vwap_series al índice de df_temp para asegurar alineación por fecha
                        vwap_aligned = vwap_series_indexed.reindex(df_temp.index)
                        # Resetear el índice para que coincida con el RangeIndex de 'df'
                        vwap_series = vwap_aligned.reset_index(drop=True)

                        # Verificar longitud antes de asignar
                        if len(vwap_series) == len(df):
                           df["vwap_dev"] = (df["Close"] - vwap_series) / vwap_series.replace(0, np.nan)
                        else:
                            print(f"⚠️ Advertencia VWAP: Discrepancia de longitud ({len(vwap_series)} vs {len(df)}). Asignando NaN.")
                    else: print("⚠️ VWAP no pudo calcularse (returned None or wrong type).")

            except Exception as e: print(f"⚠️ Error procesando VWAP con DatetimeIndex: {e}")
        else: print("⚠️ DataFrame temporal vacío después de dropna(Date) para VWAP.")
    else: print("⚠️ Columna 'Date' no encontrada para cálculo de VWAP.")


    # --- Eliminar filas con NaNs (Lógica Original) ---
    print(f"Total de Datos previo a dropna final: {len(df)}")
    rows_before = len(df)
    df = df.dropna() # Eliminar cualquier fila con CUALQUIER NaN
    rows_after = len(df)
    print(f"Filas eliminadas por dropna final: {rows_before - rows_after}")
    print(f"Datos NA restantes después de Drop: {df.isna().sum().sum()}") # Debería ser 0
    print(f"Total de Datos después de dropna final: {len(df)}")

    return df