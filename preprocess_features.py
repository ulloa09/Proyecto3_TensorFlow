import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

def fechas(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Añadimos Datetime para compatibilidad con backtest
    if 'Date' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'])
    return df

def fit_scalers(train_df):
    train_df = train_df.copy()

    """Entrena los cuatro escaladores con los datos de entrenamiento."""
    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Filtrar columnas que realmente existen en el df
    minmax_cols = [col for col in minmax_cols if col in train_df.columns]
    robust_cols = [col for col in robust_cols if col in train_df.columns]
    standard_cols = [col for col in standard_cols if col in train_df.columns]
    ohlcv_cols = [col for col in ohlcv_cols if col in train_df.columns]

    mm_scaler, rb_scaler, st_scaler, ohlcv_scaler = None, None, None, None

    try:
        if minmax_cols:
            mm_scaler = MinMaxScaler().fit(train_df[minmax_cols])
            dump(mm_scaler, "scaler_minmax.pkl")
            print(f"MinMaxScaler entrenado con {len(minmax_cols)} cols.")
        if robust_cols:
            rb_scaler = RobustScaler().fit(train_df[robust_cols])
            dump(rb_scaler, "scaler_robust.pkl")
            print(f"RobustScaler entrenado con {len(robust_cols)} cols.")
        if standard_cols:
            st_scaler = StandardScaler().fit(train_df[standard_cols])
            dump(st_scaler, "scaler_standard.pkl")
            print(f"StandardScaler entrenado con {len(standard_cols)} cols.")
        if ohlcv_cols:
            ohlcv_scaler = StandardScaler().fit(train_df[ohlcv_cols])
            dump(ohlcv_scaler, "scaler_ohlcv.pkl")
            print(f"StandardScaler (OHLCV) entrenado con {len(ohlcv_cols)} cols.")
            
    except ValueError as e:
        print(f"Error al entrenar scalers: {e}. DataFrame podría tener NaNs o estar vacío.")
        return None, None, None, None

    return mm_scaler, rb_scaler, st_scaler, ohlcv_scaler


def apply_scalers(df, mm_scaler_path="scaler_minmax.pkl", rb_scaler_path="scaler_robust.pkl", st_scaler_path="scaler_standard.pkl", ohlcv_scaler_path="scaler_ohlcv.pkl"):
    df = df.copy()
    """Aplica los escaladores ya entrenados a un DataFrame."""
    
    try: mm_scaler = load(mm_scaler_path)
    except FileNotFoundError: mm_scaler = None
    try: rb_scaler = load(rb_scaler_path)
    except FileNotFoundError: rb_scaler = None
    try: st_scaler = load(st_scaler_path)
    except FileNotFoundError: st_scaler = None
    try: ohlcv_scaler = load(ohlcv_scaler_path)
    except FileNotFoundError: ohlcv_scaler = None

    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Filtrar columnas que realmente existen en el df
    minmax_cols = [col for col in minmax_cols if col in df.columns]
    robust_cols = [col for col in robust_cols if col in df.columns]
    standard_cols = [col for col in standard_cols if col in df.columns]
    ohlcv_cols = [col for col in ohlcv_cols if col in df.columns]

    try:
        if mm_scaler and minmax_cols:
            df[minmax_cols] = mm_scaler.transform(df[minmax_cols])
        if rb_scaler and robust_cols:
            df[robust_cols] = rb_scaler.transform(df[robust_cols])
        if st_scaler and standard_cols:
            df[standard_cols] = st_scaler.transform(df[standard_cols])
        if ohlcv_scaler and ohlcv_cols:
            df[ohlcv_cols] = ohlcv_scaler.transform(df[ohlcv_cols])
    except ValueError as e:
        print(f"Error al aplicar scalers: {e}. Omitiendo escalado para el set.")
        
    return df