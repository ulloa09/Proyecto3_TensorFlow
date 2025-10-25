import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

def fechas(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
def fit_scalers(train_df):
    """Entrena los escaladores con train_df. Maneja df vacío. VWAP excluido."""
    if train_df.empty:
        print("Error: train_df vacío en fit_scalers."); return None, None, None

    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vol_spike_ratio", "bb_width"]
    
    minmax_cols_present = [col for col in minmax_cols if col in train_df.columns and train_df[col].notna().any()]
    robust_cols_present = [col for col in robust_cols if col in train_df.columns and train_df[col].notna().any()]
    standard_cols_present = [col for col in standard_cols if col in train_df.columns and train_df[col].notna().any()]

    mm_scaler, rb_scaler, st_scaler = None, None, None
    if minmax_cols_present:
        try: mm_scaler = MinMaxScaler().fit(train_df[minmax_cols_present]); dump(mm_scaler, "scaler_minmax.pkl"); print(f"MinMaxScaler entrenado con {len(minmax_cols_present)} cols.")
        except ValueError as e: print(f"Error MinMaxScaler: {e}"); mm_scaler = None
    else: print("Advertencia: No cols para MinMaxScaler.")
    if robust_cols_present:
        try: rb_scaler = RobustScaler().fit(train_df[robust_cols_present]); dump(rb_scaler, "scaler_robust.pkl"); print(f"RobustScaler entrenado con {len(robust_cols_present)} cols.")
        except ValueError as e: print(f"Error RobustScaler: {e}"); rb_scaler = None
    else: print("Advertencia: No cols para RobustScaler.")
    if standard_cols_present:
        try: st_scaler = StandardScaler().fit(train_df[standard_cols_present]); dump(st_scaler, "scaler_standard.pkl"); print(f"StandardScaler entrenado con {len(standard_cols_present)} cols.")
        except ValueError as e: print(f"Error StandardScaler: {e}"); st_scaler = None
    else: print("Advertencia: No cols para StandardScaler.")

    return mm_scaler, rb_scaler, st_scaler


def apply_scalers(df, mm_scaler_path="scaler_minmax.pkl", rb_scaler_path="scaler_robust.pkl", st_scaler_path="scaler_standard.pkl"):
    """Aplica escaladores cargados desde archivos .pkl. VWAP excluido."""
    df = df.copy()
    try: mm_scaler = load(mm_scaler_path)
    except FileNotFoundError: mm_scaler = None
    try: rb_scaler = load(rb_scaler_path)
    except FileNotFoundError: rb_scaler = None
    try: st_scaler = load(st_scaler_path)
    except FileNotFoundError: st_scaler = None

    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vol_spike_ratio", "bb_width"]

    minmax_cols_present = [col for col in minmax_cols if col in df.columns]
    robust_cols_present = [col for col in robust_cols if col in df.columns]
    standard_cols_present = [col for col in standard_cols if col in df.columns]

    if mm_scaler and minmax_cols_present:
        valid_cols = [col for col in minmax_cols_present if df[col].notna().any()]
        if valid_cols: df[valid_cols] = mm_scaler.transform(df[valid_cols])
    if rb_scaler and robust_cols_present:
        valid_cols = [col for col in robust_cols_present if df[col].notna().any()]
        if valid_cols: df[valid_cols] = rb_scaler.transform(df[valid_cols])
    if st_scaler and standard_cols_present:
        valid_cols = [col for col in standard_cols_present if df[col].notna().any()]
        if valid_cols: df[valid_cols] = st_scaler.transform(df[valid_cols])
        
    return df