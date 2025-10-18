import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def fechas(df):
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def fit_scalers(train_df):
    """Entrena los tres escaladores con los datos de entrenamiento."""
    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]

    mm_scaler = MinMaxScaler().fit(train_df[minmax_cols])
    rb_scaler = RobustScaler().fit(train_df[robust_cols])
    st_scaler = StandardScaler().fit(train_df[standard_cols])

    # Guardar
    dump(mm_scaler, "scaler_minmax.pkl")
    dump(rb_scaler, "scaler_robust.pkl")
    dump(st_scaler, "scaler_standard.pkl")

    return mm_scaler, rb_scaler, st_scaler


def apply_scalers(df, mm_scaler, rb_scaler, st_scaler):
    """Aplica los escaladores ya entrenados a un DataFrame."""
    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]

    df[minmax_cols] = mm_scaler.transform(df[minmax_cols])
    df[robust_cols] = rb_scaler.transform(df[robust_cols])
    df[standard_cols] = st_scaler.transform(df[standard_cols])
    return df