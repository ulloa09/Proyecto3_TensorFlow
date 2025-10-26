import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def fechas(df):
    """
    Converts the 'Date' column to datetime objects.
    (Keeping original function name 'fechas' to match imports).

    Args:
        df (pd.DataFrame): DataFrame with a 'Date' column.

    Returns:
        pd.DataFrame: DataFrame with 'Date' column as datetime.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def fit_scalers(train_df):
    """
    Fits three types of scalers (MinMax, Robust, Standard)
    to the training data based on feature characteristics.

    Args:
        train_df (pd.DataFrame): The training dataset.

    Returns:
        tuple: Fitted scaler objects (mm_scaler, rb_scaler, st_scaler, ohlcv_scaler).
    """
    train_df = train_df.copy()

    # Define columns for each scaling strategy
    # Oscillators (0-100 or -100-100)
    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    # Features prone to outliers (e.g., ATR)
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    # Features with Gaussian-like distribution or unbounded
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",            
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]
    # Base OHLCV data
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

    # Fit scalers *only* on training data
    mm_scaler = MinMaxScaler().fit(train_df[minmax_cols])
    rb_scaler = RobustScaler().fit(train_df[robust_cols])
    st_scaler = StandardScaler().fit(train_df[standard_cols])
    ohlcv_scaler = StandardScaler().fit(train_df[ohlcv_cols])

    # Save fitted scalers to disk
    dump(mm_scaler, "scaler_minmax.pkl")
    dump(rb_scaler, "scaler_robust.pkl")
    dump(st_scaler, "scaler_standard.pkl")
    dump(ohlcv_scaler, "scaler_ohlcv.pkl")

    return mm_scaler, rb_scaler, st_scaler, ohlcv_scaler


def apply_scalers(df, mm_scaler, rb_scaler, st_scaler, ohlcv_scaler):
    """
    Applies the already-fitted scalers to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to transform (train, test, or val).
        mm_scaler (MinMaxScaler): Fitted MinMaxScaler.
        rb_scaler (RobustScaler): Fitted RobustScaler.
        st_scaler (StandardScaler): Fitted StandardScaler.
        ohlcv_scaler (StandardScaler): Fitted StandardScaler for OHLCV.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df = df.copy()
    
    # Define columns (must match those in fit_scalers)
    minmax_cols = ["rsi_14", "rsi_28", "stoch_k_14", "stoch_k_28", "williams_r_14", "bb_percent_b"]
    robust_cols = ["atr_14", "atr_28", "donchian_width", "std_20", "tr_norm"]
    standard_cols = ["macd_12_26", "macd_signal_12_26", "macd_19_39", "macd_signal_19_39",
                     "adx_14", "adx_28", "momentum_10", "roc_20",
                     "obv", "mfi_14", "cmf_20", "vol_sma_20", "vol_zscore_20",
                     "vwap_dev", "vol_spike_ratio", "bb_width"]
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

    # Apply transformations
    df[minmax_cols] = mm_scaler.transform(df[minmax_cols])
    df[robust_cols] = rb_scaler.transform(df[robust_cols])
    df[standard_cols] = st_scaler.transform(df[standard_cols])
    df[ohlcv_cols] = ohlcv_scaler.transform(df[ohlcv_cols])
    
    return df