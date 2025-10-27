import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from operation_class import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price: float, n_shares: int)-> float:
    """
    Calculates the total current value of the portfolio.
    Value = cash + value of open long positions + value of open short positions.
    Args:
        cash (float): Current available cash.
        long_ops (list[Operation]): List of active long operations.
        short_ops (list[Operation]): List of active short operations.
        current_price (float): The current market price of the asset.
        n_shares (int): Number of shares per operation (used for shorts).
    Returns:
        float: Total portfolio value.
    """
   
    val = cash

    # Add value of long positions
    for position in long_ops:
        pnl = current_price * position.n_shares # Current value
        val += pnl

    # Add value of short positions
    # Value = (entry_price - current_price) * n_shares
    for position in short_ops:
        pnl = (position.price - current_price) * position.n_shares
        val += pnl

    return val

def make_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Adds 'fwd_ret' = Close.shift(-horizon)/Close - 1
    and trims the last 'horizon' rows (which will be NaN).
    Args:
        df (pd.DataFrame): DataFrame with 'Close' column.
        horizon (int): Number of periods to look ahead.

    Returns:
        pd.DataFrame: DataFrame with 'fwd_ret' column and trimmed end.
    """
    df = df.copy()
    # Calculate future return
    df["fwd_ret"] = df["Close"].shift(-horizon) / df["Close"] - 1
    # Drop rows where fwd_ret cannot be calculated
    if horizon > 0:
        df = df.iloc[:-horizon]
    return df

def compute_thresholds(ref_df: pd.DataFrame, lower_q: float, upper_q: float) -> tuple[float, float]:
    """
    Calculates thresholds (lower_thr, upper_thr) from ref_df['fwd_ret'] (ignores NaN).
    Args:
        ref_df (pd.DataFrame): The reference DataFrame (ideally, the training set).
        lower_q (float): Lower quantile (e.g., 0.2).
        upper_q (float): Upper quantile (e.g., 0.8).
    Returns:
        tuple[float, float]: lower_threshold, upper_threshold.
    """
    ref_df = ref_df.copy()
    if "fwd_ret" not in ref_df.columns:
        raise KeyError("compute_thresholds requires the 'fwd_ret' column in ref_df.")
    
    # Calculate quantiles on non-NaN forward returns
    upper_thr = ref_df["fwd_ret"].dropna().quantile(upper_q)
    lower_thr = ref_df["fwd_ret"].dropna().quantile(lower_q)
    return lower_thr, upper_thr

def label_by_thresholds(df: pd.DataFrame, lower_thr: float, upper_thr: float) -> pd.DataFrame:
    """
    Labels data (0=Sell, 1=Hold, 2=Buy) using fixed thresholds.
    Args:
        df (pd.DataFrame): DataFrame with 'fwd_ret' column.
        lower_thr (float): The 'Sell' threshold.
        upper_thr (float): The 'Buy' threshold.

    Returns:
        pd.DataFrame: DataFrame with 'target' column.
    """
    df = df.copy()
    if "fwd_ret" not in df.columns:
        raise KeyError("label_by_thresholds requires the 'fwd_ret' column in df.")
    
    # Default label is 1 (Hold)
    df["target"] = 1
    # Label 2 (Buy) if return is above the upper threshold
    df.loc[df["fwd_ret"] > upper_thr, "target"] = 2
    # Label 0 (Sell) if return is below the lower threshold
    df.loc[df["fwd_ret"] < lower_thr, "target"] = 0

    print(f"Labels generated: {len(df)}")
    print(f"Class distribution:\n{np.round(df.target.value_counts(normalize=True), 5)}")
    return df

def prepare_xy(train_df, val_df, test_df, exclude_cols=None):
    """
    Prepares X (features) and y (integer labels) for
    training, validation, and testing.
    Adapted for models using sparse_categorical_crossentropy.
    
    (y_test is no longer returned as it was found to be unused upstream).

    Args:
        train_df (pd.DataFrame): Scaled training data.
        val_df (pd.DataFrame): Scaled validation data.
        test_df (pd.DataFrame): Scaled test data.
        exclude_cols (list, optional): Columns to exclude from features.
    Returns:
        tuple:
            - X_train, X_val, X_test (np.ndarray): Feature arrays (float32).
            - y_train, y_val (np.ndarray): Label arrays (int).
            - feature_cols (list): List of feature names.
    """

    if exclude_cols is None:
        # Columns that are not features
        exclude_cols = ["Date", "fwd_ret", "target"]

    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Features to float32 (efficient for models)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)

    # Labels as integers (not one-hot)
    y_train = train_df["target"].astype(int).to_numpy()
    y_val   = val_df["target"].astype(int).to_numpy()
    # y_test  = test_df["target"].astype(int).to_numpy() # This was unused

    print("Data shapes:")
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_val:", X_val.shape, "| y_val:", y_val.shape)
    print("X_test:", X_test.shape) # | y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, feature_cols


def compute_class_weights(y_train):
    """
    Calculates balanced weights for classes in integer format (0, 1, 2).
    Used to handle imbalanced datasets during model training.

    Args:
        y_train (np.ndarray): Array of training labels.
    Returns:
        dict: Dictionary mapping class index to weight (e.g., {0: 1.2, 1: 0.8, 2: 1.3}).
    """
    classes = np.unique(y_train)
    # Calculate weights using sklearn's utility
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    # Format as a dictionary for Keras
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print("Class weights calculated:")
    for k, v in class_weights.items():
        print(f"  Class {k}: {v:.3f}")
    return class_weights