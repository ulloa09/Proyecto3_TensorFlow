import pandas as pd
from preprocess_features import fechas, fit_scalers, apply_scalers
from features import generate_features
# *** MODIFIED: No longer need compute_thresholds here ***
from functions import make_forward_return, label_by_thresholds, prepare_xy
from split import split_dfs

# *** REVERTED: Changed args from lower_q/upper_q back to lower/upper ***
def load_and_prepare_data(csv_path: str, horizon: int, lower: float, upper: float, split_ratios: dict):
    """
    Encapsulates data loading, feature generation, labeling, and splitting.
    (Using fixed value thresholds for labeling)

    Args:
        csv_path (str): Path to the raw CSV data.
        horizon (int): Horizon for forward return calculation.
        lower (float): Fixed value threshold for 'Sell' label.
        upper (float): Fixed value threshold for 'Buy' label.
        split_ratios (dict): Dictionary defining the train/test/validation split percentages.
    Returns:
        tuple:
            - train_df (pd.DataFrame): Training data.
            - test_df (pd.DataFrame): Test data.
            - validation_df (pd.DataFrame): Validation data.
    """
    # Load data
    datos = pd.read_csv(csv_path)
    # Create date features
    datos = fechas(datos) # 'fechas' is the function name from preprocess_features
    # Create technical features
    datos = generate_features(datos)
    # Calculate future return (forward return)
    datos = make_forward_return(datos, horizon=horizon)
    
    # *** REVERTED: Call label_by_thresholds directly, no compute_thresholds ***
    datos = label_by_thresholds(datos, lower_thr=lower, upper_thr=upper)
    
    # Drop NAs created by feature generation
    data = datos.copy().dropna()
    
    # Split data
    train_df, test_df, validation_df = split_dfs(
        data, 
        train=split_ratios['train'], 
        test=split_ratios['test'], 
        validation=split_ratios['validation']
    )
    
    return train_df, test_df, validation_df

def scale_data(train_df, test_df, validation_df):
    """
    Fits scalers with train_df and applies them to all three sets.
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        validation_df (pd.DataFrame): Validation data.
    Returns:
        tuple:
            - train_scaled (pd.DataFrame): Scaled training data.
            - test_scaled (pd.DataFrame): Scaled test data.
            - val_scaled (pd.DataFrame): Scaled validation data.
    """
    # Create and fit scalers
    min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler = fit_scalers(train_df)

    # === Apply scalers to all three sets ===
    train_scaled = apply_scalers(train_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    test_scaled = apply_scalers(test_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    val_scaled = apply_scalers(validation_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    print("\nScalers applied to all sets correctly.")

    # Preserve date (necessary for backtesting)
    train_scaled["Date"] = train_df["Date"]
    test_scaled["Date"] = test_df["Date"]
    val_scaled["Date"] = validation_df["Date"]

    
    # === Save results to disk for quick review ===
    train_scaled.to_csv("data/train_scaled.csv", index=False)
    test_scaled.to_csv("data/test_scaled.csv", index=False)
    val_scaled.to_csv("data/val_scaled.csv", index=False)
    print("Scaled data saved to 'data/' directory.")
    
    return train_scaled, test_scaled, val_scaled

def prepare_xy_data(train_scaled, val_scaled, test_scaled):
    """
    Separates the scaled DataFrames into X (features) and y (labels).
    Args:
        train_scaled (pd.DataFrame): Scaled training data.
        val_scaled (pd.DataFrame): Scaled validation data.
        test_scaled (pd.DataFrame): Scaled test data.

    Returns:
        tuple:
            - X_train, X_val, X_test (np.ndarray): Feature arrays.
            - y_train, y_val, y_test (np.ndarray): Label arrays.
            - feature_cols (list): List of feature names.
    """
    # Separation into x, y for train, test, and validation
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy(train_scaled, val_scaled, test_scaled)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols