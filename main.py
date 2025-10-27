import mlflow
import tensorflow as tf
from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
# We no longer import the full training function, just the reshaper
from cnn_model import reshape_cnn 
# We import the drift analysis and the backtest/plotting
from analysis import run_data_drift_analysis, run_backtest_and_plots

# --- Flow Configuration Constants ---

# Load and Feature Configuration
DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
FWD_RETURN_HORIZON = 5
lower = -0.1
upper = 0.002
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

# Backtest Configuration
BACKTEST_PARAMS = {
    'stop_loss': 0.3,
    'take_profit': 0.3,
    'n_shares': 30
}

# Dynamic Drift Configuration
DRIFT_PARAMS = {
    'window_size': 90,  # Lookback window (approx 4 months)
    'step_size': 21     # Recalculate drift every month
}

# --- MLFlow Model Loading ---
# This must match the name used in model_training.py
BEST_MODEL_NAME = "SystematicTradingModel"

def load_model_from_registry(model_name: str, model_version_str: str = "latest") -> (tf.keras.Model, str):
    """
    Loads a model and its 'model_type' tag from the MLFlow Model Registry.

    Args:
        model_name (str): The registered name of the model.
        model_version_str (str): The version string (e.g., "latest" or "5").

    Returns:
        tuple:
            - (tf.keras.Model): The loaded model object.
            - (str): The 'model_type' tag (e.g., "CNN1D..." or "MLP...").
    """
    print(f"\n--- Loading model '{model_name}' (Version: {model_version_str}) from MLFlow ---")
    client = mlflow.tracking.MlflowClient()
    try:
        # 1. Load the model from the registry URI
        model_uri = f"models:/{model_name}/{model_version_str}"
        model = mlflow.tensorflow.load_model(model_uri)
        print("✅ Model loaded successfully.")
        
        # 2. Get the model version object to read its tags
        # We need to resolve "latest" alias to a specific version number if used
        if model_version_str == "latest":
            mv = client.get_model_version_by_alias(model_name, model_version_str)
        else:
            mv = client.get_model_version(model_name, model_version_str)

        # 3. Retrieve the "model_type" tag we set during training
        model_type_tag = mv.tags.get("model_type", "Loaded Model")
        print(f"   Model Type: {model_type_tag}")
        
        return model, model_type_tag
        
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        print("Please ensure MLFlow server is running and the model was trained and registered.")
        return None, None


def main():
    """
    Runs the complete trading project flow:
    1. Load and prepare data.
    2. Scale features.
    3. Prepare X/y.
    4. Load the best pre-trained model from MLFlow.
    5. Reshape data based on loaded model type (CNN or MLP).
    6. Run STATIC Data Drift analysis (for tables).
    7. Run Backtest (with DYNAMIC drift) and generate plots.
    """
    
    # --- 1. Load and Prepare Data ---
    # (Saves original DataFrames for the final backtest)
    train_df, test_df, validation_df = load_and_prepare_data(
        csv_path=DATA_CSV_PATH,
        horizon=FWD_RETURN_HORIZON,
        lower=lower,
        upper=upper,
        split_ratios=SPLIT_RATIOS
    )

    # --- 2. Scale Features ---
    # (Saves scaled DataFrames for drift analysis)
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    # --- 3. Prepare X/y ---
    # These are the 2D base arrays
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    # --- 4. Load Best Model ---
    # This step replaces the training block
    best_model, model_name = load_model_from_registry(BEST_MODEL_NAME, "latest")
    
    # Exit if model failed to load
    if best_model is None:
        return

    # --- 5. Prepare Data Shape for Model ---
    # We must reshape the X data to 3D if the loaded model is a CNN
    if "CNN" in model_name:
        print("Model is a CNN, reshaping data to 3D.")
        X_train_final, X_test_final, X_val_final = reshape_cnn(X_train, X_test, X_val)
    else:
        print("Model is an MLP, using 2D data.")
        X_train_final = X_train
        X_test_final = X_test
        X_val_final = X_val

    # --- 6. Run Static Data Drift Analysis (for tables/logs) ---
    run_data_drift_analysis(
        train_scaled, test_scaled, val_scaled, feature_cols
    )

    # --- 7. Backtest and Plots (with Dynamic Drift) ---
    # The dynamic drift calculation is now handled inside run_backtest_and_plots
    run_backtest_and_plots(
        best_model, model_name,
        X_train_final, X_test_final, X_val_final,
        train_df, test_df, validation_df, # Use *original* DFs for backtest
        train_scaled, test_scaled, val_scaled, # Use *scaled* DFs for drift
        feature_cols,
        BACKTEST_PARAMS,
        DRIFT_PARAMS # Pass the drift config
    )

if __name__ == "__main__":
    main()