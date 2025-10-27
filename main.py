import config # Import the master configuration file
from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
# Import both the trainer and the new loader function
from model_training import train_and_select_best_model, load_model_from_mlflow
from analysis import run_data_drift_analysis, run_backtest_and_plots

def main():
    """
    Runs the complete trading project flow:
    1. Load and prepare data (using config).
    2. Scale features.
    3. Prepare X/y (y_test is no longer prepared).
    4. === TRAIN or LOAD model based on config.py ===
    5. Run STATIC Data Drift analysis (for tables).
    6. Run Backtest and generate plots (with DYNAMIC drift).
    """
    
    # --- 1. Load and Prepare Data ---
    train_df, test_df, validation_df = load_and_prepare_data()

    # --- 2. Scale Features ---
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    # --- 3. Prepare X/y ---
    X_train, X_val, X_test, y_train, y_val, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    # --- 4. Train or Load Model ---
    if config.TRAIN_NEW_MODEL:
        print("--- Mode: Training New Model ---")
        # This function encapsulates the test loops for CNN and MLP,
        # CNN re-shaping, and the final selection logic.
        best_model, model_name, X_train_final, X_test_final, X_val_final, _ = train_and_select_best_model(
            X_train, X_val, X_test, y_train, y_val
        )
    else:
        print("--- Mode: Loading Pre-trained Model ---")
        # This function loads the model from MLFlow and
        # handles the data reshaping (2D vs 3D) internally.
        best_model, model_name, X_train_final, X_test_final, X_val_final, _ = load_model_from_mlflow(
            X_train, X_val, X_test
        )

    # --- Check if model loading/training failed ---
    if best_model is None:
        print("❌ Failed to train or load model. Exiting.")
        return

    # --- 5. Run Static Data Drift Analysis (for tables/logs) ---
    run_data_drift_analysis(
        train_scaled, test_scaled, val_scaled, feature_cols
    )
    
    # --- 6. Backtest and Plots ---
    run_backtest_and_plots(
        best_model, model_name,
        X_train_final, X_test_final, X_val_final,
        train_df, test_df, validation_df,      # Original DFs
        train_scaled, test_scaled, val_scaled, # Scaled DFs (for drift)
        feature_cols,                          # Feature list (for drift)
        config.BACKTEST_PARAMS,                # Backtest params from config
        config.DYNAMIC_DRIFT_PARAMS            # Dynamic drift params from config
    )

if __name__ == "__main__":
    main()