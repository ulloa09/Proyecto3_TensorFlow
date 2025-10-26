from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
from model_training import train_and_select_best_model
from analysis import run_data_drift_analysis, run_backtest_and_plots

# --- Workflow Configuration Constants ---

# Load and Feature Configuration
DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
FWD_RETURN_HORIZON = 5  # How many days in the future to calculate returns for labeling
LABEL_LOWER_Q = 0.2     # Quantile for 'Sell' (0) labels
LABEL_UPPER_Q = 0.8     # Quantile for 'Buy' (2) labels
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

# Backtest Configuration
BACKTEST_PARAMS = {
    'stop_loss': 0.3,
    'take_profit': 0.3,
    'n_shares': 30
}

def main():
    """
    Executes the complete trading project pipeline:
    1. Loads and prepares raw data (features, labels).
    2. Scales features using scalers fitted on training data.
    3. Prepares X (features) and y (labels) for modeling.
    4. Trains multiple CNN and MLP configurations and selects the best model.
    5. Runs a Data Drift analysis (Train vs. Test, Train vs. Validation).
    6. Executes the final backtest using the best model and generates performance plots.
    """
    
    # --- 1. Load and Prepare Data ---
    # (Saves the original, unscaled DataFrames for the final backtest)
    print("--- 1. Loading and Preparing Data ---")
    train_df, test_df, validation_df = load_and_prepare_data(
        csv_path=DATA_CSV_PATH,
        horizon=FWD_RETURN_HORIZON,
        lower_q=LABEL_LOWER_Q,
        upper_q=LABEL_UPPER_Q,
        split_ratis=SPLIT_RATIOS
    )

    # --- 2. Scale Features ---
    # (Saves the scaled DataFrames for drift analysis and model input)
    print("\n--- 2. Scaling Features ---")
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    # --- 3. Prepare X/y ---
    print("\n--- 3. Preparing X/y Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    # --- 4. Train and Select Model ---
    # This function encapsulates the test loops for CNN and MLP,
    # the CNN reshaping, and the final selection logic.
    print("\n--- 4. Training and Selecting Best Model ---")
    best_model, model_name, X_train_final, X_test_final, X_val_final = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # --- 5. Data Drift Analysis ---
    print("\n--- 5. Running Data Drift Analysis ---")
    run_data_drift_analysis(
        train_scaled, test_scaled, val_scaled, feature_cols
    )

    # --- 6. Backtest and Plots ---
    print("\n--- 6. Running Backtest and Generating Plots ---")
    run_backtest_and_plots(
        best_model, model_name,
        X_train_final, X_test_final, X_val_final,
        train_df, test_df, validation_df, # Use *original* DFs for backtest
        BACKTEST_PARAMS
    )
    
    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    main()