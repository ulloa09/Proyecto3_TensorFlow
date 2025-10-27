from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
from model_training import train_and_select_best_model
# --- MODIFICATION: Import the new dynamic drift calculator ---
from analysis import run_data_drift_analysis, run_backtest_and_plots, calculate_dynamic_drift_series

# --- Flow Configuration Constants ---

# Load and Feature Configuration
DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
FWD_RETURN_HORIZON = 5
lower = -0.1
upper = 0.002
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

# Backtest Configuration
BACKTEST_PARAMS = {
    'stop_loss': 0.2,
    'take_profit': 0.15,
    'n_shares': 30
}

# Dynamic Drift Configuration
DRIFT_WINDOW_SIZE = 90  # Lookback window (approx 4 months)
DRIFT_STEP_SIZE = 21    # Recalculate drift every month

def main():
    """
    Runs the complete trading project flow:
    1. Load and prepare data.
    2. Scale features.
    3. Prepare X/y.
    4. Train and select the best model (CNN vs MLP).
    5. Run STATIC Data Drift analysis (for tables).
    6. Run DYNAMIC Data Drift analysis (for plotting).
    7. Run Backtest and generate plots.
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
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    # --- 4. Train and Select Model ---
    # This function encapsulates the test loops for CNN and MLP,
    # CNN re-shaping, and the final selection logic.
    best_model, model_name, X_train_final, X_test_final, X_val_final = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # --- 5. Run Static Data Drift Analysis (for tables/logs) ---
    run_data_drift_analysis(
        train_scaled, test_scaled, val_scaled, feature_cols
    )

    # --- 6. *** NEW *** Calculate Dynamic Data Drift (for plotting) ---
    # This uses the scaled data to create a time series of drift evolution
    dynamic_drift_series = calculate_dynamic_drift_series(
        train_scaled, 
        test_scaled, 
        val_scaled, 
        feature_cols,
        window_size=DRIFT_WINDOW_SIZE,
        step_size=DRIFT_STEP_SIZE
    )

    # --- 7. Backtest and Plots ---
    # Pass the new dynamic_drift_series to the plotting function
    run_backtest_and_plots(
        best_model, model_name,
        X_train_final, X_test_final, X_val_final,
        train_df, test_df, validation_df, # Use *original* DFs
        BACKTEST_PARAMS,
        drift_series=dynamic_drift_series # <-- *** PASSING THE NEW SERIES ***
    )

if __name__ == "__main__":
    main()