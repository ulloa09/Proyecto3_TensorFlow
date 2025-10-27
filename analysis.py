import numpy as np
import pandas as pd
from typing import Optional, Dict
from data_drift import get_feature_pvalues, get_feature_shift_status
from backtest import backtest
from graphs import (
    plot_portfolio_combined,
    plot_comparison_with_buy_and_hold
)

def run_data_drift_analysis(train_scaled, test_scaled, val_scaled, feature_cols):
    """
    Runs the STATIC stability analysis (Data Drift) comparing
    Test vs. Train and Validation vs. Train.
    (This is still useful for the executive summary table).
    
    Args:
        train_scaled (pd.DataFrame): Scaled training features.
        test_scaled (pd.DataFrame): Scaled test features.
        val_scaled (pd.DataFrame): Scaled validation features.
        feature_cols (list): List of feature names to compare.
    """
    # --- STATIC STABILITY ANALYSIS (DATA DRIFT) ---
    print("\n--- Starting Static Feature Stability Analysis ---")

    baseline_data_features = train_scaled[feature_cols]
    test_data_features = test_scaled[feature_cols]
    val_data_features = val_scaled[feature_cols]

    # 1. Compare Test vs. Train
    print("\nComparing Test vs. Train...")
    p_values_test = get_feature_pvalues(baseline_data_features, test_data_features)
    shift_status_test = get_feature_shift_status(baseline_data_features, test_data_features, threshold=0.05)
    drift_df_test = pd.DataFrame(list(p_values_test.items()), columns=['Feature', 'p_value'])
    drift_df_test['Drift_Detected'] = drift_df_test['Feature'].map(shift_status_test).fillna(False)
    print(f"Features with detected shift (Test): {drift_df_test['Drift_Detected'].sum()}")

    # 2. Compare Validation vs. Train
    print("\nComparing Validation vs. Train...")
    p_values_val = get_feature_pvalues(baseline_data_features, val_data_features)
    shift_status_val = get_feature_shift_status(baseline_data_features, val_data_features, threshold=0.05)
    drift_df_val = pd.DataFrame(list(p_values_val.items()), columns=['Feature', 'p_value'])
    drift_df_val['Drift_Detected'] = drift_df_val['Feature'].map(shift_status_val).fillna(False)
    print(f"Features with detected shift (Validation): {drift_df_val['Drift_Detected'].sum()}")
    print("--- Static Stability Analysis Complete ---")

def run_backtest_and_plots(
    best_model, model_name,
    X_train_final, X_test_final, X_val_final,
    train_df, test_df, validation_df, # Original DataFrames
    train_scaled, test_scaled, val_scaled, # Scaled DataFrames (for drift)
    feature_cols, # Feature list (for drift)
    backtest_params: dict,
    drift_params: dict # Dynamic drift config
):
    """
    Runs the final backtest using the winning model and generates all plots.
    Dynamic drift is now calculated *inside* the backtest calls for test and val.
    
    Args:
        best_model: The trained Keras model.
        model_name (str): The name of the model (e.g., "CNN" or "MLP").
        X_train_final (np.array): Final features for training (2D or 3D).
        X_test_final (np.array): Final features for test.
        X_val_final (np.array): Final features for validation.
        train_df (pd.DataFrame): The *original* (unscaled) training data.
        test_df (pd.DataFrame): The *original* (unscaled) test data.
        validation_df (pd.DataFrame): The *original* (unscaled) validation data.
        train_scaled (pd.DataFrame): Scaled training features (for drift baseline).
        test_scaled (pd.DataFrame): Scaled test features (for drift monitoring).
        val_scaled (pd.DataFrame): Scaled validation features (for drift monitoring).
        feature_cols (list): List of feature names to compare.
        backtest_params (dict): Dictionary with 'stop_loss', 'take_profit', 'n_shares'.
        drift_params (dict): Dictionary with 'window_size', 'step_size'.
    """
    # --- RUN BACKTEST AND GENERATE PLOTS ---
    print(f"\n--- Starting Backtest (Winning Model: {model_name}) ---")

    STOP_LOSS = backtest_params['stop_loss']
    TAKE_PROFIT = backtest_params['take_profit']
    N_SHARES = backtest_params['n_shares']

    # Generate predictions
    y_pred_train = np.argmax(best_model.predict(X_train_final), axis=1)
    y_pred_test = np.argmax(best_model.predict(X_test_final), axis=1)
    y_pred_val = np.argmax(best_model.predict(X_val_final), axis=1)

    # Assign predictions
    train_df_bt = train_df.copy()
    test_df_bt = test_df.copy()
    val_df_bt = validation_df.copy()

    train_df_bt["target"] = y_pred_train
    test_df_bt["target"] = y_pred_test
    val_df_bt["target"] = y_pred_val

    # --- Prepare for Drift Calculation ---
    # Set the baseline for drift comparison (Training features)
    baseline_features = train_scaled[feature_cols].copy()
    
    # Prepare the feature sets to be monitored
    # We reset the index to ensure alignment with the backtest iterator (0 to N)
    test_features = test_scaled[feature_cols].copy().reset_index(drop=True)
    val_features = val_scaled[feature_cols].copy().reset_index(drop=True)

    # Prepare drift parameters for the backtest function
    drift_bt_params = {
        'drift_window': drift_params['window_size'],
        'drift_step': drift_params['step_size'],
        'drift_threshold': 0.05 # Standard significance level
    }

    # Run backtests
    print("\nRunning Backtest (Train)...")
    # We don't run drift analysis on the training set
    cash_train, port_series_train, _, _, _, _, drift_series_train = backtest(
        train_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nRunning Backtest (Test) with Dynamic Drift...")
    cash_test, port_series_test, _, _, _, _, drift_series_test = backtest(
        test_df_bt, 
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES,
        baseline_features=baseline_features,
        monitoring_features=test_features,
        **drift_bt_params
    )

    print("\nRunning Backtest (Validation) with Dynamic Drift...")
    cash_val, port_series_val, _, _, _, _, drift_series_val = backtest(
        val_df_bt, 
        stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES,
        baseline_features=baseline_features,
        monitoring_features=val_features,
        **drift_bt_params
    )

    # --- Combine dynamic drift results for plotting ---
    # We concatenate the drift series from test and validation
    combined_drift_series = pd.concat([drift_series_test, drift_series_val]).sort_index()

    # --- GENERATE GRAPHS ---
    print("\n--- Generating Portfolio Graphs ---")

    # --- Plot 1: Combined Equity Curve with Dynamic Drift ---
    print("Generating Plot 1: Combined Strategy Equity & Dynamic Drift...")
    plot_portfolio_combined(
        port_series_train, 
        port_series_test, 
        port_series_val, 
        model_name,
        drift_series=combined_drift_series # Pass the combined series
    ) 

    # --- Plot 2: Strategy vs. Buy & Hold (Period-Normalized) ---
    print("Generating Plot 2: Strategy vs. Buy & Hold...")
    plot_comparison_with_buy_and_hold(
        port_series_train,
        port_series_test,
        port_series_val,
        train_df,  # Pass original dfs for B&H calculation
        test_df,
        validation_df,
        model_name
    )

    print("--- Execution Finished ---")