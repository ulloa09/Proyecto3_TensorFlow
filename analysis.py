import numpy as np
import pandas as pd
from typing import Optional
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


def calculate_dynamic_drift_series(
    train_scaled: pd.DataFrame, 
    test_scaled: pd.DataFrame, 
    val_scaled: pd.DataFrame, 
    feature_cols: list, 
    window_size: int = 90, 
    step_size: int = 30,
    threshold: float = 0.05
) -> pd.Series:
    """
    *** NEW FUNCTION ***
    Calculates the evolution of data drift over time using a sliding window.

    Args:
        train_scaled (pd.DataFrame): Scaled training features.
        test_scaled (pd.DataFrame): Scaled test features.
        val_scaled (pd.DataFrame): Scaled validation features.
        feature_cols (list): List of feature names to compare.
        window_size (int): The number of days for each drift check.
        step_size (int): The number of days to move the window forward.
        threshold (float): The p-value threshold to consider a feature "drifted".

    Returns:
        pd.Series: A time series where the index is the date (end of the window)
                   and the value is the count of drifted features.
    """
    print(f"\n--- Calculating Dynamic Drift (Window={window_size}, Step={step_size}) ---")
    
    # Baseline data (Train)
    baseline_features = train_scaled[feature_cols].copy()
    
    # Monitoring data (Test + Val)
    monitoring_data = pd.concat([
        test_scaled[feature_cols + ['Date']], 
        val_scaled[feature_cols + ['Date']]
    ]).sort_values(by='Date').set_index('Date')
    
    monitoring_features = monitoring_data[feature_cols]

    drift_counts = []
    window_end_dates = []

    # Slide the window over the monitoring data
    for i in range(window_size, len(monitoring_features), step_size):
        start_idx = i - window_size
        end_idx = i
        
        # Get the current window of data
        window_df = monitoring_features.iloc[start_idx:end_idx]
        
        # Get the p-values by comparing the window to the baseline
        p_values = get_feature_pvalues(baseline_features, window_df)
        
        # Count how many features have drifted
        drift_count = sum(1 for p in p_values.values() if p < threshold)
        
        drift_counts.append(drift_count)
        window_end_dates.append(monitoring_features.index[end_idx - 1])

    if not window_end_dates:
        print("Warning: No dynamic drift windows were calculated.")
        return pd.Series(dtype=float)

    # Create the final time series
    drift_series = pd.Series(drift_counts, index=pd.to_datetime(window_end_dates))
    drift_series.name = "Drifted Features Count"
    
    print("--- Dynamic Drift Calculation Complete ---")
    return drift_series


def run_backtest_and_plots(
    best_model, model_name,
    X_train_final, X_test_final, X_val_final,
    train_df, test_df, validation_df, # Original DataFrames
    backtest_params: dict,
    drift_series: Optional[pd.Series] = None  # <-- *** NEW ARGUMENT ***
):
    """
    Runs the final backtest using the winning model and generates all plots.
    
    Args:
        best_model: The trained Keras model.
        model_name (str): The name of the model (e.g., "CNN" or "MLP").
        X_train_final (np.array): Final features for training (2D or 3D).
        X_test_final (np.array): Final features for test.
        X_val_final (np.array): Final features for validation.
        train_df (pd.DataFrame): The *original* (unscaled) training data.
        test_df (pd.DataFrame): The *original* (unscaled) test data.
        validation_df (pd.DataFrame): The *original* (unscaled) validation data.
        backtest_params (dict): Dictionary with 'stop_loss', 'take_profit', 'n_shares'.
        drift_series (Optional[pd.Series]): The calculated dynamic drift series.
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

    # Run backtests
    print("\nRunning Backtest (Train)...")
    cash_train, port_series_train, _, _, _, _ = backtest(
        train_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nRunning Backtest (Test)...")
    cash_test, port_series_test, _, _, _, _ = backtest(
        test_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nRunning Backtest (Validation)...")
    cash_val, port_series_val, _, _, _, _ = backtest(
        val_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    # --- GENERATE GRAPHS ---
    print("\n--- Generating Portfolio Graphs ---")

    # --- Plot 1: Combined Equity Curve with Dynamic Drift ---
    print("Generating Plot 1: Combined Strategy Equity & Dynamic Drift...")
    plot_portfolio_combined(
        port_series_train, 
        port_series_test, 
        port_series_val, 
        model_name,
        drift_series=drift_series  # <-- *** PASSING THE NEW SERIES ***
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