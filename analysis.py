import numpy as np
import pandas as pd
from data_drift import get_feature_pvalues, get_feature_shift_status
from backtest import backtest
from graphs import (
    plot_portfolio_train,
    plot_portfolio_test,
    plot_portfolio_validation,
    plot_portfolio_combined
)

def run_data_drift_analysis(train_scaled, test_scaled, val_scaled, feature_cols):
    """
    Runs the stability analysis (Data Drift) comparing
    Test vs. Train and Validation vs. Train.

    Args:
        train_scaled (pd.DataFrame): Scaled training data.
        test_scaled (pd.DataFrame): Scaled test data.
        val_scaled (pd.DataFrame): Scaled validation data.
        feature_cols (list): List of feature names to compare.
    """
    # --- STABILITY ANALYSIS (DATA DRIFT) ---
    print("\n--- Starting Feature Stability Analysis ---")

    # Use scaled features (without labels) for the comparison
    baseline_data_features = train_scaled[feature_cols]
    test_data_features = test_scaled[feature_cols]
    val_data_features = val_scaled[feature_cols]

    # 1. Compare Test vs Train
    print("\nComparing Test vs. Train...")
    p_values_test = get_feature_pvalues(baseline_data_features, test_data_features)
    shift_status_test = get_feature_shift_status(baseline_data_features, test_data_features, threshold=0.05)

    drift_df_test = pd.DataFrame(list(p_values_test.items()), columns=['Feature', 'p_value'])
    drift_df_test['Drift_Detected'] = drift_df_test['Feature'].map(shift_status_test).fillna(False)
    drift_df_test = drift_df_test.sort_values(by='p_value', ascending=True)

    print(f"Features with shift detected (Test): {drift_df_test['Drift_Detected'].sum()}")
    if drift_df_test['Drift_Detected'].sum() > 0:
        print(drift_df_test[drift_df_test['Drift_Detected'] == True].to_string())

    
    # 2. Compare Validation vs Train
    print("\nComparing Validation vs. Train...")
    p_values_val = get_feature_pvalues(baseline_data_features, val_data_features)
    shift_status_val = get_feature_shift_status(baseline_data_features, val_data_features, threshold=0.05)

    drift_df_val = pd.DataFrame(list(p_values_val.items()), columns=['Feature', 'p_value'])
    drift_df_val['Drift_Detected'] = drift_df_val['Feature'].map(shift_status_val).fillna(False)
    drift_df_val = drift_df_val.sort_values(by='p_value', ascending=True)

    print(f"Features with shift detected (Validation): {drift_df_val['Drift_Detected'].sum()}")
    if drift_df_val['Drift_Detected'].sum() > 0:
        print(drift_df_val[drift_df_val['Drift_Detected'] == True].to_string())
    print("--- Stability Analysis Completed ---")


def run_backtest_and_plots(
    best_model, model_name,
    X_train_final, X_test_final, X_val_final,
    train_df, 
    test_df, validation_df, # *Original* DataFrames
    backtest_params: dict
):
    """
    Runs the final backtest using the winning model and generates the plots.

    Args:
        best_model (tf.keras.Model): The trained and selected model.
        model_name (str): Name of the model for plot titles.
        X_train_final (np.ndarray): Final input data for training set (correct shape).
        X_test_final (np.ndarray): Final input data for test set.
        X_val_final (np.ndarray): Final input data for validation set.
        train_df (pd.DataFrame): Original, *unscaled* training data.
        test_df (pd.DataFrame): Original, *unscaled* test data.
        validation_df (pd.DataFrame): Original, *unscaled* validation data.
        backtest_params (dict): Parameters for the backtest (SL, TP, n_shares).
    """
    # --- BACKTEST EXECUTION AND PLOTS ---
    print(f"\n--- Starting Backtest (Winning Model: {model_name}) ---")

    # Fixed backtest parameters
    STOP_LOSS = backtest_params['stop_loss']
    TAKE_PROFIT = backtest_params['take_profit']
    N_SHARES = backtest_params['n_shares']

    # Generate predictions for all 3 sets
    y_pred_train = np.argmax(best_model.predict(X_train_final), axis=1)
    y_pred_test = np.argmax(best_model.predict(X_test_final), axis=1)
    y_pred_val = np.argmax(best_model.predict(X_val_final), axis=1)

    # Assign predictions (use copies to avoid SettingWithCopyWarning)
    # We use the original (unscaled) DataFrames for the backtest
    train_df_bt = train_df.copy()
    test_df_bt = test_df.copy()
    val_df_bt = validation_df.copy()

    train_df_bt["target"] = y_pred_train
    test_df_bt["target"] = y_pred_test
    val_df_bt["target"] = y_pred_val

    # Run backtests and capture the portfolio SERIES (port_series_...)
    print("\nRunning Backtest (Train)...")
    cash_train, port_series_train, buy_train, sell_train, hold_train, total_ops_train = backtest(
        train_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nRunning Backtest (Test)...")
    cash_test, port_series_test, buy_test, sell_test, hold_test, total_ops_test = backtest(
        test_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    print("\nRunning Backtest (Validation)...")
    cash_val, port_series_val, buy_val, sell_val, hold_val, total_ops_val = backtest(
        val_df_bt, stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, n_shares=N_SHARES
    )

    # --- PLOT GENERATION ---
    print("\n--- Generating Portfolio Plots ---")

    # 1. Training Plot
    plot_portfolio_train(port_series_train, model_name)

    # 2. Test Plot
    plot_portfolio_test(port_series_test, model_name) 

    # 3. Validation Plot
    plot_portfolio_validation(port_series_val, model_name)

    # 4. Combined Plot
    plot_portfolio_combined(port_series_train, port_series_test, port_series_val, model_name)

    print("--- Execution Finished ---")