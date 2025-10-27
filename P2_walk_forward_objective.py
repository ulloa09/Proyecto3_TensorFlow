"""
Walk-Forward Model Evaluation

This script performs a robust evaluation of the final, trained deep learning
model using a walk-forward methodology.
It does NOT perform optimization.

The workflow is:
1.  Load and split data (using config.py).
2.  Train the best-performing model on the Train set *only*.
3.  Use the fixed strategy parameters (from config.py).
4.  Apply TimeSeriesSplit to the Evaluation set (Test + Val).
5.  For each fold:
    a. Use the *single trained model* to generate predictions.
    b. Run the backtest on those predictions.
6.  Aggregate metrics and plot the continuous equity curve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import config # Import the master configuration file

# 1. Import project-specific functions
try:
    from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
    from model_training import train_and_select_best_model
    from backtest import backtest
    from metrics import (
        annualized_calmar, 
        annualized_sharpe, 
        annualized_sortino, 
        maximum_drawdown
    )
except ImportError:
    print("Error: Make sure all required .py files (data_pipeline, model_training, etc.) are in the same directory.")
    exit()

# --- Project Configuration (from config.py) ---
# All parameters are now loaded from the config file
BACKTEST_PARAMS = config.BACKTEST_PARAMS
N_SPLITS = config.N_SPLITS_WF

def run_walk_forward_evaluation():
    """
    Executes the entire walk-forward evaluation process.
    """
    
    print("--- 1. Loading and Splitting Data (from config) ---")
    # This function now reads its parameters from config.py
    train_df, test_df, validation_df = load_and_prepare_data()
    
    print("--- 2. Scaling Data ---")
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )
    
    print("--- 3. Preparing X/y Sets ---")
    # y_test is no longer returned
    X_train, X_val, X_test, y_train, y_val, _ = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    print("--- 4. Training Best Model (on 60% Train Set) ---")
    # Train and select the single best model (MLP or CNN)
    # y_test is no longer passed
    best_model, model_name, X_train_final, X_test_final, X_val_final, _ = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val
    )
    
    print(f"\n--- 5. Model Trained: {model_name} ---")
    print("--- 6. Preparing Walk-Forward Evaluation Set (Test + Val) ---")

    # Combine the Test and Validation sets into one continuous block for evaluation
    evaluation_df = pd.concat([test_df, validation_df]).sort_index()
    
    # Combine the X features (must be in the correct 2D or 3D shape)
    X_evaluation_final = np.concatenate((X_test_final, X_val_final), axis=0)
    
    print(f"Evaluation set shape: {evaluation_df.shape}")
    print(f"Evaluation X features shape: {X_evaluation_final.shape}")

    print(f"\n--- 7. Running {N_SPLITS}-Fold Walk-Forward Backtest ---")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    all_fold_metrics = []
    all_portfolio_series = []

    # Loop over each walk-forward fold
    for i, (train_idx, test_idx) in enumerate(tscv.split(evaluation_df)):
        print(f"--- Processing Fold {i+1}/{N_SPLITS} ---")

        # Get the data for this fold's test set
        fold_data_df = evaluation_df.iloc[test_idx]
        fold_data_X = X_evaluation_final[test_idx]

        if fold_data_df.empty:
            print("  Skipping empty fold.")
            continue
            
        print(f"  Fold data size: {len(fold_data_df)} bars")

        # --- Generate Predictions ---
        # Use the *single trained model* to predict on this new fold
        y_pred_fold = np.argmax(best_model.predict(fold_data_X), axis=1)

        # --- Prepare Data for Backtest ---
        fold_backtest_df = fold_data_df.copy()
        fold_backtest_df["target"] = y_pred_fold

        # --- Run Backtest ---
        try:
            # Pass parameters from the config-loaded dictionary
            cash, port_series, _, _, _, _, _ = backtest( # Added one _ for new drift_series return
                fold_backtest_df, 
                stop_loss=BACKTEST_PARAMS['stop_loss'], 
                take_profit=BACKTEST_PARAMS['take_profit'], 
                n_shares=BACKTEST_PARAMS['n_shares']
                # Note: We are not running dynamic drift inside the walk-forward
            )
            
            if port_series.empty or len(port_series) < 2:
                print("  Backtest produced no trades or insufficient data.")
                continue

            # Store the equity curve for this fold
            all_portfolio_series.append(port_series)

            # --- Calculate Metrics for this Fold ---
            rets = port_series.pct_change().dropna()
            if rets.empty:
                print("  Backtest produced no returns.")
                continue

            metrics = {
                'Fold': i + 1,
                'Start Date': port_series.index.min().date(),
                'End Date': port_series.index.max().date(),
                'Calmar Ratio': annualized_calmar(rets.mean(), port_series),
                'Sharpe Ratio': annualized_sharpe(rets.mean(), rets.std()),
                'Sortino Ratio': annualized_sortino(rets.mean(), rets),
                'Max Drawdown': maximum_drawdown(port_series)
            }
            all_fold_metrics.append(metrics)
            
        except Exception as e:
            print(f"  Error during backtest for fold {i+1}: {e}")

    print("\n--- 8. Aggregating Walk-Forward Results ---")

    # --- Display Metrics ---
    if not all_fold_metrics:
        print("No valid metrics were calculated. Exiting.")
        return

    metrics_df = pd.DataFrame(all_fold_metrics).set_index('Fold')
    
    print("\n--- Performance per Fold ---")
    print(metrics_df.to_string(float_format="%.4f"))
    
    print("\n--- Average Performance (All Folds) ---")
    print(metrics_df.mean(numeric_only=True).to_string(float_format="%.4f"))

    # --- Plot Combined Equity Curve ---
    if not all_portfolio_series:
        print("No portfolio series to plot.")
        return

    combined_equity_curve = pd.concat(all_portfolio_series).sort_index()
    # Remove duplicates that can occur at the seams of the folds
    combined_equity_curve = combined_equity_curve[~combined_equity_curve.index.duplicated(keep='last')]

    print("\nPlotting combined walk-forward equity curve...")
    
    plt.figure(figsize=(14, 7))
    combined_equity_curve.plot(
        title=f"Walk-Forward Evaluation ({model_name} on Test+Val Sets)",
        color='blue'
    )
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Script Entry Point ---
if __name__ == "__main__":
    run_walk_forward_evaluation()