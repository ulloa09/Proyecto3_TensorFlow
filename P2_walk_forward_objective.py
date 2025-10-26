"""
Strategy Parameter Optimization using Optuna & Walk-Forward Validation

This script finds the optimal strategy parameters (Stop Loss, Take Profit, N Shares)
for the deep learning trading model using Optuna for efficient hyperparameter search.

It works by:
1. Loading and preparing all data (Train/Test/Val).
2. Training the best DL model (MLP or CNN) on the training data.
3. Generating predictions *only* on the training dataset.
4. Defining an 'objective' function for Optuna.
5. Inside the objective, Optuna suggests parameters (SL, TP, N_Shares).
6. These parameters are evaluated using a walk-forward (TimeSeriesSplit)
   cross-validation on the *training set*.
7. The metric optimized is the average Calmar Ratio (higher is better).
8. Finally, it prints the best parameter combination found.

This script should be run *before* main.py. The resulting dictionary
should be copied into the `BACKTEST_PARAMS` variable in `main.py`.
"""

import numpy as np
import pandas as pd
import optuna  # Import Optuna
from sklearn.model_selection import TimeSeriesSplit

# 1. Import project-specific functions
try:
    from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data
    from model_training import train_and_select_best_model
    from backtest import backtest
    from metrics import annualized_calmar # We need the scoring metric
except ImportError:
    print("Error: Make sure all required .py files (data_pipeline, model_training, etc.) are in the same directory.")
    exit()


# --- Project Configuration (copied from main.py) ---
DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
FWD_RETURN_HORIZON = 5
LABEL_LOWER_Q = 0.2
LABEL_UPPER_Q = 0.8
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

# 2. Define Optimization Constants
N_SPLITS_WALK_FORWARD = 5  # Number of folds for walk-forward validation
N_OPTUNA_TRIALS = 50       # Number of parameter combinations for Optuna to try

def evaluate_params_walk_forward(data_with_signals: pd.DataFrame, sl: float, tp: float, shares: int) -> float:
    """
    Evaluates a single set of strategy parameters using TimeSeriesSplit (walk-forward)
    on the provided training dataset.

    Args:
        data_with_signals (pd.DataFrame): The training DataFrame, including
                                           original price data and the
                                           model's 'target' predictions.
        sl (float): The Stop Loss percentage (e.g., 0.3).
        tp (float): The Take Profit percentage (e.g., 0.3).
        shares (int): The number of shares per trade.

    Returns:
        float: The average Calmar Ratio across all walk-forward folds.
               Returns -inf if the metric cannot be calculated.
    """
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_WALK_FORWARD)
    scores = []

    # Iterate over each walk-forward split
    for train_index, test_index in tscv.split(data_with_signals):
        
        test_data_split = data_with_signals.iloc[test_index]
        
        if test_data_split.empty:
            continue

        try:
            cash, port_series, _, _, _, _ = backtest(
                test_data_split, 
                stop_loss=sl, 
                take_profit=tp, 
                n_shares=shares
            )
        except Exception as e:
            # Silently log errors, but don't stop the optimization
            # print(f"  Warning: Backtest failed for split. Error: {e}")
            continue

        if port_series.empty or len(port_series) < 2:
            continue
            
        rets = port_series.pct_change()
        if rets.empty or rets.isnull().all():
            continue
            
        mean_t = rets.mean()
        calmar = annualized_calmar(mean_t, port_series)
        
        if not np.isnan(calmar) and np.isfinite(calmar):
            scores.append(calmar)

    if not scores:
        return -np.inf 
        
    return np.mean(scores)


def find_best_params():
    """
    Main function to run the entire optimization workflow.
    
    1. Loads data and trains the best DL model.
    2. Generates predictions on the training set.
    3. Defines an Optuna objective function.
    4. Runs the Optuna study.
    5. Prints the best parameters found.
    """
    
    print("--- 1. Preparing data and training model (this may take a while) ---")
    
    train_df, test_df, validation_df = load_and_prepare_data(
        csv_path=DATA_CSV_PATH,
        horizon=FWD_RETURN_HORIZON,
        lower_q=LABEL_LOWER_Q,
        upper_q=LABEL_UPPER_Q,
        split_ratios=SPLIT_RATIOS
    )
    
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )
    
    best_model, model_name, X_train_final, _, _ = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print(f"\n--- 2. Best model selected: {model_name} ---")
    print("--- 3. Generating predictions on TRAINING set for optimization ---")
    
    y_pred_train = np.argmax(best_model.predict(X_train_final), axis=1)
    
    train_df_bt = train_df.copy()
    train_df_bt["target"] = y_pred_train 
    
    print("\n--- 4. Starting Optuna Optimization with Walk-Forward Validation ---")

    # --- Define the Optuna Objective Function ---
    # We define it *inside* find_best_params so it has access to train_df_bt
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function. It suggests parameters and
        returns the score to be maximized.
        """
        
        # 3. Define the search space for Optuna
        params = {
            'stop_loss': trial.suggest_float('stop_loss', 0.1, 0.5),
            'take_profit': trial.suggest_float('take_profit', 0.1, 0.5),
            'n_shares': trial.suggest_int('n_shares', 10, 50, step=5)
        }

        # Evaluate this parameter combination
        score = evaluate_params_walk_forward(
            train_df_bt, 
            sl=params['stop_loss'], 
            tp=params['take_profit'], 
            shares=params['n_shares']
        )
        
        return score
    
    # --- Run the Optuna Study ---
    # We want to MAXIMIZE the Calmar Ratio
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    
    print("\n--- Optimization Complete! ---")
    print(f"Best Score (Average Calmar Ratio): {study.best_value:.4f}")
    print("Best parameters found:")
    print(study.best_params)
    print("\nCopy this dictionary into the 'BACKTEST_PARAMS' variable in your main.py file.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Suppress Optuna's trial logging to keep the output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    find_best_params()