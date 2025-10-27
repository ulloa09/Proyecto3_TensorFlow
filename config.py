"""
================================================
Project Configuration File (config.py)
================================================

This file serves as the single source of truth for all project-wide
constants, parameters, and settings.

By centralizing configuration, we ensure that modules like `main.py`,
`model_training.py`, and analysis notebooks (e.g., `dirft_report.ipynb`)
all use the exact same parameters, preventing redundancy and making
updates simple.

Sections:
1.  Data Configuration: Settings for data source and paths.
2.  Feature & Labeling: Parameters for feature engineering and target definition.
3.  Model Training: Hyperparameter search spaces and MLFlow settings.
4.  Backtest & Analysis: Parameters for strategy simulation and drift analysis.
"""

# 1. Data Configuration
# ------------------------------------------------
TICKER = "WYNN"
START_DATE = "2010-01-01"
END_DATE = "2025-10-17" # yfinance 'end' is exclusive
DATA_CSV_PATH = f'data/{TICKER.lower()}_daily_15y.csv'
SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}


# 2. Feature & Labeling Configuration
# ------------------------------------------------
FWD_RETURN_HORIZON = 5
LABEL_LOWER = -0.1
LABEL_UPPER = 0.002


# 3. Model Training Configuration
# ------------------------------------------------

# --- MLFlow Settings ---
MLFLOW_EXPERIMENT_NAME = "Proyecto3_TensorFlow"
BEST_MODEL_NAME = "SystematicTradingModel" # Registered model name in MLFlow

# --- CNN Hyperparameter Search Space ---
params_space_cnn = [
    {"num_filters": 32, "kernel_size": 1, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 64},
    {"num_filters": 32, "kernel_size": 3, "conv_blocks": 3, "dense_units": 64, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 64},
    {"num_filters": 32, "kernel_size": 4, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.15, "optimizer": "adam", "epochs": 40, "batch_size": 32}
]

# --- MLP Hyperparameter Search Space ---
mlp_param_space = [
    {"dense_blocks": 4, "dense_units": 128, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
    {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
    {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 64},
]


# 4. Backtest & Analysis Configuration
# ------------------------------------------------

# --- Backtest Strategy Parameters ---
BACKTEST_PARAMS = {
    'stop_loss': 0.3,
    'take_profit': 0.3,
    'n_shares': 30
}

# --- Dynamic Drift Parameters ---
DYNAMIC_DRIFT_PARAMS = {
    'window_size': 21,  # Lookback window (1 months)
    'step_size': 21     # Recalculate drift every month
}

# --- Walk-Forward Evaluation ---
N_SPLITS_WF = 5 # Number of folds for the walk-forward evaluation


# 5. Execution Mode
# ------------------------------------------------
# Set to True to run the full training pipeline in main.py
# Set to False to load an existing model from MLFlow for analysis in main.py
TRAIN_NEW_MODEL = False

# Specify the version to load if TRAIN_NEW_MODEL is False
# This must be a string (e.g., "1", "5", "latest")
MODEL_VERSION_TO_LOAD = "1"