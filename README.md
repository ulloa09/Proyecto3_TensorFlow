# Project 3: Systematic Trading Strategy with Deep Learning

Authors:
- Mauricio Martínez Ulloa
- David Campos Ambriz


## Overview

This project implements a systematic trading strategy for the asset *Wynn Resorts (WYNN)* using Deep Learning models (MLP and CNN). The system performs feature engineering from daily historical data, trains models to predict trading signals (Buy, Sell, Hold), manages experiments with MLFlow, monitors data drift, and evaluates the strategy through robust backtesting with realistic costs.

## Key Features

•⁠  ⁠*Feature Engineering:* Generation of over 20 technical indicators (Momentum, Volatility, Volume).
•⁠  ⁠*Deep Learning Models:* Comparison between Multilayer Perceptron (MLP) and 1D Convolutional Neural Network (CNN).
•⁠  ⁠*MLFlow Tracking:* Logging of experiments, parameters, metrics, and model versioning for reproducibility.
•⁠  ⁠*Data Drift Monitoring:* Static (notebook) and dynamic (during backtest) analysis using the KS-Test.
•⁠  ⁠*Realistic Backtesting:* Trading simulation with commissions (0.125%), borrow rate for shorts (0.25% annual), stop loss, and take profit.
•⁠  ⁠*Walk-Forward Evaluation:* Robust validation of the model on sequential out-of-sample data.
•⁠  ⁠*Centralized Configuration:* Use of ⁠ config.py ⁠ to manage all project parameters.

## Code Workflow (⁠ main.py ⁠)

The main script ⁠ main.py ⁠ orchestrates the entire flow:

1.  *Load and Prepare Data:* Reads parameters from ⁠ config.py ⁠, loads data (⁠ data_pipeline.py ⁠), generates features (⁠ features.py ⁠) and labels (⁠ functions.py ⁠).
2.  *Scale Features:* Applies different scalers (MinMax, Robust, Standard) based on feature type (⁠ preprocess_features.py ⁠).
3.  *Prepare X/y:* Separates data into sets for the model.
4.  *Train or Load Model:*
    * If ⁠ config.TRAIN_NEW_MODEL = True ⁠, trains multiple configurations of MLP and CNN, selects the best based on ⁠ val_loss ⁠, registers it in MLFlow, and saves artifacts (⁠ model_training.py ⁠, ⁠ models.py ⁠).
    * If ⁠ config.TRAIN_NEW_MODEL = False ⁠, loads a specific model (⁠ config.MODEL_VERSION_TO_LOAD ⁠) from the MLFlow registry (⁠ model_training.py ⁠).
5.  *Static Drift Analysis:* Performs an initial comparison of distributions between Train/Test/Val (⁠ analysis.py ⁠).
6.  *Backtesting and Plots:* Executes the backtest using the selected model. Calculates data drift dynamically during the Test and Validation simulation. Generates plots combining the equity curve with the drifted features count and compares the strategy against Buy & Hold (⁠ analysis.py ⁠, ⁠ backtest.py ⁠, ⁠ graphs.py ⁠).

## Setup and Execution

### 1. Clone the Repository

```bash
git clone [https://github.com/ulloa09/Proyecto3_TensorFlow.git](https://github.com/ulloa09/Proyecto3_TensorFlow.git)
cd Proyecto3_TensorFlow

And then install before running the code:
pip install -r requirements.txt

Finally you can run the code by typing in the terminal:
python main.py

