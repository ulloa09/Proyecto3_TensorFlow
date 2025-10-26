import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional # Import Optional

def plot_portfolio_train(portfolio_series: pd.Series, model_name: str):
    
    """
    1. Plots the portfolio performance on the TRAIN period.
    
    Args:
        portfolio_series (pd.Series): Time series of portfolio value.
        model_name (str): Name of the model for the title.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Portfolio Value (Train) - Model {model_name}', color='cyan')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_test(
    portfolio_series: pd.Series,
    model_name: str,
    breakpoint_date: Optional[pd.Timestamp] = None # Ensure this argument exists
):
    """
    2. Plots the portfolio performance on the TEST period.
    Optionally includes a vertical line for a drift breakpoint.

    Args:
        portfolio_series (pd.Series): Time series of portfolio value.
        model_name (str): Name of the model for the title.
        breakpoint_date (pd.Timestamp, optional): Date to draw a vertical line.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Portfolio Value (Test) - Model {model_name}', color='orange')

    # Add vertical line if breakpoint_date is provided
    if breakpoint_date is not None:
        plt.axvline(
            x=breakpoint_date,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Drift Point ({breakpoint_date.date()})'
        )
        plt.legend() # Show legend only if there is a line

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_validation(
    portfolio_series: pd.Series,
    model_name: str,
    breakpoint_date: Optional[pd.Timestamp] = None # Ensure this argument exists
):
    """
    3. Plots the portfolio performance on the VALIDATION period.
    Optionally includes a vertical line for a drift breakpoint.

    Args:
        portfolio_series (pd.Series): Time series of portfolio value.
        model_name (str): Name of the model for the title.
        breakpoint_date (pd.Timestamp, optional): Date to draw a vertical line.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Portfolio Value (Validation) - Model {model_name}', color='lightgreen')

    # Add vertical line if breakpoint_date is provided
    if breakpoint_date is not None:
        plt.axvline(
            x=breakpoint_date,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Drift Point ({breakpoint_date.date()})'
        )
        plt.legend() # Show legend only if there is a line

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_combined(port_train: pd.Series, port_test: pd.Series, port_val: pd.Series, model_name: str):
    """
    4. Combines and plots the portfolio performance across all 3 periods.

    Args:
        port_train (pd.Series): Train portfolio time series.
        port_test (pd.Series): Test portfolio time series.
        port_val (pd.Series): Validation portfolio time series.
        model_name (str): Name of the model for the title.
    """
    # Concatenate the series. The DatetimeIndex will handle the sorting.
    combined_portfolio = pd.concat([port_train, port_test, port_val]).sort_index()

    # Remove duplicates (if the initial index repeats)
    combined_portfolio = combined_portfolio[~combined_portfolio.index.duplicated(keep='last')]

    plt.figure(figsize=(14, 7))
    combined_portfolio.plot(title=f'Portfolio Value (Combined) - Model {model_name}', color='blue')

    # Add vertical lines to mark the zones
    if not port_train.empty:
        plt.axvline(port_train.index.min(), color='gray', linestyle='--', label='Start Train')
    if not port_test.empty:
        plt.axvline(port_test.index.min(), color='red', linestyle='--', label='Start Test')
    if not port_val.empty:
        plt.axvline(port_val.index.min(), color='green', linestyle='--', label='Start Validation')

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()