import numpy as np
import pandas as pd

# Define number of trading days in a year
days = 252

# --- Calculates the annualized Sharpe Ratio ---
# Receives the mean and standard deviation of returns
# and calculates the annualized Sharpe ratio.
# The Sharpe ratio measures risk-adjusted return.
def annualized_sharpe(mean: float, std: float) -> float:
    """
    Calculates the annualized Sharpe ratio.

    Args:
        mean (float): Mean of daily returns.
        std (float): Standard deviation of daily returns.

    Returns:
        float: Annualized Sharpe ratio.
    """
    annual_rets = (mean * days)
    annual_std = std * np.sqrt(days)

    return annual_rets / annual_std if annual_std > 0 else 0

# --- Calculates the Maximum Drawdown from a peak ---
# Receives a time series of values and calculates the maximum percentage drop
# from a previous cumulative high.
def maximum_drawdown(values: pd.Series) -> float:
    """
    Calculates the maximum drawdown (MDD).

    Args:
        values (pd.Series): Time series of portfolio values.

    Returns:
        float: The maximum drawdown percentage (e.g., 0.2 for 20%).
    """
    roll_max = values.cummax()
    drawdown = (roll_max - values) / roll_max
    return drawdown.max()

# --- Calculates the annualized Calmar Ratio ---
# Combines annualized return with maximum drawdown to measure
# the relationship between return and max-drawdown risk.
def annualized_calmar(mean, values) -> float:
    """
    Calculates the annualized Calmar ratio.

    Args:
        mean (float): Mean of daily returns.
        values (pd.Series): Time series of portfolio values (for MDD).

    Returns:
        float: Annualized Calmar ratio.
    """
    annual_rets = (mean * days)
    max_drawdown_val = maximum_drawdown(values)
    return annual_rets / max_drawdown_val if max_drawdown_val != 0 else 0

# --- Calculates the standard deviation of only negative returns ---
# Used to evaluate downside volatility, ignoring gains,
# and is the basis for the Sortino ratio.
def downside_deviation(rets) -> float:
    """
    Calculates the downside deviation (std of negative returns).

    Args:
        rets (pd.Series): Series of daily returns.

    Returns:
        float: Downside deviation.
    """
    negative_rets = rets[rets < 0]
    if negative_rets.empty:
        return 0
    return ((negative_rets ** 2).mean()) ** 0.5

# --- Calculates the annualized Sortino Ratio ---
# Similar to Sharpe, but uses downside deviation for the denominator,
# focusing on the risk of losses rather than total volatility.
def annualized_sortino(mean: float, rets) -> float:
    """
    Calculates the annualized Sortino ratio.

    Args:
        mean (float): Mean of daily returns.
        rets (pd.Series): Series of daily returns (for downside deviation).

    Returns:
        float: Annualized Sortino ratio.
    """
    annual_rets = (mean * days)
    annual_std_down = downside_deviation(rets) * np.sqrt(days)
    return annual_rets / annual_std_down if annual_std_down > 0 else 0

# --- Calculates the win rate ---
# Proportion of positive returns relative to the total number of operations.
# (Note: This function is defined but not used by backtest.py,
# which calculates its own win rate based on closed trades.)
def win_rate(rets: pd.Series) -> float:
    """
    Calculates the win rate based on a series of returns.

    Args:
        rets (pd.Series): Series of returns (one per trade).

    Returns:
        float: Win rate (e.g., 0.55 for 55%).
    """
    total_trades = len(rets)
    if total_trades == 0:
        return 0
    wins = (rets > 0).sum()
    return wins / total_trades