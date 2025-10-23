import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def annualized_sharpe(mean: float, std: float) -> float:
    """Calcula el Sharpe ratio anualizado para retornos diarios."""
    annual_rets = (mean * TRADING_DAYS_PER_YEAR)
    annual_std = std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annual_rets / annual_std if annual_std > 0 else 0

def maximum_drawdown(values: pd.Series) -> float:
    """Calcula la Máxima Pérdida desde un pico (Drawdown máximo)."""
    roll_max = values.cummax()
    max_drawdown = (roll_max - values) / roll_max
    return max_drawdown.max()

def annualized_calmar(mean, values) -> float:
    """Calcula el Índice de Calmar anualizado para retornos diarios."""
    annual_rets = (mean * TRADING_DAYS_PER_YEAR)
    max_drawdown = maximum_drawdown(values)
    return annual_rets / max_drawdown if max_drawdown != 0 else 0

def downside_deviation(rets) -> float:
    """Calcula la desviación estándar sólo de los retornos negativos."""
    negative_rets = rets[rets < 0]
    return ((negative_rets ** 2).mean()) ** 0.5

def annualized_sortino(mean: float, rets) -> float:
    """Calcula el Índice de Sortino anualizado para retornos diarios."""
    annual_rets = (mean * TRADING_DAYS_PER_YEAR)
    annual_std_down = downside_deviation(rets) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annual_rets / annual_std_down if annual_rets > 0 else 0

def win_rate(rets: pd.Series) -> float:
    """Calcula la tasa de aciertos (win rate) de los retornos."""
    total_trades = len(rets)
    if total_trades == 0:
        return 0
    wins = (rets > 0).sum()
    return wins / total_trades

def all_metrics(portfolio_value: pd.Series) -> pd.DataFrame:
    """
    Calcula todas las métricas de rendimiento para un portafolio.
    """
    rets = portfolio_value.pct_change().dropna()
    mean_ret = rets.mean()
    std_ret = rets.std()

    metrics = pd.DataFrame({
        'Sharpe Ratio': annualized_sharpe(mean_ret, std_ret),
        'Sortino Ratio': annualized_sortino(mean_ret, rets),
        'Calmar Ratio': annualized_calmar(mean_ret, portfolio_value),
        'Max Drawdown': maximum_drawdown(portfolio_value),
    }, index=['Metrics'])
    
    return metrics