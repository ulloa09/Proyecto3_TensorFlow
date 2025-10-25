import numpy as np
import pandas as pd

days = 252

# --- Calcula el Índice de Sharpe anualizado ---
# Esta función recibe la media y desviación estándar de retornos horarios,
# y calcula el Sharpe ratio anualizado asumiendo 8760 horas por año.
# El Sharpe ratio mide la rentabilidad ajustada por riesgo.
def annualized_sharpe(mean: float, std: float) -> float:
    annual_rets = (mean * days)
    annual_std = std * np.sqrt(days)

    return annual_rets / annual_std if annual_std > 0 else 0

# --- Calcula la Máxima Pérdida desde un pico (Drawdown máximo) ---
# Recibe una serie temporal de valores y calcula la máxima caída porcentual
# desde un máximo acumulado previo, útil para evaluar la peor caída histórica.
def maximum_drawdown(values: pd.Series) -> float:
    roll_max = values.cummax()
    max_drawdown = (roll_max - values) / roll_max
    return max_drawdown.max()

# --- Calcula el Índice de Calmar anualizado ---
# Combina la rentabilidad anualizada con la máxima pérdida para medir
# la relación entre retorno y riesgo de caída máxima.
def annualized_calmar(mean, values) -> float:
    annual_rets = (mean * days)
    max_drawdown = maximum_drawdown(values)
    return annual_rets / max_drawdown if max_drawdown != 0 else 0

# --- Calcula la desviación estándar sólo de los retornos negativos ---
# Esta métrica se usa para evaluar la volatilidad a la baja,
# ignorando las ganancias, y es base para el cálculo del Sortino ratio.
def downside_deviation(rets) -> float:
    negative_rets = rets[rets < 0]
    return ((negative_rets ** 2).mean()) ** 0.5

# --- Calcula el Índice de Sortino anualizado ---
# Similar al Sharpe, pero utiliza la desviación a la baja para el denominador,
# enfocándose en el riesgo de pérdidas en lugar de la volatilidad total.
def annualized_sortino(mean: float, rets) -> float:
    annual_rets = (mean * days)
    annual_std_down = downside_deviation(rets) * np.sqrt(days)
    return annual_rets / annual_std_down if annual_rets > 0 else 0

# --- Calcula la tasa de aciertos (win rate) ---
# Proporción de retornos positivos respecto al total de operaciones,
# indicador simple de la frecuencia de operaciones ganadoras.
def win_rate(rets: pd.Series) -> float:
    total_trades = len(rets)
    if total_trades == 0:
        return 0
    wins = (rets > 0).sum()
    return wins / total_trades