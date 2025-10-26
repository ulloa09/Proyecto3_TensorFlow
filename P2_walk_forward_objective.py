import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from backtest import backtest
import pandas as pd

# --- Propósito general ---
# Este archivo implementa la función objetivo para la optimización de parámetros
# mediante Optuna, utilizando validación cruzada temporal (walk-forward analysis).
# La función evalúa parámetros en diferentes segmentos temporales y devuelve el promedio
# del Calmar ratio obtenido, permitiendo seleccionar los parámetros óptimos para el backtest.
def walk_forward_objective(trial, data: pd.DataFrame, n_splits: int) -> float:
    """
    Función objetivo para Optuna con validación cruzada temporal (walk-forward analysis).
    Evalúa los parámetros propuestos en varios segmentos de tiempo
    y devuelve el promedio del Calmar ratio obtenido.

    Args:
        trial: objeto Optuna que genera los parámetros.
        data: DataFrame con los datos históricos.
        n_splits: número de divisiones temporales para la validación cruzada.

    Returns:
  
        float: promedio del Calmar ratio en todos los splits.
    """

    # --- Definición de parámetros a optimizar ---
    # Aquí se definen los parámetros que Optuna buscará optimizar.
    # Cada parámetro es sugerido dentro de un rango específico,
    # siguiendo la configuración esperada para el backtest.
    params = {
        'stop_loss': trial.suggest_float('stop_loss', 0.02, 0.05),
        'take_profit': trial.suggest_float('take_profit', 0.04, 0.15),
        'rsi_window': trial.suggest_int('rsi_window', 10, 30),
        'rsi_lower': trial.suggest_int('rsi_lower', 25, 35),
        'rsi_upper': trial.suggest_int('rsi_upper', 65, 75),
        'macd_fast': trial.suggest_int('macd_fast', 5, 12),
        'macd_slow': trial.suggest_int('macd_slow', 20, 40),
        'macd_signal': trial.suggest_int('macd_signal', 9, 18),
        'bb_window': trial.suggest_int('bb_window', 
20, 50),
        'bb_std': trial.suggest_int('bb_std', 1, 3),
        'obv_window': trial.suggest_int('obv_window', 20, 50),
        'atr_window': trial.suggest_int('atr_window', 10, 30),
        'atr_mult': trial.suggest_float('atr_mult', 1, 2.5),
        'adx_window': trial.suggest_int('adx_window', 10, 30),
        'adx_tresh': trial.suggest_int('adx_tresh', 20, 30),
        'n_shares': trial.suggest_float('n_shares', 0.5, 5),
    }

    # --- Configuración de la validación cruzada temporal ---
    # Se utiliza TimeSeriesSplit para dividir los datos en n_splits segmentos
    # manteniendo el orden temporal, lo que es crucial para evitar fugas de información
    # en series temporales.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    # --- Evaluación de cada split temporal ---
    # Para cada segmento de prueba generado por TimeSeriesSplit,
    # se ejecuta el backtest con los parámetros actuales y se calcula el Calmar ratio.
    for _, test_idx in tscv.split(data):
        test_data = data.iloc[test_idx].reset_index(drop=True)

        # Ejecuta tu backtest con los parámetros del trial actual
        calmar, _, _ = backtest(trial=None, data=test_data, params=params)

        # Guarda la métrica Calmar obtenida en este split
        scores.append(calmar)

    # --- Resultado final ---
    # Se devuelve el promedio del Calmar ratio obtenido en todos los splits,
    # representando el desempeño medio esperado con los parámetros actuales.
    return float(np.mean(scores))