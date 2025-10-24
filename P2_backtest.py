# Función backtesting
import numpy as np
import pandas as pd
import ta

from functions import rsi_signals, macd_signals, bbands_signals, obv_signals, atr_breakout_signals, adx_signals
from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate
from models import Operation, get_portfolio_value


def backtest(data, trial, params=None) -> float:
    # --- Preparación inicial del DataFrame ---
    # Copia el DataFrame para evitar modificar el original.
    # Convierte la columna 'timestamp' a formato datetime y establece el índice temporal.
    data = data.copy()
    data['Datetime'] = pd.to_datetime(data['timestamp'], unit= 'ms', errors='coerce')
    data.set_index('Datetime')

    # --- Definición de parámetros de trading ---
    # Si se recibe un trial de Optuna, se sugieren valores para los parámetros de la estrategia.
    # Si se reciben parámetros directamente, se usan esos valores.
    # Estos parámetros controlan los umbrales para indicadores técnicos, gestión de riesgo y tamaño de posición.
    if trial is not None:
        # --- cuando Optuna optimiza ---
        stop_loss = trial.suggest_float('stop_loss', 0.02, 0.05)
        take_profit = trial.suggest_float('take_profit', 0.04, 0.15)
        rsi_window = trial.suggest_int('rsi_window', 10, 30)
        rsi_lower = trial.suggest_int('rsi_lower', 25, 35)
        rsi_upper = trial.suggest_int('rsi_upper', 65, 75)
        macd_fast = trial.suggest_int('macd_fast', 5, 12)
        macd_slow = trial.suggest_int('macd_slow', 20, 40)  # debe ser > fast
        macd_signal = trial.suggest_int('macd_signal', 9, 18)
        bb_window = trial.suggest_int('bb_window', 20, 50)
        bb_std = trial.suggest_int('bb_std', 1, 3)
        obv_window = trial.suggest_int('obv_window', 20, 50)
        atr_window = trial.suggest_int('atr_window', 10, 30)
        atr_mult = trial.suggest_float('atr_mult', 1, 2.5)
        adx_window = trial.suggest_int('adx_window', 10, 30)
        adx_tresh = trial.suggest_int('adx_tresh', 20, 30)
        n_shares = trial.suggest_float('n_shares', 0.5, 5)
    elif params is not None:
        # --- cuando se usa con best_params ---
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        rsi_window = params['rsi_window']
        rsi_lower = params['rsi_lower']
        rsi_upper = params['rsi_upper']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        macd_signal = params['macd_signal']
        bb_window = params['bb_window']
        bb_std = params['bb_std']
        obv_window = params['obv_window']
        atr_window = params['atr_window']
        atr_mult = params['atr_mult']
        adx_window = params['adx_window']
        adx_tresh = params['adx_tresh']
        n_shares = params['n_shares']
    else:
        # En caso de no recibir ni trial ni parámetros, se lanza un error.
        raise ValueError("Debes pasar un trial de Optuna o un diccionario params.")

    # --- Cálculo de señales técnicas ---
    # Se obtienen señales de compra y venta basadas en diferentes indicadores técnicos.
    # Cada función devuelve series booleanas indicando cuándo se activa cada señal.
    buy_rsi, sell_rsi = rsi_signals(data, rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
    buy_macd, sell_macd = macd_signals(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    buy_bbands, sell_bbands = bbands_signals(data, bb_window, bb_std)
    buy_obv, sell_obv = obv_signals(data, window=obv_window)
    buy_atr, sell_atr = atr_breakout_signals(data, atr_window=atr_window, atr_mult=atr_mult)
    buy_adx, sell_adx = adx_signals(data, window=adx_window, threshold=adx_tresh)

    # --- Combinación de señales ---
    # Se concatenan las señales de compra y venta para contar cuántas se activan simultáneamente.
    # La estrategia requiere al menos 2 señales de compra o venta para generar una señal definitiva.
    buy_df = pd.concat([
                        buy_rsi,
                        buy_macd,
                        buy_bbands,
                        buy_obv,
                        buy_atr,
                        buy_adx,
    ], axis=1)
    sell_df = pd.concat([
                         sell_rsi,
                         #sell_macd,
                         sell_bbands,
                         #sell_obv,
                         #sell_atr,
                         #sell_adx
    ], axis=1)

    # Condición: al menos 2 señales activas para confirmar compra o venta
    buy_signal = (buy_df.sum(axis=1) >= 2)
    sell_signal = (sell_df.sum(axis=1) >= 2)

    # --- Preparación del DataFrame con señales ---
    # Se limpia el DataFrame para eliminar filas con datos faltantes y se añaden columnas con las señales generadas.
    historic = data.copy()
    historic['buy_signal'] = buy_signal
    historic['sell_signal'] = sell_signal

    # --- Inicialización de variables para la simulación ---
    # COM representa el costo por operación (comisión).
    # SL y TP son los niveles de stop loss y take profit.
    # cash es el capital inicial disponible.
    COM = 0.125 / 100
    SL = stop_loss
    TP = take_profit

    cash = 1_000_000

    # Listas para mantener las posiciones abiertas de tipo LONG y SHORT.
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []
    # Lista para almacenar el valor total del portafolio en cada paso.
    portfolio_value = [cash]

    # --- Iteración sobre cada fila del histórico para simular operaciones ---
    for row in historic.itertuples(index=False):

        # --- Cierre de posiciones LONG ---
        # Se verifica si el precio actual alcanza el stop loss o take profit para cerrar la posición.
        # Se añade el valor de cierre a cash descontando la comisión.
        for position in active_long_positions[:]: # Iterate over a copy of the list
            if position.stop_loss > row.Close  or position.take_profit < row.Close:
                # Close the position
                cash += row.Close * position.n_shares * (1 - COM)
                # Remove the position from active positions
                active_long_positions.remove(position)

        # --- Cierre de posiciones SHORT ---
        # Similar al cierre de LONG, pero con condiciones invertidas para stop loss y take profit.
        # Se calcula la ganancia o pérdida considerando la diferencia entre precio de entrada y cierre.
        for position in active_short_positions[:]:  # Iterate over a copy of the list
            if position.stop_loss < row.Close or position.take_profit > row.Close:
                # Close the position
                cash += ((position.price * position.n_shares) + (position.price * n_shares - row.Close * position.n_shares))*(1 - COM)
                # Remove the position from active positions
                active_short_positions.remove(position)


        # --- Apertura de nuevas posiciones LONG ---
        # Si la señal de compra está activa y hay suficiente cash, se abre una posición LONG.
        # Se descuenta el costo de la operación incluyendo comisión.
        if row.buy_signal:
            # Descontar el costo
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long_positions.append(Operation(
                    time=row.Datetime,
                    price=row.Close,
                    n_shares=n_shares,
                    stop_loss=row.Close * (1 - SL),
                    take_profit=row.Close * (1 + TP),
                    type='LONG'
                ))

        # --- Apertura de nuevas posiciones SHORT ---
        # Si la señal de venta está activa y hay suficiente cash, se abre una posición SHORT.
        # Se descuenta el costo de la operación incluyendo comisión.
        if row.sell_signal:
            # Descontar el costo
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_short_positions.append(Operation(
                    time=row.Datetime,
                    price = row.Close,
                    n_shares = n_shares,
                    stop_loss = row.Close*(1 + SL),
                    take_profit = row.Close * (1 - TP),
                    type = 'SHORT'
                ))

        # --- Actualización del valor del portafolio ---
        # Se calcula el valor total considerando cash y posiciones abiertas (long y short).
        portfolio_value.append(get_portfolio_value(
            cash, long_ops=active_long_positions, short_ops=active_short_positions,
            current_price=row.Close, COM=COM
        ))

    # --- Limpieza de posiciones abiertas al final del backtest ---
    active_long_positions = []
    active_short_positions = []

    # --- Cálculo de métricas de rendimiento ---
    # Se crea un DataFrame con el valor del portafolio y los retornos diarios.
    df = pd.DataFrame()
    df['value'] = portfolio_value
    df['rets'] = df.value.pct_change()
    #df.dropna(inplace=True)

    # Se calculan las métricas estadísticas para evaluar la estrategia:
    # - Sharpe anualizado mide el retorno ajustado al riesgo.
    # - Calmar anualizado mide retorno ajustado a la máxima caída.
    # - Sortino anualizado mide retorno ajustado a la volatilidad negativa.
    # - Win Rate es la proporción de días con retorno positivo.
    mean_t = df.rets.mean()
    std_t = df.rets.std()
    values_port = df['value']
    sharpe_anual = annualized_sharpe(mean=mean_t, std=std_t)
    calmar = annualized_calmar(mean=mean_t, values=values_port)
    sortino = annualized_sortino(mean_t, df['rets'])
    wr = win_rate(df['rets'])

    # --- Preparación de resultados ---
    # Se crea un DataFrame con el valor final del portafolio y las métricas calculadas.
    results = pd.DataFrame()
    results['Portfolio'] = df['value'].tail(1)
    results['Sharpe'] = sharpe_anual
    results['Calmar'] = calmar
    results['Sortino'] = sortino
    results['Win Rate'] = wr

    # --- Salida de la función ---
    # Si no se pasan parámetros, se devuelve solo la métrica Calmar para optimización.
    # Si se pasan parámetros, se devuelve Calmar, la serie de valores del portafolio y el DataFrame de resultados.
    if params is None:
        return calmar
    else:
        return calmar, values_port, results
