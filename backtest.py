# Función backtesting
import numpy as np
import pandas as pd
import ta

from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate
from models import Operation, get_portfolio_value


def backtest(data, stop_loss:float, take_profit:float, n_shares:float) -> float:
    # --- Preparación inicial del DataFrame ---
    # Copia el DataFrame para evitar modificar el original.
    data = data.copy()

    # --- Preparación del DataFrame con señales ---
    historic = data.copy()
    historic = historic.dropna()


    # --- Inicialización de variables para la simulación ---
    # COM representa el costo por operación (comisión).
    # SL y TP son los niveles de stop loss y take profit.
    # cash es el capital inicial disponible.
    COM = 0.125 / 100
    BORROW_RATE = 0.25 / 100
    SL = stop_loss
    TP = take_profit

    days = 252
    BORROW_DIARIO = BORROW_RATE / days

    cash = 1_000_000

    '''
    
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

'''