# Función backtesting
import numpy as np
import pandas as pd
import ta

from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate, maximum_drawdown
from functions import get_portfolio_value
from operation_class import Operation


def backtest(data, stop_loss:float, take_profit:float, n_shares:float) -> float:
    # --- Preparación inicial del DataFrame ---
    # Copia el DataFrame para evitar modificar el original.
    data = data.copy()

    if "Date" in data.columns:
        data.rename(columns={"Date": "Datetime"}, inplace=True)

    # --- Preparación del DataFrame con señales ---
    historic = data.copy()
    historic = historic.dropna()

    # --- Generar señales de compra y venta basadas en la columna target ---
    historic["buy_signal"] = historic["target"] == 2
    historic["sell_signal"] = historic["target"] == 0

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

    
    # Listas para mantener las posiciones abiertas de tipo LONG y SHORT.
    active_long_positions: list[Operation] = []
    active_short_positions: list[Operation] = []

    # Lista para almacenar el valor total del portafolio en cada paso.
    portfolio_value = [cash]

    # Listas para almacenar operaciones ganadas y perdidas
    won = 0
    lost = 0

    # Conteo de señales y operaciones abiertas
    buy = 0
    sell = 0
    hold = 0

    # --- Iteración sobre cada fila del histórico para simular operaciones ---
    for row in historic.itertuples(index=False):

        # --- Cierre de posiciones LONG ---
        # Se verifica si el precio actual alcanza el stop loss o take profit para cerrar la posición.
        # Se añade el valor de cierre a cash descontando la comisión.
        for position in active_long_positions[:]: # Iterate over a copy of the list
            if position.stop_loss > row.Close  or position.take_profit < row.Close:
                # Calcular PNL previo al cierre
                fee = (row.Close) * position.n_shares * COM
                pnl = ((row.Close - position.price) * position.n_shares) - fee
                # Close the position
                cash += row.Close * position.n_shares
                # Checar si se ganó o se perdió
                if pnl > 0:
                    won += 1
                else:
                    lost += 1
                    portfolio_value.append(pnl)
                # Remove the position from active positions
                active_long_positions.remove(position)

        # Costo de operaciones SHORT
        for position in active_short_positions[:]:
            magnitud = row.Close * position.n_shares
            costo_cobertura = magnitud * BORROW_DIARIO
            cash -= costo_cobertura

        # --- Cierre de posiciones SHORT ---
        # Similar al cierre de LONG, pero con condiciones invertidas para stop loss y take profit.
        # Se calcula la ganancia o pérdida considerando la diferencia entre precio de entrada y cierre.
        for position in active_short_positions[:]:  # Iterate over a copy of the list
            if position.stop_loss < row.Close or position.take_profit > row.Close:
                # Calcular PNL previo al cierre
                fee = row.Close * position.n_shares * COM
                pnl = ((position.price - row.Close) * position.n_shares) - fee
                # Close the position
                cash += pnl
                if pnl > 0:
                    won += 1
                else:
                    lost += 1
                    portfolio_value.append(pnl)
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
                buy += 1
                active_long_positions.append(Operation(
                    time=row.Datetime,
                    price=row.Close,
                    n_shares=n_shares,
                    stop_loss=row.Close * (1 - SL),
                    take_profit=row.Close * (1 + TP),
                    type='LONG'
                ))
        if row.target == 1:
            hold += 1

        # --- Apertura de nuevas posiciones SHORT ---
        # Si la señal de venta está activa y hay suficiente cash, se abre una posición SHORT.
        # Se descuenta el costo de la operación incluyendo comisión.
        if row.sell_signal:
            # Descontar el costo
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                sell += 1
                active_short_positions.append(Operation(
                    time=row.Datetime,
                    price = row.Close,
                    n_shares = n_shares,
                    stop_loss = row.Close*(1 + SL),
                    take_profit = row.Close * (1 - TP),
                    type = 'SHORT'
                ))
        if row.target == 1:
            hold += 1

        # --- Actualización del valor del portafolio ---
        # Se calcula el valor total considerando cash y posiciones abiertas (long y short).
        portfolio_value.append(get_portfolio_value(
            cash, long_ops=active_long_positions, short_ops=active_short_positions,
            current_price=row.Close, n_shares=n_shares,
        ))

        # Close long positions
    for position in active_long_positions:
        pnl = (row.Close - position.price) * position.n_shares * (1 - COM)
        cash += row.Close * position.n_shares * (1 - COM)
        # Ganó o perdió?
        if pnl >= 0:
            won += 1
        else:
            lost += 1

    for position in active_short_positions:
        pnl = (position.price - row.Close) * position.n_shares
        short_com = row.Close * position.n_shares * COM
        cash += pnl - short_com
        # Ganó o perdió?
        if pnl >= 0:
            won += 1
        else:
            lost += 1


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
    mean_t = df.rets.mean()
    std_t = df.rets.std()
    values_port = df['value']
    sharpe_anual = annualized_sharpe(mean=mean_t, std=std_t)
    calmar = annualized_calmar(mean=mean_t, values=values_port)
    sortino = annualized_sortino(mean_t, df['rets'])
    max_drawdown = maximum_drawdown(values_port)

    # - Win Rate
    win_rate = won / (won+lost) if (won+lost) > 0 else 0
    total_ops = won + lost + hold

    # --- Preparación de resultados ---
    # Se crea un DataFrame con el valor final del portafolio y las métricas calculadas.
    results = pd.DataFrame()
    results['Portfolio'] = df['value'].tail(1)
    results['Sharpe'] = sharpe_anual
    results['Calmar'] = calmar
    results['Sortino'] = sortino
    results['Win Rate'] = win_rate
    results['Max Drawdown'] = max_drawdown

    # --- Salida de la función ---
    # Si no se pasan parámetros, se devuelve solo la métrica Calmar para optimización.
    # Si se pasan parámetros, se devuelve Calmar, la serie de valores del portafolio y el DataFrame de resultados.
    print(results)
    print(f"Terminando con cash:{cash:.4f}, valor final port:{portfolio_value[-1]:.4f} \ntotal moves hold:{hold}, total de operaciones:{total_ops}")

    return cash, portfolio_value, buy, sell, hold, total_ops
