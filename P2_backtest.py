# Función backtesting
import numpy as np
import pandas as pd
from extras import Operation, portfolio_value


def backtest(data: pd.DataFrame, 
             cash: float, 
             stop_loss: float, 
             take_profit: float, 
             n_shares: int = 1000) -> tuple[pd.Series, float, float]:
    """
    Ejecuta un backtest basado en señales de trading pre-generadas.
    """

    # --- Preparación inicial del DataFrame ---
    data = data.copy()
    if 'Date' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'])
        data.set_index('Datetime', inplace=True)
    
    data.dropna(inplace=True) 

    # --- Definición de constantes de trading ---
    COM = 0.125 / 100
    BORROW_RATE = (0.25 / 100) / 252 # Tasa de préstamo diaria
    SL = stop_loss
    TP = take_profit

    # --- Inicialización de variables de simulación ---
    active_long: list[Operation] = []
    active_short: list[Operation] = []
    
    portfolio_hist = [cash]
    n_operations = 0
    wins = 0

    # --- Iteración sobre cada fila del histórico ---
    for row in data.itertuples(index=True): 
        
        current_price = row.Close
        current_time = row.Index

        # --- Cierre de posiciones LONG ---
        for position in active_long.copy():
            if (position.stop_loss > current_price) or (position.take_profit < current_price):
                
                pnl = (current_price - position.price) * position.n_shares * (1 - COM)
                if pnl > 0:
                    wins += 1
                n_operations += 1
                
                cash += current_price * position.n_shares * (1 - COM)
                active_long.remove(position)

        # --- Cobrar costo de préstamo (Borrow Cost) en posiciones SHORT ---
        for position in active_short:
            costo_diario = current_price * position.n_shares * BORROW_RATE
            cash -= costo_diario

        # --- Cierre de posiciones SHORT ---
        for position in active_short.copy():
            if (position.stop_loss < current_price) or (position.take_profit > current_price):
                
                pnl = (position.price - current_price) * position.n_shares
                if pnl > 0:
                    wins += 1
                n_operations += 1

                com_cost = current_price * position.n_shares * COM
                cash += pnl - com_cost
                active_short.remove(position)

        
        # --- Apertura de nuevas posiciones LONG (Señal 1) ---
        if row.signal == 1:
            cost = current_price * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long.append(Operation(
                    time=current_time,
                    price=current_price,
                    n_shares=n_shares,
                    stop_loss=current_price * (1 - SL),
                    take_profit=current_price * (1 + TP),
                    type='LONG'
                ))

        # --- Apertura de nuevas posiciones SHORT (Señal -1) ---
        if row.signal == -1:
            cost = current_price * n_shares * COM
            if cash > cost:
                cash -= cost
                active_short.append(Operation(
                    time=current_time,
                    price=current_price,
                    n_shares=n_shares,
                    stop_loss=current_price * (1 + SL),
                    take_profit=current_price * (1 - TP),
                    type='SHORT'
                ))

        # --- Actualización del valor del portafolio ---
        current_port_value = portfolio_value(
            cash, active_long, active_short,
            current_price, n_shares
        )
        portfolio_hist.append(current_port_value)

    # --- Limpieza de posiciones abiertas al final del backtest ---
    last_price = data['Close'].iloc[-1]
    
    for position in active_long.copy():
        pnl = (last_price - position.price) * position.n_shares * (1 - COM)
        if pnl > 0: wins += 1
        n_operations += 1
        cash += last_price * position.n_shares * (1 - COM)

    for position in active_short.copy():
        pnl = (position.price - last_price) * position.n_shares
        if pnl > 0: wins += 1
        n_operations += 1
        com_cost = last_price * position.n_shares * COM
        cash += pnl - com_cost

    active_long = []
    active_short = []

    # --- Cálculo de métricas de rendimiento ---
    win_rate = wins / n_operations if n_operations > 0 else 0
    
    port_value = pd.Series(portfolio_hist)
    if len(port_value) == len(data.index) + 1:
        port_value.index = [data.index[0] - pd.Timedelta(days=1)] + list(data.index)
    else:
        port_value = pd.Series(portfolio_hist)


    return port_value, cash, win_rate