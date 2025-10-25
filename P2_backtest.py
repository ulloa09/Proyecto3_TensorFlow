# Función backtesting
import numpy as np
import pandas as pd
from extras import Operation, portfolio_value

def backtest(data: pd.DataFrame,
             cash: float,
             stop_loss: float,
             take_profit: float,
             n_shares: int = 100) -> tuple[pd.Series, float, float, int, int, int, int]:
   
    # --- Preparación inicial del DataFrame ---
    data = data.copy()
    if 'Date' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'])
        data.set_index('Datetime', inplace=True)
    
    # --- Definición de constantes de trading ---
    COM = 0.125 / 100
    BORROW_RATE_ANNUAL = 0.25 / 100
    INTERVALS = 252 # Días de trading por año
    BORROW_RATE_DAILY = BORROW_RATE_ANNUAL / INTERVALS # Tasa diaria
    SL = stop_loss
    TP = take_profit

    # --- Inicialización de variables de simulación ---
    active_long: list[Operation] = []
    active_short: list[Operation] = []
    portfolio_hist = [cash] 

    positive_trades = 0
    negative_trades = 0
    buy_trades = 0
    sell_trades = 0
    hold_signals = 0 

    # --- Iteración sobre cada fila del histórico ---
    last_row = None # Para cierre final
    for row in data.itertuples(index=True):
        last_row = row # Guardar la última fila
        current_price = row.Close
        current_time = row.Index

        # --- Cierre de posiciones LONG ---
        for position in active_long.copy():
            if (current_price > position.take_profit) or (current_price < position.stop_loss):
                pnl = (current_price - position.price) * position.n_shares # PnL antes de comisión
                cash += current_price * position.n_shares * (1 - COM) # Añadir valor de venta (con comisión)
                # Contar trade como positivo/negativo
                if pnl * (1-COM) >= 0: # Comprobar PnL neto
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_long.remove(position)

        # --- Cobrar costo de préstamo (Borrow Cost) en posiciones SHORT ---
        for position in active_short:
            cover_cost = current_price * position.n_shares
            borrow_cost = cover_cost * BORROW_RATE_DAILY
            cash -= borrow_cost

        # --- Cierre de posiciones SHORT ---
        for position in active_short.copy():
            if (current_price < position.take_profit) or (current_price > position.stop_loss):
                pnl = (position.price - current_price) * position.n_shares # PnL bruto
                short_com = current_price * position.n_shares * COM # Comisión de cierre
                cash += pnl - short_com # Añadir PnL neto a cash
                # Contar trade
                if pnl - short_com >= 0:
                    positive_trades += 1
                else:
                    negative_trades += 1
                active_short.remove(position)


        # --- Procesar Señal del Modelo ---
        # --- Apertura de nuevas posiciones LONG (Señal 1) ---
        if row.signal == 1:
            position_value = current_price * n_shares * (1 + COM) # Costo total
            if cash > position_value:
                cash -= position_value # Descontar costo
                buy_trades += 1 # Contar trade de compra ejecutado
                active_long.append(Operation(
                    time=current_time, price=current_price,
                    take_profit=current_price * (1 + TP), # TP para long
                    stop_loss=current_price * (1 - SL),   # SL para long
                    n_shares=n_shares, type="LONG"
                ))
        # --- Apertura de nuevas posiciones SHORT (Señal -1) ---
        elif row.signal == -1:
            # position_value = current_price * n_shares # Valor nocional
            short_cost = current_price * n_shares * COM # Solo comisión al abrir
            if cash > short_cost: # Solo necesitamos cubrir la comisión
                cash -= short_cost # Descontar comisión
                sell_trades += 1 # Contar trade de venta ejecutado
                active_short.append(Operation(
                    time=current_time, price=current_price,
                    take_profit=current_price * (1 - TP), # TP para short
                    stop_loss=current_price * (1 + SL),   # SL para short
                    n_shares=n_shares, type="SHORT"
                ))
        else:
             hold_signals += 1 # Contar señal de hold


        # --- Actualización del valor del portafolio ---
        current_port_value = portfolio_value(
            cash, active_long, active_short,
            current_price, n_shares
        )
        portfolio_hist.append(current_port_value)

    # --- Limpieza de posiciones abiertas al final del backtest (usando last_row) ---
    if last_row:
        last_price = last_row.Close
        for position in active_long.copy(): # Usar .copy() al modificar
            pnl = (last_price - position.price) * position.n_shares # PnL bruto
            cash += last_price * position.n_shares * (1 - COM)
            if pnl * (1-COM) >= 0: positive_trades += 1
            else: negative_trades += 1
            active_long.remove(position) # Eliminar de la lista original

        for position in active_short.copy(): # Usar .copy() al modificar
            pnl = (position.price - last_price) * position.n_shares
            short_com = last_price * position.n_shares * COM
            cash += pnl - short_com
            if pnl - short_com >= 0: positive_trades += 1
            else: negative_trades += 1
            active_short.remove(position) # Eliminar de la lista original


    # --- Cálculo de métricas finales ---
    total_trades = positive_trades + negative_trades
    win_rate = positive_trades / total_trades if total_trades > 0 else 0

    port_series = pd.Series(portfolio_hist)
    if not data.empty and len(port_series) == len(data.index) + 1:
         if isinstance(data.index, pd.DatetimeIndex):
              port_series.index = [data.index[0] - pd.Timedelta(days=1)] + list(data.index)
         else: # Si no es DatetimeIndex, usar índice numérico simple para evitar error
              port_series = pd.Series(portfolio_hist)
    else:
         port_series = pd.Series(portfolio_hist) # Fallback a índice simple


    return port_series, cash, win_rate, buy_trades, sell_trades, hold_signals, total_trades