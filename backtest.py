import numpy as np
import pandas as pd
import ta

from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate, maximum_drawdown
from functions import get_portfolio_value
from operation_class import Operation

def backtest(data, stop_loss:float, take_profit:float, n_shares:float) -> tuple:
    # --- Preparación inicial del DataFrame ---
    # Copia el DataFrame para evitar modificar el original.
    data = data.copy()

    # --- Preparación del DataFrame con señales ---
    historic = data.copy()
    
    # Asegurar que el índice es Datetime
    if 'Datetime' in historic.columns:
        historic['Datetime'] = pd.to_datetime(historic['Datetime'])
        historic.set_index('Datetime', inplace=True, drop=False)
    elif isinstance(historic.index, pd.DatetimeIndex):
         # Si ya es un DatetimeIndex, asegurarse que 'Datetime' exista
         if 'Datetime' not in historic.columns:
              historic['Datetime'] = historic.index
    else:
        print("Advertencia de Backtest: No se encontró 'Datetime'. Usando índice plano.")

    historic = historic.dropna(subset=['target', 'Close', 'High', 'Low'])

    if historic.empty:
        print("Error de Backtest: DataFrame vacío después de dropna.")
        return 1_000_000, pd.Series([1_000_000]), 0.0, 0, 0, 0, 0, None

    # --- Generar señales de compra y venta basadas en la columna target (0, 1, 2) ---
    historic["buy_signal"] = historic["target"] == 2 # 2 = Compra
    historic["sell_signal"] = historic["target"] == 0 # 0 = Venta
    # 1 = Hold (no se usa para abrir, solo para contar)

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
    portfolio_dates = [historic.index[0] - pd.Timedelta(days=1)] if isinstance(historic.index, pd.DatetimeIndex) else [0]


    # Listas para almacenar operaciones ganadas y perdidas
    won = 0
    lost = 0

    # Conteo de señales y operaciones abiertas
    buy = 0
    sell = 0
    hold = 0

    last_row = None

    # --- Iteración sobre cada fila del histórico para simular operaciones ---
    for row in historic.itertuples(index=True): # Usar index=True para Datetime
        last_row = row # Guardar la última fila para cierre final
        
        # --- Cierre de posiciones LONG ---
        # Se verifica si el precio actual alcanza el stop loss o take profit para cerrar la posición.
        # Se añade el valor de cierre a cash descontando la comisión.
        for position in active_long_positions[:]: # Iterate over a copy of the list
            if position.stop_loss > row.Close  or position.take_profit < row.Close:
                # Calcular PNL previo al cierre
                fee = (row.Close) * position.n_shares * COM
                pnl = ((row.Close - position.price) * position.n_shares) - fee
                # Close the position
                cash += row.Close * position.n_shares * (1 - COM) # Ajustado para reflejar el PnL y comisión
                
                # Checar si se ganó o se perdió
                if pnl > 0:
                    won += 1
                else:
                    lost += 1
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
                cash += (position.price * position.n_shares) + pnl # Devolver el "colateral" + PnL
                
                if pnl > 0:
                    won += 1
                else:
                    lost += 1
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
                    time=row.Index, # Usar row.Index (Datetime)
                    price=row.Close,
                    n_shares=n_shares,
                    stop_loss=row.Close * (1 - SL),
                    take_profit=row.Close * (1 + TP),
                    type='LONG'
                ))
        elif row.target == 1: # Contar hold si no es compra
            hold += 1

        # --- Apertura de nuevas posiciones SHORT ---
        # Si la señal de venta está activa y hay suficiente cash, se abre una posición SHORT.
        # (Ajuste: Lógica de costo de Short)
        if row.sell_signal:
            # Descontar el costo (solo comisión) y "apartar" el colateral (precio de venta)
            cost_comision = row.Close * n_shares * COM
            if cash > cost_comision:
                cash -= cost_comision # Pagar comisión
                cash += row.Close * n_shares # Recibir el dinero de la venta
                sell += 1
                active_short_positions.append(Operation(
                    time=row.Index, # Usar row.Index (Datetime)
                    price = row.Close,
                    n_shares = n_shares,
                    stop_loss = row.Close*(1 + SL),
                    take_profit = row.Close * (1 - TP),
                    type = 'SHORT'
                ))
        elif row.target == 1: # Contar hold si no es venta
            hold += 1

        # --- Actualización del valor del portafolio ---
        # Se calcula el valor total considerando cash y posiciones abiertas (long y short).
        current_port_val = get_portfolio_value(
            cash, long_ops=active_long_positions, short_ops=active_short_positions,
            current_price=row.Close, n_shares=n_shares,
        )
        portfolio_value.append(current_port_val)
        portfolio_dates.append(row.Index)


    # --- Cierre de posiciones al final del backtest ---
    if last_row:
        # Close long positions
        for position in active_long_positions:
            fee = (last_row.Close) * position.n_shares * COM
            pnl = ((last_row.Close - position.price) * position.n_shares) - fee
            cash += last_row.Close * position.n_shares * (1 - COM)
            if pnl >= 0:
                won += 1
            else:
                lost += 1
        # Close short positions
        for position in active_short_positions:
            fee = last_row.Close * position.n_shares * COM
            pnl = ((position.price - last_row.Close) * position.n_shares) - fee
            cash += (position.price * position.n_shares) + pnl # Devolver colateral + PnL
            if pnl >= 0:
                won += 1
            else:
                lost += 1

    # --- Limpieza de posiciones abiertas al final del backtest ---
    active_long_positions = []
    active_short_positions = []

    # --- Cálculo de métricas de rendimiento ---
    # Se crea un DataFrame con el valor del portafolio y los retornos diarios.
    df = pd.DataFrame(index=portfolio_dates)
    df['value'] = portfolio_value
    df['rets'] = df.value.pct_change()
    df.dropna(subset=['rets'], inplace=True)

    # Se calculan las métricas estadísticas para evaluar la estrategia:
    mean_t = df.rets.mean()
    std_t = df.rets.std()
    values_port = df['value']
    sharpe_anual = annualized_sharpe(mean=mean_t, std=std_t)
    calmar = annualized_calmar(mean=mean_t, values=values_port)
    sortino = annualized_sortino(mean_t, df['rets'])
    max_drawdown = maximum_drawdown(values_port)

    # - Win Rate
    win_rate = won / (won+lost) if (won+lost) > 0 else 0
    total_ops = won + lost

    # --- Preparación de resultados ---
    # Se crea un DataFrame con el valor final del portafolio y las métricas calculadas.
    results = pd.DataFrame(index=['Metrics'])
    results['Portfolio'] = cash # Usar el cash final
    results['Sharpe'] = sharpe_anual
    results['Calmar'] = calmar
    results['Sortino'] = sortino
    results['Win Rate'] = win_rate
    results['Max Drawdown'] = max_drawdown

    # Convertir a Series para 'run_backtest.py'
    portfolio_series = pd.Series(portfolio_value, index=portfolio_dates)
    portfolio_series = portfolio_series[~portfolio_series.index.duplicated(keep='last')]


    # --- Salida de la función ---
    return cash, portfolio_series, win_rate, buy, sell, hold, total_ops, results