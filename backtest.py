# Backtesting Function
import numpy as np
import pandas as pd
import ta

from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate, maximum_drawdown
from functions import get_portfolio_value
from operation_class import Operation


def backtest(data, stop_loss:float, take_profit:float, n_shares:float):
    """
    Vectorized backtesting engine for a trading strategy.

    Iterates through historical data, simulates trades based on 'target' signals,
    and calculates portfolio value and performance metrics.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data and a 'target' column
                             (0=Sell, 1=Hold, 2=Buy).
        stop_loss (float): Percentage (e.g., 0.05 for 5%) for stop loss.
        take_profit (float): Percentage (e.g., 0.10 for 10%) for take profit.
        n_shares (float): Number of shares to trade per operation.

    Returns:
        tuple:
            - cash (float): Final cash value.
            - portfolio_series (pd.Series): Time series of portfolio value.
            - buy (int): Total buy operations.
            - sell (int): Total sell operations.
            - hold (int): Total hold signals.
            - total_ops (int): Total closed operations (wins + losses).
    """
    
    # --- Initial DataFrame preparation ---
    # Copy the DataFrame to avoid modifying the original.
    data = data.copy()

    # Standardize date column name
    if "Date" in data.columns:
        data.rename(columns={"Date": "Datetime"}, inplace=True)

    # --- Prepare DataFrame with signals ---
    historic = data.copy()
    historic = historic.dropna()

    if historic.empty:
        print("Backtest Error: DataFrame empty after dropna.")
        # Return an empty, formatted Series to avoid errors
        return 1_000_000, pd.Series(dtype=float), 0, 0, 0, 0
          
    
    # --- Generate buy and sell signals based on the target column ---
    historic["buy_signal"] = historic["target"] == 2
    historic["sell_signal"] = historic["target"] == 0

    # --- Initialization of variables for the simulation ---
    # COM represents the cost per operation (commission).
    # SL and TP are the stop loss and take profit levels.
    # cash is the initial available capital.
    COM = 0.125 / 100  # Commission rate (0.125%)
    BORROW_RATE = 0.25 / 100 # Annual borrow rate for shorts
    SL = stop_loss
    TP = take_profit

    days = 252 # Trading days per year
    BORROW_DIARIO = BORROW_RATE / days # Daily borrow cost

    cash = 1_000_000 # Initial capital
    
    # Lists to maintain open LONG and SHORT positions.
    active_long_positions: list[Operation] = [] 
    active_short_positions: list[Operation] = [] 

    # List to store the total portfolio value at each step.
    portfolio_value = [cash] 
    # Register dates for the Series index
    portfolio_dates = [historic['Datetime'].iloc[0] - pd.Timedelta(days=1)]

    # Lists to store winning and losing operations
    won = 0
    lost = 0

    # Count of signals and open operations
    buy = 0
    sell = 0
    hold = 0
    
    last_row = None # To store the last row for final closing

    # --- Iterate over each row of the history to simulate operations ---
    for row in historic.itertuples(index=False): 
        last_row = row # Save last row for closing positions at the end

        # --- Closing LONG positions ---
        # Check if the current price hits the stop loss or take profit to close the position.
        # The closing value is added to cash, deducting commission.
        for position in active_long_positions[:]: # Iterate over a copy
            if position.stop_loss > row.Close  or position.take_profit < row.Close:
                # Calculate PNL before closing
                fee = (row.Close) * position.n_shares * COM
                pnl = ((row.Close - position.price) * position.n_shares) - fee
  
                # Close the position
                cash += row.Close * position.n_shares 
                # Check if it was a win or loss
                if pnl > 0:
                   won += 1 
                else:
                    lost += 1 
                # Remove the position from active positions
                active_long_positions.remove(position)

 
        # --- Cost of SHORT operations (Daily Borrow Fee) ---
        for position in active_short_positions[:]: 
            magnitud = row.Close * position.n_shares
            costo_cobertura = magnitud * BORROW_DIARIO
            cash -= costo_cobertura # Deduct daily fee

        # --- Closing SHORT positions ---
        # Similar to closing LONG, but with inverted conditions for stop loss and take profit.
        # Profit or loss is calculated considering the difference between entry and exit price.
        for position in active_short_positions[:]:  # Iterate over a copy
            if position.stop_loss < row.Close or position.take_profit > row.Close:
                # Calculate PNL before closing
                fee = row.Close * position.n_shares * COM
                pnl = ((position.price - row.Close) * position.n_shares) - fee
  
                # Close the position (add PNL to cash)
                cash += pnl 
                if pnl > 0:
                    won += 1 
                else:
                    lost += 1 
                # Remove the position from active positions
                active_short_positions.remove(position)


        # --- Opening new LONG positions ---
        # If the buy signal is active and there is enough cash, open a LONG position.
        # The cost of the operation, including commission, is deducted.
        if row.buy_signal: 
            # Deduct cost
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                buy += 1
            
                # Add new operation to the active list
                active_long_positions.append(Operation(
                        time=row.Datetime,
                        price=row.Close,
                        n_shares=n_shares,
                        stop_loss=row.Close * (1 - SL),
                        take_profit=row.Close * (1 + TP),
                        type='LONG'
                    ))
        if row.target == 1:
            hold += 1 # Count hold signals

       
        # --- Opening new SHORT positions ---
        # If the sell signal is active and there is enough cash, open a SHORT position.
        # The cost of the operation (commission) is deducted.
        if row.sell_signal:
            # Deduct cost
            cost = row.Close * n_shares * (1 + COM) # Commission cost
            if cash > cost:
                cash -= cost
                sell += 1
             
                # Add new operation to the active list
                active_short_positions.append(Operation(
                        time=row.Datetime,
                        price = row.Close,
                        n_shares = n_shares,
                        stop_loss = row.Close*(1 + SL),
                        take_profit = row.Close * (1 - TP),
                        type = 'SHORT'
                    ))
        if row.target == 1:
            hold += 1 # Count hold signals (double counted, fix if needed)

  
        # --- Update portfolio value ---
        # Calculate the total value considering cash and open positions (long and short).
        current_port_val = get_portfolio_value(
            cash, long_ops=active_long_positions, short_ops=active_short_positions,
            current_price=row.Close, n_shares=n_shares,
        )
        portfolio_value.append(current_port_val)
        portfolio_dates.append(row.Datetime) # Register the date

    # --- Close all open positions at the end of the backtest ---
    if last_row:
        # Close remaining long positions
        for position in active_long_positions:
            pnl = (last_row.Close - position.price) * position.n_shares * (1 - COM)
            cash += last_row.Close * position.n_shares * (1 - COM)
            # Win or loss?
            if pnl >= 0:
                won += 1
            else:
                lost += 1

        # Close remaining short positions
        for position in active_short_positions:
            pnl = (position.price - last_row.Close) * position.n_shares
            short_com = last_row.Close * position.n_shares * COM
            cash += pnl - short_com # Add PNL, subtract commission
            # Win or loss?
            if pnl >= 0:
                won += 1
            else:
                lost += 1


    # --- Clear open positions lists after closing ---
    active_long_positions = []
    active_short_positions = []

    # --- Calculate performance metrics ---
    # Create a DataFrame with the portfolio value and daily returns.
    df = pd.DataFrame(index=portfolio_dates)
    df['value'] = portfolio_value
    df['rets'] = df.value.pct_change()
    df.dropna(subset=['rets'], inplace=True) # Use subset='rets'

    # Calculate statistical metrics to evaluate the strategy:
    # - Annualized Sharpe: measures risk-adjusted return.
    # - Annualized Calmar: measures return adjusted for maximum drawdown.
    # - Annualized Sortino: measures return adjusted for downside volatility.
    mean_t = df.rets.mean()
    std_t = df.rets.std()
    values_port_for_metrics = df['value']
    sharpe_anual = annualized_sharpe(mean=mean_t, std=std_t)
    calmar = annualized_calmar(mean=mean_t, values=values_port_for_metrics)
    sortino = annualized_sortino(mean_t, df['rets'])
    max_drawdown = maximum_drawdown(values_port_for_metrics)

    # - Win Rate
    win_rate_val = won / (won+lost) if (won+lost) > 0 else 0
    total_ops = won + lost # Total closed operations

    # --- Prepare results ---
    # Create a DataFrame with the final portfolio value and calculated metrics.
    results = pd.DataFrame()
    results['FinalPortfolioVal'] = df['value'].tail(1)
    results['Sharpe'] = sharpe_anual
    results['Calmar'] = calmar
    results['Sortino'] = sortino
    results['Win Rate'] = win_rate_val
    results['Max Drawdown'] = max_drawdown

    # Create the complete portfolio Series for plotting
    portfolio_series = pd.Series(portfolio_value, index=portfolio_dates)
    portfolio_series = portfolio_series[~portfolio_series.index.duplicated(keep='last')] # Remove duplicates

    # --- Function output ---
    print(results)
    print(f"Finishing with cash:{cash:.4f}, final portfolio value:{portfolio_value[-1]:.4f} \ntotal moves (buy+sell):{sell+buy}, total closed ops:{total_ops}")

    return cash, portfolio_series, buy, sell, hold, total_ops