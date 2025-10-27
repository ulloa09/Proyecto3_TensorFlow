import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

def plot_portfolio_combined(
    port_train: pd.Series, 
    port_test: pd.Series, 
    port_val: pd.Series, 
    model_name: str,
    drift_series: Optional[pd.Series] = None
):
    """
    1. Combines and plots the portfolio performance across all 3 periods.
    
    *** NEW ***
    Includes a secondary Y-axis (twinx) to overlay the evolution
    of the data drift (count of drifted features) over time.

    Args:
        port_train (pd.Series): Training portfolio series.
        port_test (pd.Series): Test portfolio series.
        port_val (pd.Series): Validation portfolio series.
        model_name (str): The name of the model being plotted.
        drift_series (Optional[pd.Series]): A time series where the index
            is the date and the value is the count of drifted features.
    """
    # Concatenate the series. The DatetimeIndex will handle the correct order.
    combined_portfolio = pd.concat([port_train, port_test, port_val]).sort_index()
    combined_portfolio = combined_portfolio[~combined_portfolio.index.duplicated(keep='last')]

    # --- Setup Plot ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # --- Plot 1: Portfolio Equity (Left Y-Axis) ---
    ax1.plot(combined_portfolio.index, combined_portfolio, color='blue', label=f'{model_name} Equity')
    ax1.set_title(f'Combined Portfolio Value & Data Drift - {model_name}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # --- Add Period Lines (on ax1) ---
    lines, labels = ax1.get_legend_handles_labels()
    
    if not port_train.empty:
        start_line = ax1.axvline(port_train.index.min(), color='gray', linestyle=':', linewidth=1.5, label='Train Start')
        lines.append(start_line)
        labels.append('Train Start')
    if not port_test.empty:
        test_line = ax1.axvline(port_test.index.min(), color='darkorange', linestyle='--', linewidth=2, label='Test Start')
        lines.append(test_line)
        labels.append('Test Start')
    if not port_val.empty:
        val_line = ax1.axvline(port_val.index.min(), color='darkgreen', linestyle='--', linewidth=2, label='Validation Start')
        lines.append(val_line)
        labels.append('Validation Start')

    # --- Plot 2: Data Drift (Right Y-Axis) ---
    if drift_series is not None and not drift_series.empty:
        ax2 = ax1.twinx()  # Create a second Y-axis
        ax2.plot(drift_series.index, drift_series, color='red', linestyle=':', alpha=0.7, label='Drifted Features (Count)')
        ax2.set_ylabel('Count of Drifted Features', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Get labels from ax2 and add them to ax1's legend
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines.extend(lines2)
        labels.extend(labels2)

    # Show combined legend
    ax1.legend(lines, labels, loc='best')
    
    fig.tight_layout()
    plt.show()


def plot_comparison_with_buy_and_hold(
    port_train: pd.Series, 
    port_test: pd.Series, 
    port_val: pd.Series, 
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model_name: str
):
    """
    2. Plots the strategy equity curve against a Buy & Hold (B&H) strategy.
    
    The B&H strategy is calculated independently for each period (Train, Test, Val)
    and reset to the portfolio's starting capital (1M) for that period.

    Args:
        port_train (pd.Series): Training portfolio series.
        port_test (pd.Series): Test portfolio series.
        port_val (pd.Series): Validation portfolio series.
        train_df (pd.DataFrame): The *original* (unscaled) training data.
        test_df (pd.DataFrame): The *original* (unscaled) test data.
        validation_df (pd.DataFrame): The *original* (unscaled) validation data.
        model_name (str): The name of the model (e.g., "CNN" or "MLP").
    """
    
    # 1. Combine the strategy equity
    strategy_equity = pd.concat([port_train, port_test, port_val]).sort_index()
    strategy_equity = strategy_equity[~strategy_equity.index.duplicated(keep='last')]
    strategy_equity.name = model_name

    # --- 2. Calculate B&H per Period ---
    
    def get_period_buy_and_hold(portfolio_series, price_df):
        """Helper function to normalize B&H for a single period."""
        if portfolio_series.empty or price_df.empty:
            return pd.Series(dtype=float)
            
        price_data = price_df.set_index('Date')['Close'].sort_index().dropna()
        initial_cash = portfolio_series.iloc[0]
        first_trade_date = portfolio_series.index[1]
        price_data = price_data[price_data.index >= first_trade_date]
        
        if price_data.empty:
            return pd.Series(dtype=float)

        initial_price = price_data.iloc[0]
        bh_equity = (price_data / initial_price) * initial_cash
        initial_cash_series = pd.Series([initial_cash], index=[portfolio_series.index[0]])
        
        return pd.concat([initial_cash_series, bh_equity])

    # Calculate B&H for each period separately
    bh_train = get_period_buy_and_hold(port_train, train_df)
    bh_test = get_period_buy_and_hold(port_test, test_df)
    bh_val = get_period_buy_and_hold(port_val, validation_df)

    # Combine the three B&H periods
    buy_and_hold_equity = pd.concat([bh_train, bh_test, bh_val]).sort_index()
    buy_and_hold_equity = buy_and_hold_equity[~buy_and_hold_equity.index.duplicated(keep='last')]
    buy_and_hold_equity.name = "Buy & Hold (Per-Period)"

    # --- 3. Plot both series ---
    plt.figure(figsize=(14, 7))
    
    strategy_equity.plot(title='Strategy vs. Buy & Hold (Period-Normalized)', color='blue', label=model_name)
    buy_and_hold_equity.plot(color='gray', linestyle='--', label='Buy & Hold (Per-Period)')
    
    # Add vertical lines to mark the zones
    if not port_train.empty:
        plt.axvline(port_train.index.min(), color='gray', linestyle=':', linewidth=1.5, label='Train Start')
    if not port_test.empty:
        plt.axvline(port_test.index.min(), color='darkorange', linestyle='--', linewidth=2, label='Test Start')
    if not port_val.empty:
        plt.axvline(port_val.index.min(), color='darkgreen', linestyle='--', linewidth=2, label='Validation Start')

    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()