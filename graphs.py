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


def plot_labels_over_price(df):
    """
    Plot labeled trading signals (0=sell, 1=hold, 2=buy) over the price series.

    This function visualizes where each label occurs along the price curve.
    It automatically detects the 'Close' and 'target' columns from the given DataFrame
    and highlights buy/sell/hold points for easier inspection of labeling quality.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'Close': price series to plot.
        - 'target': label column where 0=sell, 1=hold, 2=buy.

    Raises
    ------
    ValueError
        If the DataFrame does not contain both 'Close' and 'target' columns.
    """

    # --- Validate input DataFrame ---
    if 'Close' not in df.columns or 'target' not in df.columns:
        raise ValueError("DataFrame must contain both 'Close' and 'target' columns.")

    # --- Initialize plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], color='gray', alpha=0.6, label='Price')

    # --- Overlay labeled signals ---
    plt.scatter(df[df['target'] == 2].index, df[df['target'] == 2]['Close'],
                color='green', marker='^', label='Buy (2)', s=10, alpha=0.8)
    plt.scatter(df[df['target'] == 0].index, df[df['target'] == 0]['Close'],
                color='red', marker='v', label='Sell (0)', s=10, alpha=0.8)
    plt.scatter(df[df['target'] == 1].index, df[df['target'] == 1]['Close'],
                color='blue', marker='o', label='Hold (1)', s=10, alpha=0.5)

    # --- Configure plot style ---
    plt.title('Threshold-Based Labeling Visualization')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


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
        start_line = ax1.axvline(port_train.index.min(), color='gray', linestyle=':', linewidth=1.5,
                                 label='Train Start')
        lines.append(start_line)
        labels.append('Train Start')
    if not port_test.empty:
        test_line = ax1.axvline(port_test.index.min(), color='darkorange', linestyle='--', linewidth=2,
                                label='Test Start')
        lines.append(test_line)
        labels.append('Test Start')
    if not port_val.empty:
        val_line = ax1.axvline(port_val.index.min(), color='darkgreen', linestyle='--', linewidth=2,
                               label='Validation Start')
        lines.append(val_line)
        labels.append('Validation Start')

    # --- Plot 2: Data Drift (Right Y-Axis) ---
    if drift_series is not None and not drift_series.empty:
        ax2 = ax1.twinx()  # Create a second Y-axis
        ax2.plot(drift_series.index, drift_series, color='red', linestyle=':', alpha=0.7,
                 label='Drifted Features (Count)')
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


def plot_labels_over_price(df):
    """
    Plot labeled trading signals (0=sell, 1=hold, 2=buy) over the price series.

    This function visualizes where each label occurs along the price curve using the 'Date' column
    as the X-axis. It highlights buy/sell/hold points for a quick inspection of labeling quality.

    Parameters
    ----------
    df :
        DataFrame containing the following columns:
        - 'Date' (datetime): date for each observation.
        - 'Close' (float): price series to plot.
        - 'target' (int): label column where 0=sell, 1=hold, 2=buy.

    Raises
    ------
    ValueError
        If the DataFrame does not contain 'Date', 'Close', or 'target' columns.
    """
    from matplotlib.dates import AutoDateLocator, AutoDateFormatter
    import matplotlib.pyplot as plt

    # --- Validate DataFrame ---
    if not all(col in df.columns for col in ['Date', 'Close', 'target']):
        raise ValueError("DataFrame must contain 'Date', 'Close', and 'target' columns.")

    # --- Initialize Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Close'], color='gray', alpha=0.6, label='Price')

    # --- Overlay Labels ---
    ax.scatter(df.loc[df['target'] == 2, 'Date'], df.loc[df['target'] == 2, 'Close'],
               color='green', marker='^', label='Buy (2)', s=10, alpha=0.8)
    ax.scatter(df.loc[df['target'] == 0, 'Date'], df.loc[df['target'] == 0, 'Close'],
               color='red', marker='v', label='Sell (0)', s=10, alpha=0.8)
    ax.scatter(df.loc[df['target'] == 1, 'Date'], df.loc[df['target'] == 1, 'Close'],
               color='blue', marker='o', label='Hold (1)', s=10, alpha=0.5)

    # --- Configure Date Axis ---
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # --- Style ---
    ax.set_title('Threshold-Based Labeling Visualization')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_training_history(cnn_history, mlp_history):
    """
    Plot training and validation accuracy and loss curves for both CNN and MLP models.

    Args:
        cnn_history: Keras History object from the best CNN model.
        mlp_history: Keras History object from the best MLP model.

    The function displays two plots:
        1. Accuracy curves for CNN and MLP (train and validation).
        2. Loss curves for CNN and MLP (train and validation).
    """
    # Extract history dictionaries from Keras History objects if needed
    cnn_hist = cnn_history.history if hasattr(cnn_history, 'history') else cnn_history
    mlp_hist = mlp_history.history if hasattr(mlp_history, 'history') else mlp_history

    # --- Accuracy Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(cnn_hist['accuracy'], label='CNN Train Acc', color='blue')
    plt.plot(cnn_hist['val_accuracy'], label='CNN Val Acc', color='blue', linestyle='--')
    plt.plot(mlp_hist['accuracy'], label='MLP Train Acc', color='orange')
    plt.plot(mlp_hist['val_accuracy'], label='MLP Val Acc', color='orange', linestyle='--')
    plt.title('CNN vs. MLP Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Loss Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(cnn_hist['loss'], label='CNN Train Loss', color='blue')
    plt.plot(cnn_hist['val_loss'], label='CNN Val Loss', color='blue', linestyle='--')
    plt.plot(mlp_hist['loss'], label='MLP Train Loss', color='orange')
    plt.plot(mlp_hist['val_loss'], label='MLP Val Loss', color='orange', linestyle='--')
    plt.title('CNN vs. MLP Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()