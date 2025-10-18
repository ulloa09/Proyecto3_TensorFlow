import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtest import backtest

def show_results(train_df, test_df, validation_df, params, plot: bool = True):
    """
    Calcula y grafica rendimientos agrupados (mensuales, trimestrales y anuales)
    usando el valor de portafolio resultante del backtest en train, test y validation.
    También devuelve las series/tablitas para usarlas en reportes.

    Parámetros:
        train_df, test_df, validation_df: DataFrames con columna 'timestamp' (en ms) o 'Datetime'.
        params: dict con los mejores parámetros para el backtest.
        plot: si True, muestra las gráficas.

    Retorna:
        dict con las series de rendimientos y la curva total del portafolio.
    """

    def run_and_align(df):
        _, value, _ = backtest(trial=None, data=df, params=params)
        # Construir índice temporal
        if 'timestamp' in df.columns:
            idx = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        elif 'Datetime' in df.columns:
            idx = pd.to_datetime(df['Datetime'], errors='coerce')
        else:
            idx = pd.to_datetime(df.index, errors='coerce')

        idx = idx[idx.notna()]
        v = value.copy()

        # Alinear longitudes
        if len(v) == len(df) + 1:
            v = v.iloc[1:]
        if len(v) != len(idx):
            min_len = min(len(v), len(idx))
            v = v.iloc[-min_len:]
            idx = idx.iloc[-min_len:]

        v.index = idx
        v.name = 'value'
        return v

    v_train = run_and_align(train_df)
    v_test = run_and_align(test_df)
    v_val = run_and_align(validation_df)

    # Serie completa del portafolio
    portfolio = pd.concat([v_train, v_test, v_val]).sort_index()
    portfolio = portfolio[~portfolio.index.duplicated(keep='last')]

    def period_returns(series, rule):
        return series.resample(rule).last().pct_change().dropna()

    monthly = period_returns(portfolio, 'M')
    quarterly = period_returns(portfolio, 'Q')
    annual = period_returns(portfolio, 'Y')

    def print_table(name, s):
        if name == 'Monthly':
            periods = s.index.to_period('M').astype(str)
        elif name == 'Quarterly':
            periods = s.index.to_period('Q').astype(str)
        else:
            periods = s.index.to_period('Y').astype(str)
        tbl = pd.DataFrame({'Period': periods, 'Return %': (s.values * 100).round(2)})
        print(f"\n{name} returns")
        print(tbl.to_string(index=False))

    print_table('Annual', annual)
    print_table('Quarterly', quarterly)
    print_table('Monthly', monthly)
    print('Rendimiento anual promedio:\n',np.round(annual.mean(),4))

    if plot:
        (annual * 100).plot(kind='bar', figsize=(10, 4), title='Annual Returns (%)')
        plt.axhline(0, linewidth=1)
        plt.tight_layout()
        plt.show()

        (quarterly * 100).plot(kind='bar', figsize=(14, 4), title='Quarterly Returns (%)')
        plt.axhline(0, linewidth=1)
        plt.tight_layout()
        plt.show()

        (monthly.tail(60) * 100).plot(kind='bar', figsize=(14, 4), title='Monthly Returns - last 60 months (%)')
        plt.axhline(0, linewidth=1)
        plt.tight_layout()
        plt.show()

    return {
        'series': portfolio,
        'monthly': monthly,
        'quarterly': quarterly,
        'annual': annual
    }