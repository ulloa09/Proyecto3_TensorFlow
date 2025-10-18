import yfinance as yf
import numpy as np
import pandas as pd

import pandas as pd
import yfinance as yf

# Descarga directa y limpia desde cero
data = yf.download(
    tickers="WYNN",
    start="2010-01-01",
    end="2025-10-17",
    interval="1d",
    auto_adjust=False,
    progress=False
)

# Aplana si existe un nivel superior de columnas (por el ticker)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Reiniciar índice
data = data.reset_index()

# Reordenar columnas y eliminar cualquier encabezado falso
data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
data['Date'] = pd.to_datetime(data['Date'])

# Guardar limpio
data.to_csv('wynn_daily_15y.csv', index=False)

print(data.head())
print(f"Datos descargados: {len(data)} registros entre {data['Date'].min()} y {data['Date'].max()}")
