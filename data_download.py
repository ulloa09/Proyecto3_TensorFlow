import yfinance as yf
import numpy as np
import pandas as pd

# Descarga de datos históricos diarios de los últimos 15 años
data = yf.download(
    tickers='WYNN',
    start='2010-01-01',
    end='2025-10-17',
    interval='1d'         # datos diarios
)

# Limpieza básica
data = data.dropna()
data.to_csv('wynn_daily_15y.csv')  # copia local
print(data.head())
print(f"Datos descargados: {len(data)} registros entre {data.index.min()} y {data.index.max()}")