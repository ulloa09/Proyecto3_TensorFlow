import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional # Importar Optional

def plot_portfolio_train(portfolio_series: pd.Series, model_name: str):
    
    """
    1. Grafica el comportamiento del portafolio en el periodo de TRAIN.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Valor del Portafolio (Train) - Modelo {model_name}', color='cyan')
    plt.ylabel('Valor del Portafolio ($)')
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_test(
    portfolio_series: pd.Series,
    model_name: str,
    breakpoint_date: Optional[pd.Timestamp] = None # Asegúrate que este argumento exista
):
    """
    2. Grafica el comportamiento del portafolio en el periodo de TEST.
    Incluye una línea vertical en el punto de quiebre del drift.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Valor del Portafolio (Test) - Modelo {model_name}', color='orange')

    # Añadir línea vertical si breakpoint_date se proporciona
    if breakpoint_date is not None:
        plt.axvline(
            x=breakpoint_date,
            color='red',
            linestyle='--',
            linewidth=2,
         
            label=f'Punto de Drift ({breakpoint_date.date()})'
        )
        plt.legend() # Mostrar la leyenda solo si hay línea

    plt.ylabel('Valor del Portafolio ($)')
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_validation(
    portfolio_series: pd.Series,
    model_name: str,
    breakpoint_date: Optional[pd.Timestamp] = None # Asegúrate que este argumento exista
):
    """
    3. Grafica el comportamiento del portafolio en el periodo de VALIDATION.
    Incluye una línea vertical en el punto de quiebre del drift.
    """
    plt.figure(figsize=(12, 6))
    portfolio_series.plot(title=f'Valor del Portafolio (Validation) - Modelo {model_name}', color='lightgreen')

    # Añadir línea vertical si breakpoint_date se proporciona
    if breakpoint_date is not None:
        plt.axvline(
            x=breakpoint_date,
            color='red',
            linestyle='--',
            linewidth=2,
         
            label=f'Punto de Drift ({breakpoint_date.date()})'
        )
        plt.legend() # Mostrar la leyenda solo si hay línea

    plt.ylabel('Valor del Portafolio ($)')
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_portfolio_combined(port_train: pd.Series, port_test: pd.Series, port_val: pd.Series, model_name: str):
    """
    4. Combina y grafica el comportamiento del portafolio en los 3 periodos.
    """
    # Concatenar las series. El índice Datetime se encargará del orden.
    combined_portfolio = pd.concat([port_train, port_test, port_val]).sort_index()

    # Eliminar duplicados (si el índice inicial se repite)
    combined_portfolio = combined_portfolio[~combined_portfolio.index.duplicated(keep='last')]

    plt.figure(figsize=(14, 7))
    combined_portfolio.plot(title=f'Valor del Portafolio (Combinado) - Modelo {model_name}', color='blue')

    # Añadir líneas verticales para marcar las zonas
    if not port_train.empty:
        plt.axvline(port_train.index.min(), color='gray', linestyle='--', label='Inicio Train')
    if not port_test.empty:
        plt.axvline(port_test.index.min(), 

color='red', linestyle='--', label='Inicio Test')
    if not port_val.empty:
        plt.axvline(port_val.index.min(), color='green', linestyle='--', label='Inicio Validation')

    plt.ylabel('Valor del Portafolio ($)')
    plt.xlabel('Fecha')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()