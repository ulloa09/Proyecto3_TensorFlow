from dataclasses import dataclass
import pandas as pd

@dataclass
class Operation:
    """
    Representa una operación de trading abierta.
    """
    time: any # Timestamp de apertura
    price: float # Precio de apertura
    stop_loss: float
    take_profit: float
    n_shares: int
    type: str # 'LONG' o 'SHORT'


def portfolio_value(cash: float,
                        long_ops: list[Operation], 
                        short_ops: list[Operation], 
                        current_price: float, 
                        n_shares: float) -> float:
    """
    Calcula el valor total actual del portafolio (equity).
    """
    port_val = cash

    # Sumar el valor de las posiciones largas abiertas
    for position in long_ops:
        port_val += current_price * position.n_shares

    # Sumar el PnL de las posiciones cortas abiertas
    for position in short_ops:
        # El PnL de un short es el (precio_entrada - precio_actual)
        pnl = (position.price - current_price) * position.n_shares
        # Sumamos el PnL al 'port_val'. No sumamos el 'cash' inicial
        # porque el 'cash' ya lo tiene (o se usó para colateral).
        # Esta es una forma simplificada de equity.
        port_val += pnl 

    return port_val