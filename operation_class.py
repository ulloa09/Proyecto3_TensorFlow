from dataclasses import dataclass

@dataclass
class Operation:
    """
    A data class to store information about an active trading operation.
    """
    time: str        # Entry time/date
    price: float     # Entry price
    stop_loss: float # Price level for stop loss
    take_profit: float# Price level for take profit
    n_shares: int    # Number of shares
    type: str        # 'LONG' or 'SHORT'