import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from operation_class import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price: float, n_shares: int)-> float:

    val = cash

    # Agregar posiciones largas
    for position in long_ops:
        pnl = current_price * position.n_shares
        val += pnl

    # Agregar posiciones cortas
    for position in short_ops:
        pnl = (position.price - current_price) * position.n_shares
        val += pnl

    return val

def make_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Agrega 'fwd_ret' = Close.shift(-horizon)/Close - 1 y recorta las últimas 'horizon' filas.
    """
    df = df.copy()
    df["fwd_ret"] = df["Close"].shift(-horizon) / df["Close"] - 1
    if horizon > 0:
        df = df.iloc[:-horizon]
    return df

def compute_thresholds(ref_df: pd.DataFrame, lower_q: float, upper_q: float) -> tuple[float, float]:
    ref_df = ref_df.copy()
    """
    Calcula umbrales (lower_thr, upper_thr) a partir de ref_df['fwd_ret'] (ignora NaN).
    """
    if "fwd_ret" not in ref_df.columns:
        raise KeyError("compute_thresholds requiere la columna 'fwd_ret' en ref_df.")
    upper_thr = ref_df["fwd_ret"].quantile(upper_q)
    lower_thr = ref_df["fwd_ret"].quantile(lower_q)
    return lower_thr, upper_thr

def label_by_thresholds(df: pd.DataFrame, lower_thr: float, upper_thr: float) -> pd.DataFrame:
    """
    Etiqueta -1/0/1 usando umbrales fijos (sin recalcular percentiles).
    """
    df = df.copy()
    if "fwd_ret" not in df.columns:
        raise KeyError("label_by_thresholds requiere la columna 'fwd_ret' en df.")
    df["target"] = 1
    df.loc[df["fwd_ret"] > upper_thr, "target"] = 2
    df.loc[df["fwd_ret"] < lower_thr, "target"] = 0

    print(f"Etiquetas generadas: {len(df)} \n"
          f"Total por clase:{np.round(df.target.value_counts() / len(df),5)} \n")
    return df

def prepare_xy(train_df, val_df, test_df, exclude_cols=None):
    """
    Prepara X (features) e y (etiquetas enteras) para entrenamiento, validación y prueba.
    Adaptado para modelos con sparse_categorical_crossentropy (como CNN 1D actual).
    """

    if exclude_cols is None:
        exclude_cols = ["Date", "fwd_ret", "target"]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Features a float32
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)

    # Etiquetas como enteros (no one-hot)
    y_train = train_df["target"].astype(int).to_numpy()
    y_val   = val_df["target"].astype(int).to_numpy()
    y_test  = test_df["target"].astype(int).to_numpy()

    print("Shapes:")
    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_val:", X_val.shape, "| y_val:", y_val.shape)
    print("X_test:", X_test.shape, "| y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def compute_class_weights(y_train):
    """
    Calcula pesos balanceados para clases en formato entero (0, 1, 2).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print("Pesos de clase calculados:")
    for k, v in class_weights.items():
        print(f"  Clase {k}: {v:.3f}")
    return class_weights