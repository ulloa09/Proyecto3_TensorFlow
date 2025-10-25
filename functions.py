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
    # Evitar división por cero
    close_safe = df["Close"].replace(0, np.nan)
    df["fwd_ret"] = df["Close"].shift(-horizon) / close_safe - 1
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
    
    fwd_ret_valid = ref_df["fwd_ret"].dropna()
    if fwd_ret_valid.empty:
        print("Advertencia: 'fwd_ret' no tiene datos válidos para calcular cuantiles.")
        return 0.0, 0.0

    upper_thr = fwd_ret_valid.quantile(upper_q)
    lower_thr = fwd_ret_valid.quantile(lower_q)
    
    print(f"Umbrales calculados: Lower (q={lower_q}): {lower_thr:.5f}, Upper (q={upper_q}): {upper_thr:.5f}")
    return lower_thr, upper_thr

def label_by_thresholds(df: pd.DataFrame, lower_thr: float, upper_thr: float) -> pd.DataFrame:
    """
    Etiqueta -1/0/1 usando umbrales fijos (sin recalcular percentiles).
    Etiquetas: 0 (Venta/Short), 1 (Hold), 2 (Compra/Long)
    """
    df = df.copy()
    if "fwd_ret" not in df.columns:
        print("Advertencia: 'fwd_ret' no encontrado en label_by_thresholds. Asignando 'target' como 1 (Hold).")
        df["target"] = 1
        return df

    # Iniciar todo como Hold (señal 1)
    df["target"] = 1
    # Etiquetar Compra (señal 2)
    df.loc[df["fwd_ret"] > upper_thr, "target"] = 2
    # Etiquetar Venta (señal 0)
    df.loc[df["fwd_ret"] < lower_thr, "target"] = 0

    print(f"Etiquetas generadas (0=S, 1=H, 2=L): {len(df)}")
    if not df.empty:
        print(f"Distribución:\n{np.round(df.target.value_counts(normalize=True).sort_index(), 5)}")
    else:
        print("Distribución: N/A (DataFrame vacío)")
    return df


def one_hot_encode(y, num_classes=None):
    """
    Versión propia de to_categorical.
    Convierte un vector de etiquetas (0,1,2,...) en una matriz one-hot.
    """
    y = np.array(y, dtype=int)
    min_y = np.min(y) if y.size > 0 else 0
    if min_y < 0:
        raise ValueError(f"Las etiquetas deben ser >= 0 para one-hot encoding, se encontró: {min_y}")
    
    max_val = np.max(y) if y.size > 0 else -1
    if num_classes is None:
        num_classes = max_val + 1
    elif num_classes <= max_val:
        num_classes = max_val + 1
    
    num_classes = max(1, num_classes) # Ensure at least 1 class
    
    one_hot = np.zeros((y.size, num_classes))
    if y.size > 0:
        one_hot[np.arange(y.size), y] = 1
    return one_hot

def prepare_xy(train_df, val_df, test_df, exclude_cols=None):
    """
    Prepara X (features) e y (etiquetas) para entrenamiento, validación y prueba.
    Convierte las etiquetas en formato one-hot para usarlas en modelos de clasificación.
    Parámetros:
        train_df, val_df, test_df: DataFrames con los datos ya escalados y etiquetados.
    exclude_cols: columnas que no se usarán como features.

    Devuelve:
        X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, feature_cols
    """
    if exclude_cols is None:
        # Usamos la lista de exclusión base de tu compañero
        base_exclude = ["Date", "fwd_ret", "target", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Datetime"]
        exclude_cols = [col for col in base_exclude if col in train_df.columns]

    # Selecciona columnas de features (todas menos las excluidas)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No se encontraron columnas de features después de excluir.")
        
    print(f"Usando {len(feature_cols)} features: {feature_cols[:5]}...")

    # Features → arrays numpy
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)

    # Etiquetas → vectores de clase (0, 1, 2)
    y_train = train_df["target"].astype(int).to_numpy()
    y_val   = val_df["target"].astype(int).to_numpy()
    y_test  = test_df["target"].astype(int).to_numpy()

    # Convierte a formato one-hot (para categorical_crossentropy)
    y_train_oh = one_hot_encode(y_train, num_classes=3)
    y_val_oh = one_hot_encode(y_val, num_classes=3)
    y_test_oh = one_hot_encode(y_test, num_classes=3)

    print("Shapes:")
    print("X_train:", X_train.shape, "| y_train:", y_train_oh.shape)
    print("X_val:", X_val.shape, "| y_val:", y_val_oh.shape)
    print("X_test:", X_test.shape, "| y_test:", y_test_oh.shape)

    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, feature_cols

def compute_class_weights(y_train):
    """
    Calcula pesos balanceados para las clases del conjunto de entrenamiento.
    Útil cuando hay desbalance (por ejemplo, muchas etiquetas de clase 1 y pocas de 0 o 2).
    Parámetros:
        y_train : np.ndarray
            Etiquetas del conjunto de entrenamiento (pueden ser one-hot o enteros).
    Devuelve:
        class_weights : dict
            Diccionario con los pesos de clase {0: peso0, 1: peso1, 2: peso2}.
    """
    import numpy as np

    # Si viene one-hot (matriz), la convertimos a enteros
    if y_train.ndim > 1:
        y_train_int = np.argmax(y_train, axis=1)
    else:
        y_train_int = y_train.astype(int)

    if y_train_int.size == 0:
        print("Advertencia: y_train vacío en compute_class_weights.")
        return {0: 1.0, 1: 1.0, 2: 1.0}

    # Calcula pesos balanceados
    classes = np.unique(y_train_int)
    
    # Aseguramos que todas las clases (0, 1, 2) sean consideradas
    all_classes = np.array([0, 1, 2])
    
    weights = compute_class_weight(class_weight="balanced", classes=all_classes, y=y_train_int)
    class_weights = {int(c): float(w) for c, w in zip(all_classes, weights)}

    print("Pesos de clase calculados:")
    for k, v in class_weights.items():
        print(f"  Clase {k}: {v:.3f}")
    return class_weights