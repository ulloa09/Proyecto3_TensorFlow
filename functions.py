import numpy as np
import pandas as pd
# Importar directamente class_weight
from sklearn.utils import class_weight

# --- make_forward_return (Sin cambios) ---
def make_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Agrega 'fwd_ret' y recorta las últimas 'horizon' filas. Maneja df vacío.
    """
    df = df.copy()
    if df.empty or len(df) <= horizon:
        print("Advertencia: DataFrame demasiado corto o vacío para calcular fwd_ret.")
        if not df.empty:
             df["fwd_ret"] = np.nan
             return df.iloc[0:0]
        else: return df
    # Evitar división por cero
    close_safe = df["Close"].replace(0, np.nan)
    df["fwd_ret"] = df["Close"].shift(-horizon) / close_safe - 1
    if horizon > 0: df = df.iloc[:-horizon]
    return df

def label_by_fixed_threshold(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Etiqueta -1 (Short), 0 (Hold), 1 (Long) usando un umbral fijo alpha.
    """
    df = df.copy()
    if df.empty:
        print("Advertencia: DataFrame vacío para label_by_fixed_threshold.")
        if "target" not in df.columns: df["target"] = pd.Series(dtype=int)
        return df

    if "fwd_ret" not in df.columns:
        print("Advertencia: Columna 'fwd_ret' no encontrada en label_by_fixed_threshold. Añadiendo NaNs y etiqueta Hold.")
        df['fwd_ret'] = np.nan
        df["target"] = 0 # Asignar Hold por defecto
        return df

    # Iniciar todo como Hold (señal 0)
    df["target"] = 0
    # Etiquetar Long (señal 1) donde el retorno futuro supera alpha
    df.loc[df["fwd_ret"] > alpha, "target"] = 1
    # Etiquetar Short (señal -1) donde el retorno futuro es menor que -alpha
    df.loc[df["fwd_ret"] < -alpha, "target"] = -1

    print(f"Etiquetas generadas (-1=S, 0=H, 1=L) con alpha={alpha}: {len(df)}")
    if not df.empty: print(f"Distribución:\n{np.round(df.target.value_counts(normalize=True).sort_index(), 5)}")
    else: print("Distribución: N/A (DataFrame vacío)")
    return df

# --- one_hot_encode (Sin cambios) ---
def one_hot_encode(y, num_classes=None):
    y = np.array(y, dtype=int)
    min_y = np.min(y) if y.size > 0 else 0
    if min_y < 0:
        raise ValueError(f"Las etiquetas deben ser >= 0 para one-hot encoding, se encontró: {min_y}")
    max_val = np.max(y) if y.size > 0 else -1
    if num_classes is None: num_classes = max_val + 1
    elif num_classes <= max_val: num_classes = max_val + 1
    num_classes = max(1, num_classes) # Ensure at least 1 class
    one_hot = np.zeros((y.size, num_classes))
    if y.size > 0: one_hot[np.arange(y.size), y] = 1
    return one_hot

# --- prepare_xy (Mapeo de -1,0,1 a 0,1,2 - Sin cambios) ---
def prepare_xy(train_df, val_df, test_df, exclude_cols=None):
    """
    Prepara X (features) e y (etiquetas one-hot 0,1,2) para Keras.
    Mapea internamente target de -1,0,1 a 0,1,2.
    """
    if exclude_cols is None:
        base_exclude = ["Date", "fwd_ret", "target", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Datetime"]
        exclude_cols = [col for col in base_exclude if col in train_df.columns]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    if not feature_cols: raise ValueError("No se encontraron columnas de features.")
    print(f"Usando {len(feature_cols)} features: {feature_cols[:5]}...")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test  = test_df[feature_cols].to_numpy(dtype=np.float32)

    y_train_orig = train_df["target"].astype(int).to_numpy()
    y_val_orig   = val_df["target"].astype(int).to_numpy()
    y_test_orig  = test_df["target"].astype(int).to_numpy()

    # Mapeo a 0, 1, 2: (-1 -> 0, 0 -> 1, 1 -> 2)
    y_train_mapped = y_train_orig + 1
    y_val_mapped   = y_val_orig + 1
    y_test_mapped  = y_test_orig + 1

    y_train_oh = one_hot_encode(y_train_mapped, num_classes=3)
    y_val_oh = one_hot_encode(y_val_mapped, num_classes=3)
    y_test_oh = one_hot_encode(y_test_mapped, num_classes=3)

    print("Shapes:")
    print("X_train:", X_train.shape, "| y_train_oh:", y_train_oh.shape)
    print("X_val:", X_val.shape, "| y_val_oh:", y_val_oh.shape)
    print("X_test:", X_test.shape, "| y_test_oh:", y_test_oh.shape)

    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, feature_cols

# --- compute_class_weights (Sin cambios, acepta y mapeada 0,1,2) ---
def compute_class_weights(y_train):
    """
    Calcula pesos balanceados para las clases (0, 1, 2).
    """
    if y_train.ndim > 1 and y_train.shape[1] > 1: y_train_int = np.argmax(y_train, axis=1)
    elif y_train.ndim == 1: y_train_int = y_train.astype(int)
    else: raise ValueError("Formato de y_train no reconocido.")
    if y_train_int.size == 0: print("Advertencia: y_train vacío."); return {0: 1.0, 1: 1.0, 2: 1.0}
    unique_classes = np.unique(y_train_int)
    # if not np.all(np.isin(unique_classes, [0, 1, 2])): print(f"Advertencia: Clases inesperadas: {unique_classes}.")
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train_int)
    class_weights_dict = {int(c): float(w) for c, w in zip(np.array([0, 1, 2]), weights)}
    print("Pesos de clase calculados (para 0, 1, 2):")
    for k in [0, 1, 2]: print(f"  Clase {k}: {class_weights_dict.get(k, 0.0):.3f}")
    return class_weights_dict