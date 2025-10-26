import pandas as pd
from preprocess_features import fechas, fit_scalers, apply_scalers
from features import generate_features
from functions import make_forward_return, compute_thresholds, label_by_thresholds, prepare_xy
from split import split_dfs

def load_and_prepare_data(csv_path: str, horizon: int, lower_q: float, upper_q: float, split_ratios: dict):
    """
    Encapsula la carga, generación de features, etiquetado y división de datos.
    """
    # Carga de datos
    datos = pd.read_csv(csv_path)
    # Creación de fechas
    datos = fechas(datos)
    # Creación de features
    datos = generate_features(datos)
    # Calcular rendimiento futuro (forward return)
    datos = make_forward_return(datos, horizon=horizon)
    # Definir umbrales dinámicos (percentiles)
    lower_thr, upper_thr = compute_thresholds(datos, lower_q=lower_q, upper_q=upper_q)
    # Asignar etiquetas
    datos = label_by_thresholds(datos, lower_thr, upper_thr)
    # Drop de NAs
    data = datos.copy().dropna()
    # Split de datos
    train_df, test_df, validation_df = split_dfs(
        data, 
        train=split_ratios['train'], 
        test=split_ratios['test'], 
        validation=split_ratios['validation']
    )
    
    return train_df, test_df, validation_df

def scale_data(train_df, test_df, validation_df):
    """
    Entrena escaladores con train_df y los aplica a los tres conjuntos.
    """
    # Creación y entrenamiento de escaladores
    min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler = fit_scalers(train_df)

    # === Aplicar los escaladores a los tres conjuntos ===
    train_scaled = apply_scalers(train_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    test_scaled = apply_scalers(test_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    val_scaled = apply_scalers(validation_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
    print("\n \nEscaladores aplicados a todos los conjuntos correctamente.")

    # Conservar fecha (necesaria backtesting)
    train_scaled["Date"] = train_df["Date"]
    test_scaled["Date"] = test_df["Date"]
    val_scaled["Date"] = validation_df["Date"]

    # === Guardar resultados en disco para revisión rápida ===
    train_scaled.to_csv("data/train_scaled.csv", index=False)
    test_scaled.to_csv("data/test_scaled.csv", index=False)
    val_scaled.to_csv("data/val_scaled.csv", index=False)
    
    return train_scaled, test_scaled, val_scaled

def prepare_xy_data(train_scaled, val_scaled, test_scaled):
    """
    Separa los DataFrames escalados en X (features) e y (etiquetas).
    """
    # Separación en x, y para train, test y validation
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy(train_scaled, val_scaled, test_scaled)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols