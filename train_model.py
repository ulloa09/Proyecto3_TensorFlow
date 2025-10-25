import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from split import split_dfs
from features import generate_features
from functions import make_forward_return, compute_thresholds, label_by_thresholds, prepare_xy, compute_class_weights
from preprocess_features import fechas, fit_scalers, apply_scalers
from drift_analisis import revisar_drift_features
from cnn_model import build_cnn_model
from mlp_model import build_mlp_model


def run_training_pipeline():
    """
    Función principal para el pipeline de entrenamiento.
    Usa lógica de cuantiles (q=0.33, q=0.66) y horizonte 1.
    """
    # --- 1. Carga y Features ---
    print("Iniciando carga y generación de features...")
    try: 
        datos = pd.read_csv('data/wynn_daily_15y.csv')
    except FileNotFoundError: 
        print("Error: 'data/wynn_daily_15y.csv' no encontrado. Ejecuta data_download.py")
        return
    
    
    datos = fechas(datos) 
    datos = generate_features(datos) 
    
    if datos.empty: 
        print("Error: DataFrame vacío después de generate_features.")
        return

    # --- 2. Lógica de Target (Quantiles) ---
    print("Calculando forward returns y etiquetas (Quantiles)...")
    horizon = 5
    datos = make_forward_return(datos, horizon) 
    
    if datos.empty or 'fwd_ret' not in datos.columns or datos['fwd_ret'].isnull().all():
        print(f"Error: DataFrame vacío o sin 'fwd_ret' válidos después de make_forward_return(horizon={horizon}).")
        return

    # Dividir ANTES de calcular cuantiles para evitar look-ahead
    train_df_raw, test_df_raw, validation_df_raw = split_dfs(datos, train=60, test=20, validation=20)

    LOWER_Q, UPPER_Q = 0.2, 0.8
    print(f"Usando cuantiles balanceados: Q_lower={LOWER_Q}, Q_upper={UPPER_Q}")
    lower_thr, upper_thr = compute_thresholds(train_df_raw, lower_q=LOWER_Q, upper_q=UPPER_Q)
    
    # Aplicar umbrales fijos a todos los sets
    train_df = label_by_thresholds(train_df_raw, lower_thr, upper_thr)
    test_df = label_by_thresholds(test_df_raw, lower_thr, upper_thr)
    validation_df = label_by_thresholds(validation_df_raw, lower_thr, upper_thr)

    # --- 3. Definir Columnas y DropNA ---
    base_exclude = ["Date", "fwd_ret", "target", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Datetime"]
    
    feature_cols = [c for c in train_df.columns if c not in base_exclude]
    cols_to_check_for_nan = feature_cols + ['target'] + ["Open", "High", "Low", "Close", "Volume"]
    
    print("Eliminando filas con NaNs en columnas de features, OHLCV o target...")
    train_df = train_df.dropna(subset=cols_to_check_for_nan)
    test_df = test_df.dropna(subset=cols_to_check_for_nan)
    validation_df = validation_df.dropna(subset=cols_to_check_for_nan)

    if train_df.empty: 
        print("Error: train_df vacío después de dropna. No se puede entrenar.")
        return
    print(f"Tamaños después de dropna: Train={len(train_df)}, Test={len(test_df)}, Val={len(validation_df)}")

    # --- 4. Entrenar y Aplicar Scalers ---
    print("Entrenando scalers (4-scalers) en train_df...")
    
    mm_scaler, rb_scaler, st_scaler, ohlcv_scaler = fit_scalers(train_df)
    
    if any(s is None for s in [mm_scaler, rb_scaler, st_scaler, ohlcv_scaler]): 
        print("Error: Falló entrenamiento de scalers (uno o más son None).")
        return 

    print("Escaladores entrenados y guardados.")
    
    train_scaled = apply_scalers(train_df.copy(), mm_scaler, rb_scaler, st_scaler, ohlcv_scaler)
    test_scaled = apply_scalers(test_df.copy(), mm_scaler, rb_scaler, st_scaler, ohlcv_scaler)
    val_scaled = apply_scalers(validation_df.copy(), mm_scaler, rb_scaler, st_scaler, ohlcv_scaler)
    
    print("Escaladores aplicados.")

    # Guardar CSVs escalados (necesarios para run_backtest.py)
    print("Guardando datos escalados en CSV...")
    try:
        train_scaled.to_csv("data/train_scaled.csv", index=False)
        test_scaled.to_csv("data/test_scaled.csv", index=False)
        val_scaled.to_csv("data/val_scaled.csv", index=False)
        print("Datos escalados guardados.")
    except Exception as e:
        print(f"Error al guardar los CSV escalados: {e}")
        return

    # --- 5. ANÁLISIS DE DATA DRIFT ---
    print("\n--- Iniciando Análisis de Data Drift ---")
    drift_alpha = 0.05 
    feature_cols_final = [c for c in train_scaled.columns if c not in base_exclude]
    
    print("Comparando Test vs Train Features...")
    _ = revisar_drift_features(train_scaled, test_scaled, feature_cols_final, alpha=drift_alpha)
    
    print("\nComparando Validation vs Train Features...")
    _ = revisar_drift_features(train_scaled, val_scaled, feature_cols_final, alpha=drift_alpha)
    print("--- Análisis de Data Drift Completado ---")
    
    # --- 6. Preparar Datos X, y ---
    print("\nPreparando datos X/y para Keras (etiquetas 0, 1, 2)...")
    try:
        # prepare_xy aún devuelve one-hot (y_train_oh)
        X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, _ = prepare_xy(
             train_scaled, val_scaled, test_scaled
        )
    except ValueError as e: 
        print(f"Error en prepare_xy: {e}.")
        return 

    # Convertir etiquetas one-hot (ej. [0,0,1]) a enteros (ej. 2) 
    # para sparse_categorical_crossentropy
    print("Convirtiendo etiquetas a formato de enteros (sparse)...")
    y_train_int = np.argmax(y_train_oh, axis=1)
    y_val_int = np.argmax(y_val_oh, axis=1)
    
    print("Calculando pesos de clase (class weights)...")
    # compute_class_weights puede manejar tanto one-hot como enteros, 
    # así que podemos seguir pasándole y_train_oh (o y_train_int)
    class_weights_dict = compute_class_weights(y_train_int)

    # --- 7. Definir Configuraciones y Callbacks ---
    # (Usando las configuraciones simples de la iteración anterior)
    print("Usando configuraciones de modelo MÁS SIMPLES para evitar overfitting.")
    configuraciones_mlp = [ {"dense_layers": 1, "dense_units": 64}, {"dense_layers": 2, "dense_units": 32}, ]
    configuraciones_cnn = [ {"conv_layers": 2, "conv_filters": 32, "dense_units": 64}, {"conv_layers": 2, "conv_filters": 16, "dense_units": 32}, ]

    callback_parada = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mlflow.tensorflow.autolog()

    # --- 8. Entrenar Modelos MLP ---
    print("\n--- Entrenando Modelos MLP ---")
    mlflow.set_experiment("MLP_Models_WYNN_Quantile_H1_Simple") 
    forma_entrada_mlp = X_train.shape[1]
    for config in configuraciones_mlp:
        nombre_run = f"MLP_layers{config['dense_layers']}_units{config['dense_units']}_relu_Weighted"
        with mlflow.start_run(run_name=nombre_run) as run:
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config)
            mlflow.log_param("activation", "relu")
            mlflow.log_param("optimizer", "adam")
            mlflow.set_tag("tipo_modelo", "MLP")
            
            modelo_mlp = build_mlp_model(forma_entrada_mlp, config)
            
            # Pasar y_train_int y y_val_int a model.fit()
            modelo_mlp.fit( X_train, y_train_int, validation_data=(X_val, y_val_int), 
                            epochs=50, batch_size=32, callbacks=[callback_parada], 
                            verbose=1, class_weight=class_weights_dict 
            ) 
            
            mlflow.tensorflow.log_model(modelo_mlp, "model", registered_model_name=nombre_run)

    # --- 9. Entrenar Modelos CNN ---
    print("\n--- Entrenando Modelos CNN ---")
    mlflow.set_experiment("CNN_Models_WYNN_Quantile_H1_Simple") 
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)
    
    forma_entrada_cnn = X_train_cnn.shape[1]
    for config in configuraciones_cnn:
        nombre_run = f"CNN_conv{config['conv_layers']}_filters{config['conv_filters']}_dense{config['dense_units']}_tanh_Weighted"
        with mlflow.start_run(run_name=nombre_run) as run:
             print(f"Entrenando: {nombre_run}")
             mlflow.log_params(config)
             mlflow.log_param("activation", "tanh")
             mlflow.log_param("optimizer", "adam")
             mlflow.set_tag("tipo_modelo", "CNN")
             
             modelo_cnn = build_cnn_model(forma_entrada_cnn, config)
             
             # Pasar y_train_int y y_val_int a model.fit()
             modelo_cnn.fit( X_train_cnn, y_train_int, validation_data=(X_val_cnn, y_val_int), 
                             epochs=50, batch_size=32, callbacks=[callback_parada], 
                             verbose=1, class_weight=class_weights_dict
             )
             
             mlflow.tensorflow.log_model(modelo_cnn, "model", registered_model_name=nombre_run)

    print("\nEntrenamiento de modelos DL finalizado.")

# --- Punto de entrada ---
if __name__ == "__main__":
    print("Iniciando script de entrenamiento de modelos...")
    run_training_pipeline()
    print("Script de entrenamiento completado.")