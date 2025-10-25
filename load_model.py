import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from P2_split import split_dfs
from features import generate_features
from functions import make_forward_return, label_by_fixed_threshold, prepare_xy
from preprocess_features import fechas, fit_scalers, apply_scalers
from drift_detection import analyze_feature_drift # Importar la nueva función


def build_cnn_model(input_shape, params):
    model = Sequential(); model.add(Input(shape=(input_shape, 1)))
    num_filters = params.get("conv_filters", 32); conv_layers = params.get("conv_layers", 2); activation = 'tanh'
    for _ in range(conv_layers): model.add(Conv1D(filters=num_filters, kernel_size=3, activation=activation, padding='same')); model.add(MaxPooling1D(pool_size=2)); num_filters *= 2
    dense_units = params.get("dense_units", 64); model.add(Flatten()); model.add(Dense(dense_units, activation=activation)); model.add(Dense(3, activation='softmax'))
    optimizer = "adam"; model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']); return model

def build_mlp_model(input_shape, params):
    model = Sequential(); model.add(Input(shape=(input_shape,))); dense_layers = params.get("dense_layers", 2); units = params.get("dense_units", 128); activation = 'relu'
    for _ in range(dense_layers): model.add(Dense(units, activation=activation))
    model.add(Dense(3, activation='softmax')); optimizer = "adam"; model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']); return model


def run_models():
    """
    Función principal. Usa horizonte 1 día y umbral fijo alpha=0.05 para targets.
    SIN class_weight. VWAP eliminado. Dropna específico. Incluye Data Drift Analysis.
    """
    # --- 1. Carga, Features, FwdReturn ---
    print("Iniciando carga y generación de features...")
    try: datos = pd.read_csv('data/wynn_daily_15y.csv')
    except FileNotFoundError: print("Error: 'data/wynn_daily_15y.csv' no encontrado."); return
    fechas(datos)
    datos = generate_features(datos)
    if datos.empty: print("Error: DataFrame vacío después de generate_features."); return
    print("Calculando forward returns...")
    horizon = 1; alpha = 0.05
    datos = make_forward_return(datos, horizon)
    if datos.empty: print(f"Error: DataFrame vacío después de make_forward_return(horizon={horizon})."); return

    # --- 2. Split ---
    print("Dividiendo datos en train/test/validation...")
    train_df, test_df, validation_df = split_dfs(datos, train=60, test=20, validation=20)
    if train_df.empty or test_df.empty or validation_df.empty: print("Error: DataFrames vacíos después de split_dfs."); return
    print(f"Tamaños antes de dropna: Train={len(train_df)}, Test={len(test_df)}, Val={len(validation_df)}")

    # --- 3. Etiquetar con Umbral Fijo ---
    print(f"Etiquetando con umbral fijo alpha={alpha}...")
    print("Etiquetando train_df..."); train_df = label_by_fixed_threshold(train_df, alpha)
    print("Etiquetando test_df..."); test_df = label_by_fixed_threshold(test_df, alpha)
    print("Etiquetando validation_df..."); validation_df = label_by_fixed_threshold(validation_df, alpha)

    # --- 4. Definir Columnas de Features y Target ANTES de dropna ---
    exclude_cols = ["Date", "fwd_ret", "target", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Datetime"]
    potential_feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    feature_cols = [col for col in potential_feature_cols if train_df[col].notna().any()] # Columnas que SÍ vamos a usar
    print(f"Columnas identificadas como features válidas ({len(feature_cols)}): {feature_cols[:5]}...")
    cols_to_check_for_nan = feature_cols + ['target']

    # --- 5. Eliminar NaNs Específicamente ---
    print("Eliminando filas con NaNs en columnas de features o target...")
    rows_before_train = len(train_df); train_df = train_df.dropna(subset=cols_to_check_for_nan)
    print(f"Train DF: {rows_before_train - len(train_df)} filas eliminadas.")
    rows_before_test = len(test_df); test_df = test_df.dropna(subset=cols_to_check_for_nan)
    print(f"Test DF: {rows_before_test - len(test_df)} filas eliminadas.")
    rows_before_val = len(validation_df); validation_df = validation_df.dropna(subset=cols_to_check_for_nan)
    print(f"Validation DF: {rows_before_val - len(validation_df)} filas eliminadas.")
    if train_df.empty: print("Error: train_df vacío después de dropna específico."); return
    print(f"Tamaños después de dropna: Train={len(train_df)}, Test={len(test_df)}, Val={len(validation_df)}")

    # --- 6. Entrenar y Aplicar Scalers ---
    print("Entrenando scalers en train_df...")
    scalers = fit_scalers(train_df)
    if any(s is None for s in scalers): print("Error: Falló entrenamiento de scalers."); return
    print("Escaladores entrenados y guardados.")
    train_scaled = apply_scalers(train_df.copy())
    test_scaled = apply_scalers(test_df.copy())
    val_scaled = apply_scalers(validation_df.copy())
    print("Escaladores aplicados.")
    if train_scaled.empty or test_scaled.empty or val_scaled.empty: print("Error: DataFrames vacíos después de escalar."); return

    # --- 7. ANÁLISIS DE DATA DRIFT ---
    # Usar las feature_cols definidas antes y los DFs escalados
    print("\n--- Iniciando Análisis de Data Drift ---")
    drift_alpha = 0.05 # Nivel de significancia para el drift

    # Comparar Test vs Train
    print("Comparando Test vs Train Features...")
    drift_test_results = analyze_feature_drift(train_scaled, test_scaled, feature_cols, alpha=drift_alpha)
    
    # Comparar Validation vs Train
    print("\nComparando Validation vs Train Features...")
    drift_val_results = analyze_feature_drift(train_scaled, val_scaled, feature_cols, alpha=drift_alpha)
    print("--- Análisis de Data Drift Completado ---")
    

    # --- 8. Preparar Datos X, y (Ahora paso 8) ---
    print("\nPreparando datos X/y para Keras...")
    try:
         # Usar feature_cols ya definida y filtrada
         X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, _ = prepare_xy(
              train_scaled, val_scaled, test_scaled, exclude_cols= [c for c in train_scaled.columns if c not in feature_cols + ['target']]
         )
    except ValueError as e: print(f"Error en prepare_xy: {e}."); return

    # --- 9. Definir Configuraciones y Callbacks (Ahora paso 9) ---
    configuraciones_mlp = [ {"dense_layers": 3, "dense_units": 150}, {"dense_layers": 2, "dense_units": 200}, ]
    configuraciones_cnn = [ {"conv_layers": 3, "conv_filters": 64, "dense_units": 100}, {"conv_layers": 4, "conv_filters": 16, "dense_units": 50}, ]
    callback_parada = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    mlflow.tensorflow.autolog()

    # --- 10. Entrenar Modelos MLP (Ahora paso 10) ---
    print("\n--- Entrenando Modelos MLP ---")
    mlflow.set_experiment("MLP_Models_WYNN_FixedAlpha_H1_NoWeight")
    forma_entrada_mlp = X_train.shape[1]
    for config in configuraciones_mlp:
        nombre_run = f"MLP_layers{config['dense_layers']}_units{config['dense_units']}_relu"
        with mlflow.start_run(run_name=nombre_run) as run:
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config); mlflow.log_param("activation", "relu"); mlflow.log_param("optimizer", "adam"); mlflow.set_tag("tipo_modelo", "MLP")
            modelo_mlp = build_mlp_model(forma_entrada_mlp, config)
            modelo_mlp.fit( X_train, y_train_oh, validation_data=(X_val, y_val_oh), epochs=50, batch_size=32, callbacks=[callback_parada], verbose=6 ) # Sin class_weight
            mlflow.tensorflow.log_model(modelo_mlp, "model", registered_model_name=nombre_run)

    # --- 11. Entrenar Modelos CNN (Ahora paso 11) ---
    print("\n--- Entrenando Modelos CNN ---")
    mlflow.set_experiment("CNN_Models_WYNN_FixedAlpha_H1_NoWeight")
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_val_cnn = np.expand_dims(X_val, axis=-1)
    forma_entrada_cnn = X_train_cnn.shape[1]
    for config in configuraciones_cnn:
        nombre_run = f"CNN_conv{config['conv_layers']}_filters{config['conv_filters']}_dense{config['dense_units']}_tanh"
        with mlflow.start_run(run_name=nombre_run) as run:
             print(f"Entrenando: {nombre_run}")
             mlflow.log_params(config); mlflow.log_param("activation", "tanh"); mlflow.log_param("optimizer", "adam"); mlflow.set_tag("tipo_modelo", "CNN")
             modelo_cnn = build_cnn_model(forma_entrada_cnn, config)
             modelo_cnn.fit( X_train_cnn, y_train_oh, validation_data=(X_val_cnn, y_val_oh), epochs=50, batch_size=32, callbacks=[callback_parada], verbose=6 ) # Sin class_weight
             mlflow.tensorflow.log_model(modelo_cnn, "model", registered_model_name=nombre_run)

    print("\nEntrenamiento de modelos DL finalizado.")

# --- Punto de entrada ---
if __name__ == "__main__":
    print("Iniciando script de entrenamiento de modelos...")
    run_models()
    print("Script de entrenamiento completado.")