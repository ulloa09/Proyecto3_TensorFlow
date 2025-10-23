import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from P2_split import split_dfs
from signals import generate_features, generate_targets
from preprocess_features import fechas, fit_scalers, apply_scalers

def construir_arquitectura_cnn(forma_entrada, config_modelo):
    """
    Construye un modelo CNN (Red Neuronal Convolucional) 1D.
    """
    modelo = Sequential()
    modelo.add(Input(shape=(forma_entrada, 1)))
    
    num_filtros = config_modelo.get("filtros", 64)
    capas_conv = config_modelo.get("capas_conv", 3)
    activacion = config_modelo.get("activacion", "relu")
    
    for i in range(capas_conv):
        modelo.add(Conv1D(filters=num_filtros, kernel_size=3, activation=activacion, padding="same"))
        modelo.add(BatchNormalization())
        if i < 2: # No hacer pooling en la última capa conv
             modelo.add(MaxPooling1D(pool_size=2))
        num_filtros *= 2 # Duplicar filtros

    modelo.add(Flatten())
    modelo.add(Dense(config_modelo.get("unidades_densas", 100), activation=activacion))
    modelo.add(Dropout(0.4)) # Aumentar dropout
    modelo.add(Dense(3, activation='softmax')) # 3 clases

    modelo.compile(optimizer=config_modelo.get("optimizador", "adam"),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo


def construir_arquitectura_mlp(forma_entrada, config_modelo):
    """
    Construye un modelo MLP (Perceptrón Multicapa).
    """
    modelo = Sequential()
    modelo.add(Input(shape=(forma_entrada,)))

    capas_densas = config_modelo.get("capas_densas", 3)
    unidades = config_modelo.get("unidades_densas", 150)
    activacion = config_modelo.get("activacion", "tanh")

    for _ in range(capas_densas):
        modelo.add(Dense(unidades, activation=activacion))
        modelo.add(Dropout(0.2))
        unidades = int(unidades / 2) # Reducir unidades en capas subsiguientes

    modelo.add(Dense(3, activation='softmax')) # 3 clases

    modelo.compile(optimizer=config_modelo.get("optimizador", "adam"),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo


def run_models():
    """
    Función principal para Cargar, Pre-procesar, Entrenar y Guardar modelos.
    """
    
    # --- 1. Carga y Pre-procesamiento de Datos ---
    print("Iniciando carga y pre-procesamiento...")
    try:
        datos = pd.read_csv('data/wynn_daily_15y.csv')
    except FileNotFoundError:
        print("Error: 'data/wynn_daily_15y.csv' no encontrado.")
        print("Asegúrate de ejecutar 'data_download.py' primero.")
        return
        
    datos = fechas(datos)
    datos = generate_features(datos)
    datos = generate_targets(datos, horizon=1, lower_q=0.2, upper_q=0.8)

    train_df, test_df, validation_df = split_dfs(datos, train=60, test=20, validation=20)

    min_max_scaler, robust_scaler, standard_scaler = fit_scalers(train_df)
    print("Escaladores entrenados y guardados.")

    train_scaled = apply_scalers(train_df.copy(), min_max_scaler, robust_scaler, standard_scaler)
    test_scaled = apply_scalers(test_df.copy(), min_max_scaler, robust_scaler, standard_scaler)
    val_scaled = apply_scalers(validation_df.copy(), min_max_scaler, robust_scaler, standard_scaler)

    print("Escaladores aplicados a todos los conjuntos.")

    train_scaled.to_csv("data/train_scaled.csv", index=False)
    test_scaled.to_csv("data/test_scaled.csv", index=False)
    val_scaled.to_csv("data/val_scaled.csv", index=False)
    print("Archivos escalados guardados en /data/")

    # --- 2. Preparar Datos para Keras ---
    columnas_excluir = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'fwd_ret', 'target']
    columnas_features = [col for col in train_scaled.columns if col not in columnas_excluir]
    
    print(f"Se usarán {len(columnas_features)} features para el entrenamiento.")
    
    x_train = train_scaled[columnas_features]
    x_val = val_scaled[columnas_features]
    
    # Mapeamos 'target' de (-1, 0, 1) a (0, 1, 2)
    y_train = train_scaled['target'] + 1
    y_val = val_scaled['target'] + 1

    # --- 3. Definir NUEVAS Configuraciones ---
    configuraciones_mlp = [
        {"capas_densas": 3, "unidades_densas": 150, "activacion": "tanh", "optimizador": "adam"},
        {"capas_densas": 2, "unidades_densas": 200, "activacion": "relu", "optimizador": "adam"},
    ]
    
    configuraciones_cnn = [
        {"capas_conv": 3, "filtros": 64, "activacion": "relu", "unidades_densas": 100, "optimizador": "adam"},
        {"capas_conv": 4, "filtros": 16, "activacion": "relu", "unidades_densas": 50, "optimizador": "adam"},
    ]

    callback_parada = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    mlflow.tensorflow.autolog()

    # --- 4. Entrenar Modelos MLP ---
    print("\n--- Entrenando Modelos MLP ---")
    mlflow.set_experiment("MLP_Models_WYNN")
    forma_entrada_mlp = x_train.shape[1]
    
    for config in configuraciones_mlp:
        nombre_run = f"MLP_capas{config['capas_densas']}_unidades{config['unidades_densas']}_{config['activacion']}"
        with mlflow.start_run(run_name=nombre_run) as run:
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config)
            mlflow.set_tag("tipo_modelo", "MLP")

            modelo_mlp = construir_arquitectura_mlp(forma_entrada_mlp, config)
            
            modelo_mlp.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=100, 
                batch_size=32,
                callbacks=[callback_parada],
                verbose=2
            )
            # Registrar modelo con un nombre único
            mlflow.tensorflow.log_model(modelo_mlp, "model", registered_model_name=nombre_run)
            
    # --- 5. Entrenar Modelos CNN ---
    print("\n--- Entrenando Modelos CNN ---")
    mlflow.set_experiment("CNN_Models_WYNN")
    
    x_train_cnn = np.expand_dims(x_train.values, axis=-1)
    x_val_cnn = np.expand_dims(x_val.values, axis=-1)
    forma_entrada_cnn = x_train_cnn.shape[1] 

    for config in configuraciones_cnn:
        nombre_run = f"CNN_capas{config['capas_conv']}_filtros{config['filtros']}_{config['activacion']}"
        with mlflow.start_run(run_name=nombre_run) as run:
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config)
            mlflow.set_tag("tipo_modelo", "CNN")
            
            modelo_cnn = construir_arquitectura_cnn(forma_entrada_cnn, config)
            
            modelo_cnn.fit(
                x_train_cnn, y_train,
                validation_data=(x_val_cnn, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[callback_parada],
                verbose=2
            )
            # Registrar modelo con un nombre único
            mlflow.tensorflow.log_model(modelo_cnn, "model", registered_model_name=nombre_run)

    print("\nEntrenamiento de modelos DL finalizado.")


# --- Punto de entrada para ejecutar el entrenamiento ---
if __name__ == "__main__":
    print("Iniciando script de entrenamiento de modelos...")
    run_models()
    print("Script de entrenamiento completado.")