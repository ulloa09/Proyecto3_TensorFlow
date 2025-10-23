import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping


# --- Propósito general ---
# Ahora define, entrena y registra los modelos de Deep Learning (MLP y CNN) usando MLflow.

def construir_arquitectura_cnn(forma_entrada, config_modelo):
    """
    Construye un modelo CNN (Red Neuronal Convolucional) 1D.
    """
    modelo = Sequential()
    modelo.add(Input(shape=(forma_entrada, 1)))
    
    num_filtros = config_modelo.get("filtros", 32)
    capas_conv = config_modelo.get("capas_conv", 2)
    activacion = config_modelo.get("activacion", "relu")
    
    for _ in range(capas_conv):
        modelo.add(Conv1D(filters=num_filtros, kernel_size=3, activation=activacion))
        modelo.add(MaxPooling1D(pool_size=2))
        num_filtros *= 2  # Duplicar filtros en la siguiente capa

    modelo.add(Flatten())
    modelo.add(Dense(config_modelo.get("unidades_densas", 64), activation=activacion))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(3, activation='softmax')) # 3 clases: -1, 0, 1 (mapeado a 0, 1, 2)

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

    capas_densas = config_modelo.get("capas_densas", 2)
    unidades = config_modelo.get("unidades_densas", 64)
    activacion = config_modelo.get("activacion", "relu")

    for _ in range(capas_densas):
        modelo.add(Dense(unidades, activation=activacion))
        modelo.add(Dropout(0.3))

    modelo.add(Dense(3, activation='softmax')) # 3 clases: -1, 0, 1 (mapeado a 0, 1, 2)

    modelo.compile(optimizer=config_modelo.get("optimizador", "adam"),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo


def iniciar_entrenamiento_modelos(df_entrenamiento, df_validacion):
    """
    Función principal para entrenar y registrar los modelos MLP y CNN.
    """
    print("Iniciando proceso de entrenamiento de modelos DL...")
    
    # --- 1. Preparar Datos ---
    # Los datos ya vienen escalados, solo separamos X (features) de y (target)
    
    # Excluimos columnas que no son features
    columnas_excluir = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'fwd_ret', 'target']
    # Nos aseguramos de solo tomar columnas que sí existen en el df
    columnas_features = [col for col in df_entrenamiento.columns if col not in columnas_excluir]
    
    print(f"Se usarán {len(columnas_features)} features para el entrenamiento.")
    
    x_train = df_entrenamiento[columnas_features]
    x_val = df_validacion[columnas_features]
    
    # Mapeamos 'target' de (-1, 0, 1) a (0, 1, 2) para Keras
    y_train = df_entrenamiento['target'] + 1
    y_val = df_validacion['target'] + 1

    # --- 2. Definir Configuraciones ---
    configuraciones_mlp = [
        {"capas_densas": 2, "unidades_densas": 128, "activacion": "relu", "optimizador": "adam"},
        {"capas_densas": 3, "unidades_densas": 64, "activacion": "relu", "optimizador": "adam"},
    ]
    
    configuraciones_cnn = [
        {"capas_conv": 2, "filtros": 32, "activacion": "relu", "unidades_densas": 64, "optimizador": "adam"},
        {"capas_conv": 3, "filtros": 32, "activacion": "tanh", "unidades_densas": 64, "optimizador": "adam"},
    ]

    callback_parada = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # --- 3. Entrenar Modelos MLP ---
    print("\n--- Entrenando Modelos MLP ---")
    forma_entrada_mlp = x_train.shape[1]
    
    for config in configuraciones_mlp:
        nombre_run = f"MLP_capas{config['capas_densas']}_unidades{config['unidades_densas']}_{config['activacion']}"
        with mlflow.start_run(run_name=nombre_run):
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config)
            mlflow.set_tag("tipo_modelo", "MLP")

            modelo_mlp = construir_arquitectura_mlp(forma_entrada_mlp, config)
            
            modelo_mlp.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=100, # Aumentado de 2 a 100, pero con EarlyStopping
                batch_size=32,
                callbacks=[callback_parada, mlflow.tensorflow.MLflowCallback()],
                verbose=2
            )
            # MLflow guardará el modelo automáticamente gracias al autologging (si está activo) o al callback

    # --- 4. Entrenar Modelos CNN ---
    print("\n--- Entrenando Modelos CNN ---")
    # CNN requiere una entrada 3D: (muestras, timesteps, features)
    # Aquí asumimos que cada fila es un timestep (timesteps=1) o que reformateamos
    # Para CNN 1D, necesita (muestras, features, 1) o (muestras, timesteps, features)
    
    x_train_cnn = np.expand_dims(x_train.values, axis=-1)
    x_val_cnn = np.expand_dims(x_val.values, axis=-1)
    forma_entrada_cnn = x_train_cnn.shape[1] # (num_features, 1)

    for config in configuraciones_cnn:
        nombre_run = f"CNN_capas{config['capas_conv']}_filtros{config['filtros']}_{config['activacion']}"
        with mlflow.start_run(run_name=nombre_run):
            print(f"Entrenando: {nombre_run}")
            mlflow.log_params(config)
            mlflow.set_tag("tipo_modelo", "CNN")
            
            modelo_cnn = construir_arquitectura_cnn(forma_entrada_cnn, config)
            
            modelo_cnn.fit(
                x_train_cnn, y_train,
                validation_data=(x_val_cnn, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[callback_parada, mlflow.tensorflow.MLflowCallback()],
                verbose=2
            )

    print("\nEntrenamiento de modelos DL finalizado. Revisa MLflow UI.")

if __name__ == "__main__":
    print("Este archivo contiene las funciones de entrenamiento.")
    print("Ejecuta el 'main.py' de entrenamiento para iniciar el proceso.")