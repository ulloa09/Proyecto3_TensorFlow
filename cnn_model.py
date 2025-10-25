import numpy as np
import tensorflow as tf
import mlflow
import pandas as pd

def build_cnn_model(params, input_shape, n_classes):
    """
    Construye una CNN 1D ligera para clasificación multiclase (0=venta,1=hold,2=compra)
    Pensada para datos financieros tabulares re-formateados como (timesteps, features).

    input_shape: tuple (timesteps, n_features)
    n_classes: número de clases de salida (ej. 3)
    params: dict con hiperparámetros del modelo
    """

    # Hiperparámetros con defaults
    num_filters   = params.get("num_filters", 32)
    kernel_size   = params.get("kernel_size", 3)
    conv_blocks   = params.get("conv_blocks", 2)
    dense_units   = params.get("dense_units", 64)
    activation    = params.get("activation", "relu")
    dropout_rate  = params.get("dropout", 0.2)
    optimizer_name = params.get("optimizer", "adam")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Bloques convolucionales 1D
    for _ in range(conv_blocks):
        model.add(tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation=activation,
            padding="causal"  # causal respeta el orden temporal cuando luego uses ventanas >1
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=1))  # pool_size=1 no altera tamaño ahora
        # Nota: cuando uses ventanas >1, puedes cambiar pool_size=2 para resumir el pasado
        #       y va a seguir funcionando

    # Aplastar a vector
    model.add(tf.keras.layers.Flatten())

    # Capa densa intermedia
    model.add(tf.keras.layers.Dense(dense_units, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    # Capa de salida
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    # Compilar

    model.compile(
        optimizer=optimizer_name,
        loss="sparse_categorical_crossentropy",  # etiquetas enteras 0,1,2
        metrics=["accuracy"]
    )

    return model


def train_cnn_model(model, X_train_seq, y_train, X_val_seq, y_val, params):
    """
    Entrena el modelo CNN con MLFlow.
    Usa train como entrenamiento y test como validación (por ahora).
    Devuelve history y el modelo entrenado.
    """

    batch_size = params.get("batch_size", 64)
    epochs     = params.get("epochs", 20)
    class_weight = params.get("class_weight")

    # MLFlow autolog
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        run_name = (
            f"CNN1D_filters{params.get('num_filters',32)}_"
            f"blocks{params.get('conv_blocks',2)}_"
            f"dense{params.get('dense_units',64)}_"
            f"act{params.get('activation','relu')}"
        )
        mlflow.set_tag("run_name", run_name)

        history = model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            class_weight=class_weight,
        )

        # Métricas finales en validación
        final_val_acc  = float(history.history["val_accuracy"][-1])
        final_val_loss = float(history.history["val_loss"][-1])

        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_val_loss", final_val_loss)

        # Guardar el modelo en MLFlow
        mlflow.tensorflow.log_model(model, artifact_path="cnn_model")

    return history, model, final_val_acc, final_val_loss



def test_multiple_cnn_configs(params_space, X_train_seq, y_train, X_val_seq, y_val, class_weights):
    """
    Entrena varios modelos CNN1D con distintas configuraciones de hiperparámetros
    y devuelve el mejor modelo según accuracy de validación.
    """

    best_acc = -np.inf
    best_model = None
    best_params = None

    for params in params_space:
        # Agrega class_weight al diccionario
        params["class_weight"] = class_weights

        print(f"\nEntrenando modelo con parámetros: {params}\n")

        # Construir modelo
        model = build_cnn_model(
            params=params,
            input_shape=X_train_seq.shape[1:],
            n_classes=3
        )

        # Entrenar modelo
        history, trained_model, acc, loss = train_cnn_model(
            model,
            X_train_seq, y_train,
            X_val_seq, y_val,
            params
        )

        print(f"Final val_acc={acc:.4f}, val_loss={loss:.4f}")

        # Guardar mejor modelo
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_params = params

    print(f"\n=== Mejor modelo encontrado ===")
    print(f"Params: {best_params}")
    print(f"Validation Accuracy: {best_acc:.4f}")

    return best_model, best_params, best_acc


def reshape_cnn(X_train, X_test, X_val):
    """
    Da formato a los datasets para poder usarlos en una CNN 1D sin contexto temporal.
    Agrega una dimensión ficticia de tiempo (timesteps=1).

    Parámetros:
    -----------
    X_train, X_test, X_val : np.ndarray o pd.DataFrame
        Conjuntos de features.
    y_train, y_test, y_val : np.ndarray o pd.Series
        Etiquetas correspondientes.

    Retorna:
    --------
    X_train_r, y_train, X_test_r, y_test, X_val_r, y_val
        Los X_* con shape (samples, 1, features)
    """
    # Convertir a arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)

    # Expandir dimensión para CNN 1D
    X_train_r = np.expand_dims(X_train, axis=1)
    X_test_r = np.expand_dims(X_test, axis=1)
    X_val_r = np.expand_dims(X_val, axis=1)

    # Mostrar shapes resultantes
    print(f"✅ X_train: {X_train_r.shape}")
    print(f"✅ X_test:  {X_test_r.shape}")
    print(f"✅ X_val:   {X_val_r.shape}")

    return X_train_r, X_test_r, X_val_r
