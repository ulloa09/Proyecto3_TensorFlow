import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

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
    kernel_size   = params.get("kernel_size", 1)   # 1 por ahora; más grande cuando metas ventanas
    conv_blocks   = params.get("conv_blocks", 2)
    dense_units   = params.get("dense_units", 64)
    activation    = params.get("activation", "relu")
    dropout_rate  = params.get("dropout", 0.2)

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
    optimizer_name = params.get("optimizer", "adam")
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
            verbose=2
        )

        # Métricas finales en validación
        final_val_acc  = float(history.history["val_accuracy"][-1])
        final_val_loss = float(history.history["val_loss"][-1])

        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_val_loss", final_val_loss)

        # Guardar el modelo en MLFlow
        mlflow.tensorflow.log_model(model, artifact_path="cnn_model")

    return history, model
