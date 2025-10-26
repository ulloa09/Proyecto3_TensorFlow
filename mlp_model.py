# mlp_model.py
import tensorflow as tf

def build_mlp_model(params, input_shape, num_classes):
    """
    Construye un modelo MLP para clasificación multiclase.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Capas ocultas densas
    for _ in range(params["dense_blocks"]):
        model.add(tf.keras.layers.Dense(params["dense_units"], activation=params["activation"]))
        model.add(tf.keras.layers.Dropout(params["dropout"]))

    # Capa de salida
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=params["optimizer"],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


import mlflow

def train_mlp_model(model, X_train, y_train, X_val, y_val, params):
    """
    Entrena el modelo MLP y registra métricas con MLFlow.
    """
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Proyecto3_TensorFlow")

    with mlflow.start_run():
        run_name = (
            f"MLP_dense{params.get('dense_units',64)}_"
            f"blocks{params.get('dense_blocks',2)}_"
            f"act{params.get('activation','relu')}"
        )
        mlflow.set_tag("run_name", run_name)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=2
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("final_val_loss", val_loss)
        mlflow.tensorflow.log_model(model, artifact_path="mlp_model")

    return history, model, val_acc, val_loss