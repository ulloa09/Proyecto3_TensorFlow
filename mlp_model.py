import tensorflow as tf
from keras.src.metrics import Recall, Precision
import mlflow

def build_mlp_model(params, input_shape, num_classes):
    """
    Builds an MLP (Multi-Layer Perceptron) model for multiclass classification.

    Args:
        params (dict): Dictionary with model hyperparameters.
        input_shape (tuple): Shape of the input (n_features,).
        num_classes (int): Number of output classes (e.g., 3).

    Returns:
        tf.keras.Model: The compiled MLP model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Hidden dense layers
    for _ in range(params["dense_blocks"]):
        model.add(tf.keras.layers.Dense(params["dense_units"], activation=params["activation"]))
        model.add(tf.keras.layers.Dropout(params["dropout"]))

    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=params["optimizer"],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def train_mlp_model(model, X_train, y_train, X_val, y_val, params):
    """
    Trains the MLP model and logs metrics with MLFlow.

    Args:
        model (tf.keras.Model): The model to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        params (dict): Dictionary with training hyperparameters.

    Returns:
        tuple:
            - history (History): Keras history object.
            - model (tf.keras.Model): The trained model.
            - final_val_acc (float): Final validation accuracy.
            - final_val_loss (float): Final validation loss.
    """
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Proyecto3_TensorFlow") # Ensure experiment name is set

    class_weights = params.get("class_weights") # Get class weights if provided

    with mlflow.start_run():
        # Create a descriptive run name
        run_name = (
            f"MLP_dense{params.get('dense_units',64)}_"
            f"blocks{params.get('dense_blocks',2)}_"
            f"act{params.get('activation','relu')}"
        )
        mlflow.set_tag("run_name", run_name)

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=2, # Show one line per epoch
            class_weight=class_weights,
        )

        # Evaluate and log final metrics
        val_loss, val_acc, *_ = model.evaluate(X_val, y_val, verbose=0)
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("final_val_loss", val_loss)
        
        # Save the model to MLFlow
        mlflow.tensorflow.log_model(model, artifact_path="mlp_model")

    
    return history, model, val_acc, val_loss