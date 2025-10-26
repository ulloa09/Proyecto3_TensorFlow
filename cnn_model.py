import numpy as np
import tensorflow as tf
import mlflow
import pandas as pd
from keras.src.metrics import Precision, Recall


def build_cnn_model(params, input_shape, n_classes):
    """
    Builds a lightweight 1D CNN for multiclass classification (0=sell, 1=hold, 2=buy).
    Designed for tabular financial data reformatted as (timesteps, features).

    Args:
        params (dict): Dictionary with model hyperparameters.
        input_shape (tuple): Shape of the input (timesteps, n_features).
        n_classes (int): Number of output classes (e.g., 3).

    Returns:
        tf.keras.Model: The compiled CNN model.
    """

    # Hyperparameters with defaults
    num_filters   = params.get("num_filters", 32)
    kernel_size   = params.get("kernel_size", 3)
    conv_blocks   = params.get("conv_blocks", 2)
    dense_units   = params.get("dense_units", 64)
    activation    = params.get("activation", "relu")
    dropout_rate  = params.get("dropout", 0.2)
    optimizer_name = params.get("optimizer", "adam")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # 1D Convolutional Blocks
    for _ in range(conv_blocks):
        model.add(tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation=activation,
            padding="causal"  # 'causal' respects temporal order
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=1)) # pool_size=1 doesn't alter size
        # Note: When using windows > 1, you can change pool_size=2
        #       to summarize the past, and it will still work.

    # Flatten to a vector
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Intermediate dense layer
    model.add(tf.keras.layers.Dense(dense_units, activation=activation))

    # Output layer
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=optimizer_name,
        loss="sparse_categorical_crossentropy",  # For integer labels 0, 1, 2
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
    )

    return model


def train_cnn_model(model, X_train_seq, y_train, X_val_seq, y_val, params):
    """
    Trains the CNN model with MLFlow logging.

    Args:
        model (tf.keras.Model): The model to train.
        X_train_seq (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val_seq (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        params (dict): Dictionary with training hyperparameters (epochs, batch_size, etc.).

    Returns:
        tuple:
            - history (History): Keras history object.
            - model (tf.keras.Model): The trained model.
            - final_val_acc (float): Final validation accuracy.
            - final_val_loss (float): Final validation loss.
    """

    batch_size = params.get("batch_size", 64)
    epochs     = params.get("epochs", 20)
    class_weight = params.get("class_weight")

    # MLFlow autolog
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Proyecto3_TensorFlow") # Ensure experiment name is set


    with mlflow.start_run():
        # Create a descriptive run name
        run_name = (
            f"CNN1D_filters{params.get('num_filters',32)}_"
            f"blocks{params.get('conv_blocks',2)}_"
            f"dense{params.get('dense_units',64)}_"
            f"act{params.get('activation','relu')}"
        )
        mlflow.set_tag("run_name", run_name)

        # Train the model
        history = model.fit(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2, # Show one line per epoch
            class_weight=class_weight,
        )

        # Log final metrics
        final_val_acc  = float(history.history["val_accuracy"][-1])
        final_val_loss = float(history.history["val_loss"][-1])

        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_val_loss", final_val_loss)

        # Save the model to MLFlow
        mlflow.tensorflow.log_model(model, artifact_path="cnn_model")

    return history, model, final_val_acc, final_val_loss


def reshape_cnn(X_train, X_test, X_val):
    """
    Formats datasets for use in a 1D CNN without temporal context.
    Adds a dummy time dimension (timesteps=1).

    Args:
        X_train (np.ndarray): 2D Training features (samples, features).
        X_test (np.ndarray): 2D Test features.
        X_val (np.ndarray): 2D Validation features.

    Returns:
        tuple:
            - X_train_r (np.ndarray): 3D Training features (samples, 1, features).
            - X_test_r (np.ndarray): 3D Test features.
            - X_val_r (np.ndarray): 3D Validation features.
    """
    
    # Convert to numpy arrays (if they are pandas DataFrames)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)

    # Expand dimension for 1D CNN (samples, timesteps, features)
    X_train_r = np.expand_dims(X_train, axis=1)
    X_test_r = np.expand_dims(X_test, axis=1)
    X_val_r = np.expand_dims(X_val, axis=1)

    # Show resulting shapes
    print(f"✅ X_train reshaped to: {X_train_r.shape}")
    print(f"✅ X_test reshaped to:  {X_test_r.shape}")
    print(f"✅ X_val reshaped to:   {X_val_r.shape}")

    return X_train_r, X_test_r, X_val_r