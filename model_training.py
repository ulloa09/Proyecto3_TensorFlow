import numpy as np
import mlflow
import tensorflow as tf # Added import
import traceback # Added import for exception handling
from cnn_model import build_cnn_model, train_cnn_model, reshape_cnn
from mlp_model import build_mlp_model, train_mlp_model
from functions import compute_class_weights
# Imports needed for standalone execution
from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data


# Define a consistent name for the registered model in MLFlow
BEST_MODEL_NAME = "SystematicTradingModel"

def register_model_mlflow(model, descriptive_name, val_accuracy, best_params):
    """
    Registers the best model with a consistent name in MLFlow
    and adds its descriptive_name (e.g., "CNN_...") as a tag.
    Uses log_model with registered_model_name for simplicity.

    Args:
        model (tf.keras.Model): The winning model object.
        descriptive_name (str): The descriptive name (e.g., "MLP_dense...").
        val_accuracy (float): The final validation accuracy of the model.
        best_params (dict): The hyperparameter dictionary of the best model.
    """
    print(f"\nRegistering model to MLFlow as: {BEST_MODEL_NAME}")

    try:
        # Log the model AND register it simultaneously
        # This will automatically create the registered model if it doesn't exist
        # and create a new version.
        model_info = mlflow.tensorflow.log_model(
            model,
            artifact_path="model", # Standard artifact path
            registered_model_name=BEST_MODEL_NAME # Instruct MLFlow to register/version
            # The problematic pip_requirements argument is removed
            # pip_requirements="-r requirements.txt"
            # You might need: input_example=..., signature=... for full pyfunc compatibility
        )

        print(f"Model logged and registered. URI: {model_info.model_uri}")

        # Now, get the version number that was just created
        client = mlflow.tracking.MlflowClient()

        # Get the latest version list (usually just one item)
        latest_versions = client.get_latest_versions(name=BEST_MODEL_NAME, stages=["None"])

        if not latest_versions:
             raise Exception(f"Could not find the latest version for model '{BEST_MODEL_NAME}' after logging.")

        # The latest version is the first element
        new_version_info = latest_versions[0]
        new_version = new_version_info.version

        print(f"Identified newly created version: {new_version}")

        # Set the tag on this specific version
        client.set_model_version_tag(
            name=BEST_MODEL_NAME,
            version=new_version,
            key="model_type",
            value=descriptive_name
        )

        print(f"✅ Successfully registered model '{BEST_MODEL_NAME}' version {new_version}")
        print(f"   Model Type Tag set to: {descriptive_name}")

    except Exception as e:
        print(f"❌ Error during model registration or tagging: {e}")
        # Print stack trace for debugging if needed
        traceback.print_exc()


def train_and_select_best_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Encapsulates the entire model training and selection process.
    1. Calculates class_weights.
    2. Prepares data for CNN (reshape).
    3. Trains and evaluates all CNN configurations.
    4. Trains and evaluates all MLP configurations.
    5. Compares and selects the best model (MLP vs. CNN) based on validation accuracy.
    6. Registers the winning model to MLFlow.
    7. Returns the winning model, its name, accuracy, and the final X datasets.

    Args:
        X_train (np.ndarray): 2D Training features.
        X_val (np.ndarray): 2D Validation features.
        X_test (np.ndarray): 2D Test features.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple:
            - best_model (tf.keras.Model): The winning model.
            - model_name (str): A descriptive name for the model.
            - X_train_final (np.ndarray): Training features in the correct shape (2D or 3D).
            - X_test_final (np.ndarray): Test features in the correct shape.
            - X_val_final (np.ndarray): Validation features in the correct shape.
            - best_model_accuracy (float): The validation accuracy of the winning model.
    """

    # 1. Class balancing
    class_weights = compute_class_weights(y_train)

    # 2. Re-Shape data for CNN input (samples, timesteps, features)
    X_train_r, X_test_r, X_val_r = reshape_cnn(X_train, X_test, X_val)
    print(f"New dimensions for CNN: \nTrain: {X_train_r.shape}, Test: {X_test_r.shape}, Val: {X_val_r.shape}")


    # 3. Define CNN hyperparameter search space
    params_space_cnn = [
        {"num_filters": 32, "kernel_size": 1, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 64},
        {"num_filters": 32, "kernel_size": 3, "conv_blocks": 3, "dense_units": 64, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 30, "batch_size": 64},
        {"num_filters": 32, "kernel_size": 4, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.15, "optimizer": "adam", "epochs": 40, "batch_size": 32}
    ]

    # === MULTIPLE CNN CONFIGURATION TEST ===
    results_cnn = [] # To store metrics for each run
    best_acc_cnn = -1.0 # Initialize with a low value
    best_model_cnn = None
    best_params_cnn = None

    for i, params in enumerate(params_space_cnn, 1):
        print(f"\n🔹 Training CNN configuration: {i}/{len(params_space_cnn)}: {params}")

        # Inject class_weights into the params
        params_with_weights = params.copy()
        params_with_weights['class_weight'] = class_weights

        # Build the model
        cnn_model_i = build_cnn_model(params, X_train_r.shape[1:], 3) # 3 classes

        # Train the model
        history, trained_model, val_acc_cnn, val_loss_cnn = train_cnn_model(
            cnn_model_i, X_train_r, y_train, X_val_r, y_val, params_with_weights
        )

        # Evaluate on validation
        print(f"✅ CNN Model {i} -> val_accuracy: {val_acc_cnn:.4f}, val_loss: {val_loss_cnn:.4f}")

        # Save results
        results_cnn.append({"config": i, "params": params, "val_acc": val_acc_cnn, "val_loss": val_loss_cnn})

        # Save best model
        if val_acc_cnn > best_acc_cnn:
            best_acc_cnn = val_acc_cnn
            best_model_cnn = trained_model
            best_params_cnn = params

    print("\n🏆 Best CNN configuration found:")
    print(best_params_cnn)

    # 4. Define MLP hyperparameter search space
    mlp_param_space = [
        {"dense_blocks": 4, "dense_units": 128, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 64},
    ]

    # === MULTIPLE MLP CONFIGURATION TEST ===
    results_mlp = []
    best_acc_mlp = -1.0 # Initialize with a low value
    best_model_mlp = None
    best_params_mlp = None

    for i, params in enumerate(mlp_param_space, 1):
        print(f"\n🔹 Training MLP configuration {i}/{len(mlp_param_space)}: {params}")

        # Inject class_weights
        params_with_weights = params.copy()
        params_with_weights['class_weights'] = class_weights # Note: mlp_model expects 'class_weights'

        # Build model
        mlp_model_i = build_mlp_model(params, X_train.shape[1:], 3) # 3 classes

        # Train model
        history, trained_model, val_acc, val_loss = train_mlp_model(
            mlp_model_i, X_train, y_train, X_val, y_val, params_with_weights
        )

        print(f"✅ MLP {i} -> val_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        results_mlp.append({"config": i, "params": params, "val_acc": val_acc, "val_loss": val_loss})

        # Save best model
        if val_acc > best_acc_mlp:
            best_acc_mlp = val_acc
            best_model_mlp = trained_model
            best_params_mlp = params

    print("\n🏆 Best MLP configuration found:")
    print(best_params_mlp)


    # === 5. & 6. FINAL MODEL SELECTION AND EXECUTION ===

    # This will store the final accuracy of the best model
    best_model_accuracy = -1.0

    # Determine the winning model (MLP or CNN)
    if best_acc_mlp > best_acc_cnn:
        best_model_accuracy = best_acc_mlp
        print(f"\n🏆 Winning Model: MLP (Val Acc: {best_model_accuracy:.4f})")
        best_model = best_model_mlp
        best_params = best_params_mlp
        model_name = f"MLP_dense{best_params.get('dense_blocks',2)}_units{best_params.get('dense_units',64)}"
        # Use 2D data shape for MLP
        X_train_final = X_train
        X_test_final = X_test
        X_val_final = X_val
    else:
        best_model_accuracy = best_acc_cnn
        print(f"\n🏆 Winning Model: CNN (Val Acc: {best_model_accuracy:.4f})")
        best_model = best_model_cnn
        best_params = best_params_cnn
        model_name = f"CNN1D_filters{best_params.get('num_filters',32)}_blocks{best_params.get('conv_blocks',2)}"
        # Use 3D (reshaped) data for CNN
        X_train_final = X_train_r
        X_test_final = X_test_r
        X_val_final = X_val_r

    # --- 6. Register the winning model to MLFlow ---
    if best_model is not None:
        # Pass all info needed for the registration
        register_model_mlflow(best_model, model_name, best_model_accuracy, best_params)
    else:
        print("Error: No best model was selected. Skipping registration.")
        model_name = None # Ensure model_name is None if no model

    return best_model, model_name, X_train_final, X_test_final, X_val_final, best_model_accuracy


if __name__ == "__main__":
    """
    This block allows the script to be run directly
    (e.g., `python model_training.py`) to perform the full
    data prep, model training, and MLFlow registration process.
    """

    # --- Configuration Constants (from main.py) ---
    DATA_CSV_PATH = 'data/wynn_daily_15y.csv'
    FWD_RETURN_HORIZON = 5
    lower = -0.1
    upper = 0.002
    SPLIT_RATIOS = {'train': 60, 'test': 20, 'validation': 20}

    print("--- 1. Loading and Preparing Data ---")
    train_df, test_df, validation_df = load_and_prepare_data(
        csv_path=DATA_CSV_PATH,
        horizon=FWD_RETURN_HORIZON,
        lower=lower,
        upper=upper,
        split_ratios=SPLIT_RATIOS
    )

    print("\n--- 2. Scaling Features ---")
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    print("\n--- 3. Preparing X/y ---")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )

    print("\n--- 4. Starting Model Training, Selection, and Registration ---")
    # This function now handles training, selection, and MLFlow registration
    # Capture the returned 'best_acc'
    best_model, model_name, _, _, _, best_acc = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("\n--- Training Script Finished ---")
    if model_name:
        # Print the model name and its validation accuracy
        print(f"✅ Best model selected: {model_name} (Val Acc: {best_acc:.4f})")
        print(f"✅ Model registered in MLFlow as '{BEST_MODEL_NAME}'")
    else:
        print("❌ Training script completed, but no model was selected or registered.")