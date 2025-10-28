import numpy as np
import mlflow
import traceback
import config 
from models import build_cnn_model, train_cnn_model, reshape_cnn, build_mlp_model, train_mlp_model
from graphs import plot_training_history
from functions import compute_class_weights
from data_pipeline import load_and_prepare_data, scale_data, prepare_xy_data

# Define a consistent name for the registered model in MLFlow
BEST_MODEL_NAME = config.BEST_MODEL_NAME

def register_model_mlflow(model, descriptive_name, val_accuracy, best_params):
    """
    Registers the best model with a consistent name in MLFlow
    and adds its descriptive_name (e.g., "CNN_...") as a tag.
    
    This function now ALSO tags the model version with its
    validation accuracy and the hyperparameters used to achieve it.

    Args:
        model (tf.keras.Model): The winning model object.
        descriptive_name (str): The descriptive name (e.g., "MLP_dense...").
        val_accuracy (float): The final validation accuracy of the model.
        best_params (dict): The hyperparameter dictionary of the best model.
    """
    print(f"\nRegistering model to MLFlow as: {BEST_MODEL_NAME}")

    try:
        # Log the model AND register it simultaneously
        model_info = mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=BEST_MODEL_NAME 
        )

        print(f"Model logged and registered. URI: {model_info.model_uri}")

        # Now, get the version number that was just created
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(name=BEST_MODEL_NAME, stages=["None"])

        if not latest_versions:
             raise Exception(f"Could not find the latest version for model '{BEST_MODEL_NAME}' after logging.")

        new_version_info = latest_versions[0]
        new_version = new_version_info.version

        print(f"Identified newly created version: {new_version}")

        # --- Set tags on the new model version ---
        
        # Set the model type tag
        client.set_model_version_tag(
            name=BEST_MODEL_NAME,
            version=new_version,
            key="model_type",
            value=descriptive_name
        )
        
        # Set the validation accuracy tag
        client.set_model_version_tag(
            name=BEST_MODEL_NAME,
            version=new_version,
            key="val_accuracy",
            value=f"{val_accuracy:.4f}"
        )
        
        # Log hyperparameters as individual tags
        for key, value in best_params.items():
            # Ensure value is a simple type for tagging
            param_value = str(value) if not isinstance(value, (str, int, float, bool)) else value
            client.set_model_version_tag(
                name=BEST_MODEL_NAME,
                version=new_version,
                key=f"param_{key}",
                value=param_value 
            )

        print(f"✅ Successfully registered model '{BEST_MODEL_NAME}' version {new_version}")
        print(f"   Model Type Tag set to: {descriptive_name}")
        print(f"   Val Accuracy Tag set to: {val_accuracy:.4f}")
        print(f"   Hyperparameters tagged.")

    except Exception as e:
        print(f"❌ Error during model registration or tagging: {e}")
        traceback.print_exc()


def load_model_from_mlflow(X_train, X_val, X_test):
    """
    Loads a pre-trained model and its metadata from MLFlow.
    
    This function fetches the model specified in config.py,
    checks its 'model_type' tag (CNN or MLP), and returns
    the model and correctly-shaped data (2D or 3D).

    Args:
        X_train (np.ndarray): 2D Training features.
        X_val (np.ndarray): 2D Validation features.
        X_test (np.ndarray): 2D Test features.

    Returns:
        tuple (matching train_and_select_best_model):
            - model (tf.keras.Model): The loaded model.
            - model_name (str): The descriptive name from the tag.
            - X_train_final (np.ndarray): Training features (2D or 3D).
            - X_test_final (np.ndarray): Test features (2D or 3D).
            - X_val_final (np.ndarray): Validation features (2D or 3D).
            - val_accuracy (float): The validation accuracy from the tag.
    """
    model_reg_name = config.BEST_MODEL_NAME
    model_version = config.MODEL_VERSION_TO_LOAD
    
    print(f"\n--- Loading Model from MLFlow ---")
    print(f"Model: {model_reg_name}, Version: {model_version}")

    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get model version details to read tags
        version_info = client.get_model_version(model_reg_name, model_version)
        tags = version_info.tags
        
        # Extract metadata from tags
        model_name = tags.get("model_type", "LoadedModel") # Get descriptive name
        val_accuracy = float(tags.get("val_accuracy", 0.0))
        
        print(f"Loaded Model Type: {model_name}")
        print(f"Registered Val Accuracy: {val_accuracy:.4f}")

        # Define the model URI to load
        model_uri = f"models:/{model_reg_name}/{model_version}"
        
        # Load the model
        model = mlflow.tensorflow.load_model(model_uri)
        print("Model loaded successfully.")

        # Check the model type from the tag to determine data shape
        if "CNN" in model_name:
            print("Model is a CNN. Reshaping data to 3D.")
            # Reshape data to 3D for CNN
            X_train_final, X_test_final, X_val_final = reshape_cnn(X_train, X_test, X_val)
        else:
            print("Model is an MLP. Using 2D data.")
            # Use 2D data for MLP
            X_train_final = X_train
            X_test_final = X_test
            X_val_final = X_val
        
        return model, model_name, X_train_final, X_test_final, X_val_final, val_accuracy

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure MLFlow server is running and model/version exist.")
        traceback.print_exc()
        return None, None, None, None, None, None


def train_and_select_best_model(X_train, X_val, X_test, y_train, y_val):
    """
    Encapsulates the entire model training and selection process.
    (y_test is no longer an argument as it was unused).
    
    1. Calculates class_weights.
    2. Prepares data for CNN (reshape).
    3. Trains and evaluates all CNN configurations (from config.py).
    4. Trains and evaluates all MLP configurations (from config.py).
    5. Compares and selects the best model (MLP vs. CNN) based on validation accuracy.
    6. Registers the winning model to MLFlow.
    7. Returns the winning model, its name, accuracy, and the final X datasets.

    Args:
        X_train (np.ndarray): 2D Training features.
        X_val (np.ndarray): 2D Validation features.
        X_test (np.ndarray): 2D Test features.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        
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
    # Parameters are now loaded from config
    params_space_cnn = config.params_space_cnn

    # === MULTIPLE CNN CONFIGURATION TEST ===
    results_cnn = [] 
    best_acc_cnn = -1.0 
    best_model_cnn = None
    best_params_cnn = None
    best_history_cnn = None
    min_loss_cnn = 10

    for i, params in enumerate(params_space_cnn, 1):
        print(f"\n🔹 Training CNN configuration: {i}/{len(params_space_cnn)}: {params}")

        params_with_weights = params.copy()
        params_with_weights['class_weight'] = class_weights

        cnn_model_i = build_cnn_model(params, X_train_r.shape[1:], 3) # 3 classes

        history, trained_model, val_acc_cnn, val_loss_cnn = train_cnn_model(
            cnn_model_i, X_train_r, y_train, X_val_r, y_val, params_with_weights
        )

        print(f"✅ CNN Model {i} -> val_accuracy: {val_acc_cnn:.4f}, val_loss: {val_loss_cnn:.4f}")
        results_cnn.append({"config": i, "params": params, "val_acc": val_acc_cnn, "val_loss": val_loss_cnn, "history": history})

        if val_acc_cnn < min_loss_cnn:
            min_loss_cnn = val_loss_cnn
            best_acc_cnn = val_acc_cnn
            best_model_cnn = trained_model
            best_params_cnn = params
            best_history_cnn = history

    print("\n🏆 Best CNN configuration found:")
    print(best_params_cnn)

    # 4. Define MLP hyperparameter search space
    # Parameters are now loaded from config
    mlp_param_space = config.mlp_param_space

    # === MULTIPLE MLP CONFIGURATION TEST ===
    results_mlp = []
    best_acc_mlp = -1.0
    best_model_mlp = None
    best_params_mlp = None
    best_history_mlp = None
    min_loss_mlp = 10

    for i, params in enumerate(mlp_param_space, 1):
        print(f"\n🔹 Training MLP configuration {i}/{len(mlp_param_space)}: {params}")

        params_with_weights = params.copy()
        params_with_weights['class_weights'] = class_weights

        mlp_model_i = build_mlp_model(params, X_train.shape[1:], 3) # 3 classes

        history, trained_model, val_acc, val_loss = train_mlp_model(
            mlp_model_i, X_train, y_train, X_val, y_val, params_with_weights
        )

        print(f"✅ MLP {i} -> val_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        results_mlp.append({"config": i, "params": params, "val_acc": val_acc, "val_loss": val_loss, "history": history})

        if val_loss < min_loss_mlp:
            min_loss_mlp = val_loss
            best_acc_mlp = val_acc
            best_model_mlp = trained_model
            best_params_mlp = params
            best_history_mlp = history

    print("\n🏆 Best MLP configuration found:")
    print(best_params_mlp)

    # === 5. & 6. FINAL MODEL SELECTION AND EXECUTION ===

    best_model_accuracy = -1.0
    best_history = []
    min_loss_general = 15

    if min_loss_mlp < min_loss_cnn:
        best_model_accuracy = best_acc_mlp
        print(f"\n🏆 Winning Model: MLP (Val Acc: {best_model_accuracy:.4f})")
        best_model = best_model_mlp
        best_params = best_params_mlp
        best_history = best_history_mlp
        min_loss_general = min_loss_cnn
        model_name = f"MLP_dense{best_params.get('dense_blocks',2)}_units{best_params.get('dense_units',64)}"
        X_train_final = X_train
        X_test_final = X_test
        X_val_final = X_val
    else:
        best_model_accuracy = best_acc_cnn
        print(f"\n🏆 Winning Model: CNN (Val Acc: {best_model_accuracy:.4f})")
        best_model = best_model_cnn
        best_params = best_params_cnn
        min_loss_general = min_loss_cnn
        model_name = f"CNN1D_filters{best_params.get('num_filters',32)}_blocks{best_params.get('conv_blocks',2)}"
        X_train_final = X_train_r
        X_test_final = X_test_r
        X_val_final = X_val_r
        best_history = best_history_cnn

    # --- 6. Register the winning model to MLFlow ---
    if best_model is not None:
        # Pass all info needed for the registration
        register_model_mlflow(best_model, model_name, best_model_accuracy, best_params)
    else:
        print("Error: No best model was selected. Skipping registration.")
        model_name = None

    plot_training_history(best_history_cnn, best_history_mlp)

    return best_model, model_name, X_train_final, X_test_final, X_val_final, best_model_accuracy


if __name__ == "__main__":
    """
    This block allows the script to be run directly
    (e.g., `python model_training.py`) to perform the full
    data prep, model training, and MLFlow registration process.
    """

    # --- Configuration Constants (from config.py) ---
    print("--- 1. Loading and Preparing Data (from config) ---")
    train_df, test_df, validation_df = load_and_prepare_data()

    print("\n--- 2. Scaling Features ---")
    train_scaled, test_scaled, val_scaled = scale_data(
        train_df, test_df, validation_df
    )

    print("\n--- 3. Preparing X/y ---")
    X_train, X_val, X_test, y_train, y_val, feature_cols = prepare_xy_data(
        train_scaled, val_scaled, test_scaled
    )

    print("\n--- 4. Starting Model Training, Selection, and Registration ---")
    best_model, model_name, _, _, _, best_acc = train_and_select_best_model(
        X_train, X_val, X_test, y_train, y_val
    )

    print("\n--- Training Script Finished ---")
    if model_name:
        print(f"✅ Best model selected: {model_name} (Val Acc: {best_acc:.4f})")
        print(f"✅ Model registered in MLFlow as '{BEST_MODEL_NAME}'")
    else:
        print("❌ Training script completed, but no model was selected or registered.")