import numpy as np
from cnn_model import build_cnn_model, train_cnn_model, reshape_cnn
from mlp_model import build_mlp_model, train_mlp_model
from functions import compute_class_weights

def train_and_select_best_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Encapsulates the entire model training and selection process.
    1. Calculates class_weights.
    2. Prepares data for CNN (reshape).
    3. Trains and evaluates all CNN configurations.
    4. Trains and evaluates all MLP configurations.
    5. Compares and selects the best model (MLP vs. CNN) based on validation accuracy.
    6. Returns the winning model and the final X datasets (with the correct shape).

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
        {"dense_blocks": 4, "dense_units": 128, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 60, "batch_size": 64},
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

    # Determine the winning model (MLP or CNN)
    if best_acc_mlp > best_acc_cnn:
        print("\n🏆 Winning Model: MLP")
        best_model = best_model_mlp
        best_params = best_params_mlp
        model_name = f"MLP_dense{best_params.get('dense_blocks',2)}_units{best_params.get('dense_units',64)}"
        # Use 2D data shape for MLP
        X_train_final = X_train
        X_test_final = X_test
        X_val_final = X_val
    else:
        print("\n🏆 Winning Model: CNN")
        best_model = best_model_cnn
        best_params = best_params_cnn
        model_name = f"CNN1D_filters{best_params.get('num_filters',32)}_blocks{best_params.get('conv_blocks',2)}"
        # Use 3D (reshaped) data for CNN
        X_train_final = X_train_r
        X_test_final = X_test_r
        X_val_final = X_val_r
        
    return best_model, model_name, X_train_final, X_test_final, X_val_final