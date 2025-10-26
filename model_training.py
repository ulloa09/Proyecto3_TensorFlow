import numpy as np
from cnn_model import build_cnn_model, train_cnn_model, reshape_cnn
from mlp_model import build_mlp_model, train_mlp_model
from functions import compute_class_weights

def train_and_select_best_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Encapsula toodo el proceso de entrenamiento y selección de modelos.
    1. Calcula class_weights.
    2. Prepara datos para CNN (reshape).
    3. Entrena y evalúa todas las configuraciones de CNN.
    4. Entrena y evalúa todas las configuraciones de MLP.
    5. Compara y selecciona el mejor modelo (MLP vs CNN).
    6. Devuelve el modelo ganador y los conjuntos de datos X finales (con el shape correcto).
    """
    
    # Balanceo de clases
    class_weights = compute_class_weights(y_train)

    #RE-Shape para ingresar datos a CNN
    X_train_r, X_test_r, X_val_r = reshape_cnn(X_train, X_test, X_val)
    print(f"Nuevas dimensiones para CNN: \n{X_train_r.shape, X_test_r.shape, X_val_r.shape}")


    # Construcción del mejor modelo CNN
    params_space_cnn = [
        {"num_filters": 32, "kernel_size": 1, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 40, "batch_size": 64},
        {"num_filters": 32, "kernel_size": 3, "conv_blocks": 3, "dense_units": 64, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 30, "batch_size": 64},
        {"num_filters": 32, "kernel_size": 4, "conv_blocks": 2, "dense_units": 128, "activation": "sigmoid", "dropout": 0.15, "optimizer": "adam", "epochs": 40, "batch_size": 32}
    ]

    # === PRUEBA MULTIPLE DE CONFIGURACIONES CNN ===
    results = []  # para guardar métricas o referencias de cada corrida
    best_acc_cnn = 0
    best_model_cnn = None
    best_params_cnn = None

    for i, params in enumerate(params_space_cnn, 1):
        print(f"\n🔹 Entrenando configuración CNN: {i}/{len(params_space_cnn)}: {params}")

        # Inyectar class_weights en los params
        params_with_weights = params.copy()
        params_with_weights['class_weight'] = class_weights

        # Construcción del modelo
        cnn_model_i = build_cnn_model(params, X_train_r.shape[1:], 3)

        # Entrenamiento del modelo
        history, trained_model, val_acc_cnn, val_loss_cnn = train_cnn_model(
            cnn_model_i, X_train_r, y_train, X_val_r, y_val, params_with_weights
        )

        # Evaluación en validación
        print(f"✅ Modelo CNN {i} -> val_accuracy: {val_acc_cnn:.4f}, val_loss: {val_loss_cnn:.4f}")

        # Guardar resultados
        results.append({"config": i, "params": params, "val_acc": val_acc_cnn, "val_loss": val_loss_cnn})

        # Guardar mejor modelo
        if val_acc_cnn > best_acc_cnn:
            best_acc_cnn = val_acc_cnn
            best_model_cnn = trained_model
            best_params_cnn = params

    print("\n🏆 Mejor configuración CNN ncontrada:")
    print(best_params_cnn)

    # === PRUEBA MULTIPLE DE CONFIGURACIONES MLP ===
    mlp_param_space = [
        {"dense_blocks": 4, "dense_units": 128, "activation": "relu", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 50, "batch_size": 32},
        {"dense_blocks": 3, "dense_units": 64, "activation": "sigmoid", "dropout": 0.2, "optimizer": "adam", "epochs": 60, "batch_size": 64},
    ]

    results_mlp = []
    best_acc_mlp = 0
    best_model_mlp = None
    best_params_mlp = None

    for i, params in enumerate(mlp_param_space, 1):
        print(f"\n🔹 Entrenando configuración MLP {i}/{len(mlp_param_space)}: {params}")
        mlp_model_i = build_mlp_model(params, X_train.shape[1:], 3)
        
        history, trained_model, val_acc, val_loss = train_mlp_model(
            mlp_model_i, X_train, y_train, X_val, y_val, params
        )
        
        print(f"✅ MLP {i} -> val_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        results_mlp.append({"config": i, "params": params, "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_acc_mlp:
            best_acc_mlp = val_acc
            best_model_mlp = trained_model
            best_params_mlp = params

    print("\n🏆 Mejor configuración MLP encontrada:")
    print(best_params_mlp)


    # === SELECCIÓN DEL MODELO GANADOR Y EJECUCIÓN FINAL ===

    # Determinar el modelo ganador (MLP o CNN)
    if best_acc_mlp > best_acc_cnn:
        print("\n🏆 Modelo Ganador: MLP")
        best_model = best_model_mlp
        best_params = best_params_mlp
        model_name = f"MLP_dense{best_params.get('dense_blocks',2)}_units{best_params.get('dense_units',64)}"
        # Usar datos con forma 2D para MLP
        X_train_final = X_train
        X_test_final = X_test
        X_val_final = X_val
    else:
        print("\n🏆 Modelo Ganador: CNN")
        best_model = best_model_cnn
        best_params = best_params_cnn
        model_name = f"CNN1D_filters{best_params.get('num_filters',32)}_blocks{best_params.get('conv_blocks',2)}"
        # Usar datos con forma 3D (reformateados) para CNN
        X_train_final = X_train_r
        X_test_final = X_test_r
        X_val_final = X_val_r
        
    return best_model, model_name, X_train_final, X_test_final, X_val_final