import numpy as np
import pandas as pd

from backtest import backtest
from cnn_model import build_cnn_model, train_cnn_model, reshape_cnn
from mlp_model import build_mlp_model, train_mlp_model
from split import split_dfs
from preprocess_features import fechas, fit_scalers, apply_scalers
from functions import make_forward_return, compute_thresholds, label_by_thresholds, prepare_xy, compute_class_weights
from features import generate_features

# Carga de datos
datos = pd.read_csv('data/wynn_daily_15y.csv')
# Creación de fechas
datos = fechas(datos)
# Creación de features
datos = generate_features(datos)
# Calcular rendimiento futuro (forward return)
datos = make_forward_return(datos, horizon=5)
# Definir umbrales dinámicos (percentiles)
lower_thr, upper_thr = compute_thresholds(datos, lower_q=0.2, upper_q=0.8)
# Asignar etiquetas
datos = label_by_thresholds(datos, lower_thr, upper_thr)
# Drop de NAs
data = datos.copy().dropna()
# Split de datos
train_df, test_df, validation_df = split_dfs(data, train=60, test=20, validation=20)

# Creación y entrenamiento de escaladores
min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler = fit_scalers(train_df)

# === Aplicar los escaladores a los tres conjuntos ===
train_scaled = apply_scalers(train_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
test_scaled = apply_scalers(test_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
val_scaled = apply_scalers(validation_df, min_max_scaler, robust_scaler, standard_scaler, ohlcv_scaler)
print("\n \nEscaladores aplicados a todos los conjuntos correctamente.")

# Conservar fecha (necesaria backtesting)
train_scaled["Date"] = train_df["Date"]
test_scaled["Date"] = test_df["Date"]
val_scaled["Date"] = validation_df["Date"]


# === Guardar resultados en disco para revisión rápida ===
train_scaled.to_csv("data/train_scaled.csv", index=False)
test_scaled.to_csv("data/test_scaled.csv", index=False)
val_scaled.to_csv("data/val_scaled.csv", index=False)


# Separación en x, y para train, test y validation
X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy(train_scaled, val_scaled, test_scaled)


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

    # Construcción del modelo
    cnn_model_i = build_cnn_model(params, X_train_r.shape[1:], 3)

    # Entrenamiento del modelo
    history, trained_model, val_acc_cnn, val_loss_cnn = train_cnn_model(cnn_model_i, X_train_r, y_train, X_val_r, y_val, params)

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
    history, trained_model, val_acc, val_loss = train_mlp_model(mlp_model_i, X_train, y_train, X_val, y_val, params)
    print(f"✅ MLP {i} -> val_accuracy: {val_acc:.4f}, val_loss: {val_loss:.4f}")
    results_mlp.append({"config": i, "params": params, "val_acc": val_acc, "val_loss": val_loss})

    if val_acc > best_acc_mlp:
        best_acc_mlp = val_acc
        best_model_mlp = trained_model
        best_params_mlp = params

print("\n🏆 Mejor configuración MLP encontrada:")
print(best_params_mlp)

if best_acc_mlp > best_acc_cnn:
    best_model = best_model_mlp
    X_test_r = X_test
else:
    best_model = best_model_cnn


# Prueba de resultados EN TEST del mejor modelo con backtest
print(f"Evaluando el modelo ganador en TEST:{best_model}")
y_pred_test = np.argmax(best_model.predict(X_test_r), axis=1)
test_df["target"] = y_pred_test
cash, portfolio_value, buy, sell, hold, total_ops = backtest(test_df, stop_loss=0.015, take_profit=0.1, n_shares=10)

# Prueba de resultados EN VALIDATION del mejor modelo con backtest
print(f"Evaluando el modelo ganador en VALIDATION:{best_model}")
y_pred_val = np.argmax(best_model.predict(X_test_r), axis=1)
test_df["target"] = y_pred_val
cash, portfolio_value, buy, sell, hold, total_ops = backtest(validation_df, stop_loss=0.015, take_profit=0.1, n_shares=10)
