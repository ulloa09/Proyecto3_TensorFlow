import numpy as np
import pandas as pd

from backtest import backtest
from cnn_model import reshape_for_cnn, build_cnn_model, train_cnn_model
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
datos = make_forward_return(datos, horizon=1)
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

# RE-formateo de X para CNN
X_train_seq = reshape_for_cnn(X_train)
X_test_seq  = reshape_for_cnn(X_test)
X_val_seq   = reshape_for_cnn(X_val)
print(f"Nuevas dimensiones para CNN: \n{X_train_seq.shape, X_test_seq.shape, X_val_seq.shape}")

# Construcción de modelo CNN
params = {"conv_layers": 2, "conv_filters": 64, "activation": "relu", "dense_units": 64, "dropout": 0.25}
model1 = build_cnn_model(params, input_shape=X_train_seq.shape[1:], n_classes=3)
print(f"Modelo CNN creado con éxito:{model1}")
history, cnn_model, acc, loss = train_cnn_model(
    model1,
    X_train_seq, y_train,
    X_test_seq, y_test,
    params
)
print(f"Modelo entrenado en train y prueba en test \n Acc:{acc:.4f} y Loss:{loss:.4f}")


y_pred = np.argmax(cnn_model.predict(X_test_seq), axis=1)
test_df["target"] = y_pred
cash, portfolio_value, buy, sell, hold, total_ops = backtest(test_df, stop_loss=0.015, take_profit=0.2, n_shares=10)
