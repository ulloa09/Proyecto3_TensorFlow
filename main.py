import pandas as pd
from P2_split import split_dfs
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


min_max_scaler, robust_scaler, standard_scaler = fit_scalers(train_df)
print(f"Escaladores entrenados {min_max_scaler, robust_scaler, standard_scaler}\nEscaladores guardados (SE DEBE PASAR SOLO TRAIN DF)")

# === 5. Aplicar los escaladores a los tres conjuntos ===
train_scaled = apply_scalers(train_df, min_max_scaler, robust_scaler, standard_scaler)
test_scaled = apply_scalers(test_df, min_max_scaler, robust_scaler, standard_scaler)
val_scaled = apply_scalers(validation_df, min_max_scaler, robust_scaler, standard_scaler)

print("\n \nEscaladores aplicados a todos los conjuntos correctamente.")

# === 6. (Opcional) Guardar resultados en disco para revisión rápida ===
train_scaled.to_csv("data/train_scaled.csv", index=False)
test_scaled.to_csv("data/test_scaled.csv", index=False)
val_scaled.to_csv("data/val_scaled.csv", index=False)

print(f"Tamaños de dfs escalados \ntrain:{train_scaled.shape, train_scaled.shape == train_df.shape}\ntest:{test_scaled.shape, test_scaled.shape == test_df.shape }\nvalidation:{val_scaled.shape, val_scaled.shape == validation_df.shape}")

print(train_scaled)
print(val_scaled)

# Separación en x, y para train, test y validation
X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_xy(train_scaled, val_scaled, test_scaled)

# Balanceo de clases
class_weights = compute_class_weights(y_train)

