import pandas as pd
from P2_split import split_dfs
from preprocess_features import fechas, fit_scalers, apply_scalers
from signals import generate_features, generate_targets


datos = pd.read_csv('data/wynn_daily_15y.csv')
datos = fechas(datos)
datos = generate_features(datos)
datos = generate_targets(datos, horizon=1, lower_q=0.2, upper_q=0.8)

train_df, test_df, validation_df = split_dfs(datos, train=60, test=20, validation=20)

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

print("Flujo completado ✅, datos listos para pasar a modelo")
print(f"Tamaños de dfs escalados \ntrain:{train_scaled.shape, train_scaled.shape == train_df.shape}\ntest:{test_scaled.shape, test_scaled.shape == test_df.shape }\nvalidation:{val_scaled.shape, val_scaled.shape == validation_df.shape}")

print(train_scaled)
print(val_scaled)

