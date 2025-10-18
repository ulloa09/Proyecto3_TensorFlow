import pandas as pd

# --- Propósito: dividir un DataFrame en Train/Test/Validation manteniendo el orden temporal ---
def split_dfs(data, train:int, test:int, validation:int):

    # --- Validación de proporciones ---
    # Asegura que los tres porcentajes cubran el 100% del dataset.
    assert train + test + validation == 100, "La suma de train, test y validation debe ser 100 exacto."

    # --- Cálculo de índices de corte ---
    # Define los límites de cada bloque con base en los porcentajes indicados.
    n = len(data)
    train_corte = int(n * train / 100)
    test_corte = train_corte + int(n * test / 100)

    # --- Creación de subconjuntos ---
    # Extrae las particiones en el orden: Train (inicio→train_corte),
    # Test (train_corte→test_corte) y Validation (test_corte→fin).
    train_df = data.iloc[:train_corte]
    test_df = data.iloc[train_corte:test_corte]
    validation_df = data.iloc[test_corte:]

    # --- Retorno ---
    # Devuelve las tres particiones para su uso en backtests/evaluaciones.
    return train_df, test_df, validation_df