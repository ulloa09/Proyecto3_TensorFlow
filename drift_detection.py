import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List

def run_ks_test(reference_data: pd.Series, comparison_data: pd.Series) -> Tuple[float, float]:
    """
    Ejecuta el test de Kolmogorov-Smirnov (KS) entre dos series de datos.

    Parámetros:
        reference_data (pd.Series): Datos de referencia (ej. del conjunto de entrenamiento).
        comparison_data (pd.Series): Nuevos datos a comparar (ej. del conjunto de prueba).

    Retorna:
        Tuple[float, float]: (Estadístico KS, p-value)
    """
    # Eliminar NaNs para evitar errores en ks_2samp
    ref = reference_data.dropna()
    comp = comparison_data.dropna()

    if ref.empty or comp.empty:
        print(f"Advertencia: Datos vacíos para KS test en la columna {reference_data.name}. Devolviendo NaN.")
        return np.nan, np.nan # O algún valor indicativo

    statistic, p_value = ks_2samp(ref, comp)
    return statistic, p_value

def analyze_feature_drift(reference_df: pd.DataFrame,
                          comparison_df: pd.DataFrame,
                          feature_columns: List[str],
                          alpha: float = 0.05) -> pd.DataFrame:
    """
    Analiza el data drift para una lista de features entre dos DataFrames usando el test KS.

    Parámetros:
        reference_df (pd.DataFrame): DataFrame de referencia (entrenamiento).
        comparison_df (pd.DataFrame): DataFrame a comparar (prueba o validación).
        feature_columns (List[str]): Lista de nombres de columnas (features) a analizar.
        alpha (float): Nivel de significancia para detectar drift (default: 0.05).

    Retorna:
        pd.DataFrame: Un DataFrame con los resultados:
                      ['Feature', 'KS_Statistic', 'p_value', 'Drift_Detected']
    """
    drift_results = []

    print(f"\nAnalizando drift entre DataFrames (Ref: {reference_df.shape}, Comp: {comparison_df.shape}) con alpha={alpha}")

    for col in feature_columns:
        if col not in reference_df.columns:
            print(f"Advertencia: La columna '{col}' no existe en el DataFrame de referencia. Omitiendo.")
            continue
        if col not in comparison_df.columns:
            print(f"Advertencia: La columna '{col}' no existe en el DataFrame de comparación. Omitiendo.")
            continue

        ks_stat, p_val = run_ks_test(reference_df[col], comparison_df[col])

        # Determinar si hay drift (p-value < alpha indica que las distribuciones son diferentes)
        drift_detected = p_val < alpha if not pd.isna(p_val) else False

        drift_results.append({
            'Feature': col,
            'KS_Statistic': ks_stat,
            'p_value': p_val,
            'Drift_Detected': drift_detected
        })

    results_df = pd.DataFrame(drift_results)
    results_df = results_df.sort_values(by='p_value', ascending=True) # Ordenar por p-value (más drifteadas primero)

    # Resumen
    drifted_features = results_df[results_df['Drift_Detected'] == True]['Feature'].tolist()
    print(f"Análisis completado. Features con drift detectado ({len(drifted_features)}):")
    if drifted_features:
        print(drifted_features)
    else:
        print("Ninguno")

    return results_df