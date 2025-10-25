import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List

def ejecutar_ks_test(datos_referencia: pd.Series, datos_comparacion: pd.Series) -> Tuple[float, float]:
    """
    Ejecuta el test de Kolmogorov-Smirnov (KS) entre dos series de datos.
    Parámetros:
        datos_referencia (pd.Series): Datos de referencia (ej. del conjunto de entrenamiento).
        datos_comparacion (pd.Series): Nuevos datos a comparar (ej. del conjunto de prueba).
    Retorna:
        Tuple[float, float]: (Estadístico KS, valor_p)
    """
    # Eliminar NaNs para evitar errores en ks_2samp
    ref = datos_referencia.dropna()
    comp = datos_comparacion.dropna()

    if ref.empty or comp.empty:
        print(f"Advertencia: Datos vacíos para KS test en la columna {datos_referencia.name}. Devolviendo NaN.")
        return np.nan, np.nan 

    statistic, valor_p = ks_2samp(ref, comp)
    return statistic, valor_p

def revisar_drift_features(df_referencia: pd.DataFrame,
                          df_comparacion: pd.DataFrame,
                          columnas_features: List[str],
                          alpha: float = 0.05) -> pd.DataFrame:
    """
    Analiza el data drift para una lista de features entre dos DataFrames usando el test KS.

    Parámetros:
        df_referencia (pd.DataFrame): DataFrame de referencia (entrenamiento).
        df_comparacion (pd.DataFrame): DataFrame a comparar (prueba o validación).
        columnas_features (List[str]): Lista de nombres de columnas (features) a analizar.
        alpha (float): Nivel de significancia para detectar drift (default: 0.05).
    Retorna:
        pd.DataFrame: Un DataFrame con los resultados:
                      ['Feature', 'KS_Statistic', 'p_value', 'Drift_Detected']
    """
    resultados_drift = []

    print(f"\nAnalizando drift entre DataFrames (Ref: {df_referencia.shape}, Comp: {df_comparacion.shape}) con alpha={alpha}")

    for col in columnas_features:
        if col not in df_referencia.columns:
            print(f"Advertencia: La columna '{col}' no existe en el DataFrame de referencia. Omitiendo.")
            continue
        if col not in df_comparacion.columns:
            print(f"Advertencia: La columna '{col}' no existe en el DataFrame de comparación. Omitiendo.")
            continue

        ks_stat, valor_p = ejecutar_ks_test(df_referencia[col], df_comparacion[col])

        # Determinar si hay drift (valor_p < alpha indica que las distribuciones son diferentes)
        drift_detectado = valor_p < alpha if not pd.isna(valor_p) else False

        resultados_drift.append({
            'Feature': col,
            'KS_Statistic': ks_stat,
            'p_value': valor_p,
            'Drift_Detected': drift_detectado
        })

    results_df = pd.DataFrame(resultados_drift)
    results_df = results_df.sort_values(by='p_value', ascending=True) # Ordenar por valor_p

    # Resumen
    features_con_drift = results_df[results_df['Drift_Detected'] == True]['Feature'].tolist()
    print(f"Análisis completado. Features con drift detectado ({len(features_con_drift)}):")
    if features_con_drift:
        print(features_con_drift)
    else:
        print("Ninguno")

    return results_df