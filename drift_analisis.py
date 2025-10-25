import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List, Optional

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
        # Silenciamos la advertencia para el análisis rodante
        # print(f"Advertencia: Datos vacíos para KS test en la columna {datos_referencia.name}. Devolviendo NaN.")
        return np.nan, np.nan 

    statistic, valor_p = ks_2samp(ref, comp)
    return statistic, valor_p

def revisar_drift_features(df_referencia: pd.DataFrame,
                          df_comparacion: pd.DataFrame,
                          columnas_features: List[str],
                          alpha: float = 0.05,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Analiza el data drift para una lista de features entre dos DataFrames usando el test KS.
    """
    resultados_drift = []

    if verbose:
        print(f"\nAnalizando drift entre DataFrames (Ref: {df_referencia.shape}, Comp: {df_comparacion.shape}) con alpha={alpha}")

    for col in columnas_features:
        if col not in df_referencia.columns or col not in df_comparacion.columns:
            continue

        ks_stat, valor_p = ejecutar_ks_test(df_referencia[col], df_comparacion[col])
        drift_detectado = valor_p < alpha if not pd.isna(valor_p) else False

        resultados_drift.append({
            'Feature': col,
            'KS_Statistic': ks_stat,
            'p_value': valor_p,
            'Drift_Detected': drift_detectado
        })

    results_df = pd.DataFrame(resultados_drift)
    results_df = results_df.sort_values(by='p_value', ascending=True) # Ordenar por valor_p

    if verbose:
        features_con_drift = results_df[results_df['Drift_Detected'] == True]['Feature'].tolist()
        print(f"Análisis completado. Features con drift detectado ({len(features_con_drift)}):")
        if features_con_drift:
            print(features_con_drift)
        else:
            print("Ninguno")

    return results_df

def find_rolling_drift_breakpoint(
    df_referencia: pd.DataFrame, 
    df_comparacion: pd.DataFrame, 
    columnas_features: List[str],
    window_size: int = 90, 
    drift_threshold: float = 0.20
) -> Optional[pd.Timestamp]:
    """
    Analiza el data drift en ventanas rodantes para encontrar un "punto de quiebre".

    Compara el df_referencia (entrenamiento) contra ventanas del df_comparacion (prueba/val).

    Parámetros:
        df_referencia (pd.DataFrame): Datos de entrenamiento (escalados).
        df_comparacion (pd.DataFrame): Datos de prueba o validación (escalados).
        columnas_features (List[str]): Lista de features a revisar.
        window_size (int): Tamaño de la ventana rodante en días (ej. 90 días).
        drift_threshold (float): Porcentaje de features (ej. 0.20 = 20%) 
                                 que deben fallar el test KS (p < 0.05) 
                                 para considerar un "punto de quiebre".

    Retorna:
        Optional[pd.Timestamp]: La fecha del primer día de la ventana donde
                                se supera el umbral de drift, o None si no se supera.
    """
    print(f"\nIniciando análisis de drift rodante (Ventana: {window_size} días, Umbral: {drift_threshold*100}%)...")
    
    # Asegurar que el DataFrame de comparación tenga un índice Datetime
    if not isinstance(df_comparacion.index, pd.DatetimeIndex):
        if 'Datetime' in df_comparacion.columns:
            df_comparacion = df_comparacion.set_index('Datetime')
        else:
            print("Error en Drift Rodante: df_comparacion no tiene índice 'Datetime'.")
            return None

    df_comparacion = df_comparacion.sort_index()
    
    # Crear un objeto 'resampler' usando el tamaño de la ventana
    # 'on' usa el índice, 'closed' y 'label' asegura que trabajemos con periodos pasados
    resampler = df_comparacion.resample(f'{window_size}D', label='right', closed='right')
    
    total_features = len(columnas_features)
    if total_features == 0:
        print("Error en Drift Rodante: No se proporcionaron columnas de features.")
        return None

    for period_end, window_df in resampler:
        if window_df.empty or len(window_df) < window_size * 0.5: # Omitir ventanas muy pequeñas
            continue

        # 1. Ejecutar el análisis de drift para esta ventana vs. referencia
        drift_results_df = revisar_drift_features(
            df_referencia, 
            window_df, 
            columnas_features,
            verbose=False # Desactivar el print ruidoso
        )
        
        # 2. Calcular el porcentaje de features con drift
        features_con_drift = drift_results_df['Drift_Detected'].sum()
        drift_percentage = features_con_drift / total_features
        
        # 3. Comprobar si se superó el umbral
        if drift_percentage > drift_threshold:
            breakpoint_date = window_df.index.min()
            print(f"  -> ¡PUNTO DE QUIEBRE ENCONTRADO!")
            print(f"     Fecha: {breakpoint_date.date()}")
            print(f"     Features con Drift: {features_con_drift}/{total_features} ({drift_percentage:.1%})")
            return breakpoint_date # Devolver la fecha de inicio de la ventana rota

    print("  -> Análisis de drift rodante completado. No se encontró un punto de quiebre.")
    return None