import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict

def distribution_shift(base_distribution: pd.Series, 
                              current_distribution: pd.Series, 
                              significance_level: float = 0.05) -> bool:
 
    """
    Compara dos series (distribuciones) usando el test KS.
    Devuelve True si el p-value es menor que el nivel de significancia,
    indicando un shift estadísticamente significativo.
    Parámetros:
        base_distribution (pd.Series): Los datos de referencia (ej. entrenamiento).
        current_distribution (pd.Series): Los nuevos datos a comparar (ej. prueba).
        significance_level (float): El umbral de p-value para detectar el shift.
    Retorna:
        bool: True si se detecta un shift, False en caso contrario.
    """
    # Se eliminan NaNs para que el test KS funcione
    reference_data = base_distribution.dropna()
    new_data = current_distribution.dropna()
    
    if reference_data.empty or new_data.empty:
        return False # No se puede comparar si uno está vacío

    try:
        _, p_value = ks_2samp(reference_data, new_data)
        return p_value < significance_level
    except Exception:
        return False

def get_feature_shift_status(baseline_features: pd.DataFrame, 
  
                            monitoring_features: pd.DataFrame, 
                             threshold: float = 0.05) -> Dict[str, bool]:
    """
    Calcula el estado de shift (True/False) para cada feature en dos DataFrames.
    Parámetros:
        baseline_features (pd.DataFrame): El DataFrame de referencia.
        monitoring_features (pd.DataFrame): El DataFrame nuevo a comparar.
        threshold (float): El umbral de significancia.
    Retorna:
        dict: Un diccionario {nombre_feature: True/False} indicando si hay shift.
    """
    shift_report = {}
    
    for col_name in baseline_features.columns:
        if col_name in monitoring_features.columns:
            # Revisa si la distribución de esta columna ha cambiado
            is_shifted = distribution_shift(
                baseline_features[col_name], 
                monitoring_features[col_name], 
   
                threshold
            )
            shift_report[col_name] = is_shifted
            
    return shift_report

def get_feature_pvalues(reference_set: pd.DataFrame, 
                        comparison_set: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula los p-values del test 
    KS para cada feature entre dos DataFrames.
    
    Parámetros:
        reference_set (pd.DataFrame): El DataFrame de referencia.
        comparison_set (pd.DataFrame): El DataFrame nuevo a comparar.

    Retorna:
        dict: Un diccionario {nombre_feature: p_value}.
    """
    pvalue_map = {}
    
    for feature in reference_set.columns:
        if feature in comparison_set.columns:
            
            ref_data = reference_set[feature].dropna()
            comp_data = comparison_set[feature].dropna()
            
            if ref_data.empty or comp_data.empty:
     
                p_val = np.nan # Asignar NaN si no hay datos para comparar
            else:
                try:
                    _, p_val = ks_2samp(ref_data, comp_data)
                except Exception:
         
                    p_val = np.nan
                    
            pvalue_map[feature] = p_val
            
    return pvalue_map