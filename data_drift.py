import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict

def distribution_shift(base_distribution: pd.Series, 
                              current_distribution: pd.Series, 
                              significance_level: float = 0.05) -> bool:
 
    """
    Compares two series (distributions) using the Kolmogorov-Smirnov (KS) test.
    Returns True if the p-value is less than the significance level,
    indicating a statistically significant shift.

    Args:
        base_distribution (pd.Series): The reference data (e.g., training).
        current_distribution (pd.Series): The new data to compare (e.g., test).
        significance_level (float): The p-value threshold to detect the shift.

    Returns:
        bool: True if a shift is detected, False otherwise.
    """
    # Remove NaNs for the KS test to work
    reference_data = base_distribution.dropna()
    new_data = current_distribution.dropna()
    
    if reference_data.empty or new_data.empty:
        return False # Cannot compare if one is empty

    try:
        _, p_value = ks_2samp(reference_data, new_data)
        return p_value < significance_level
    except Exception:
        # Catch potential errors in ks_2samp (e.g., all constant values)
        return False

def get_feature_shift_status(baseline_features: pd.DataFrame, 
                            monitoring_features: pd.DataFrame, 
                             threshold: float = 0.05) -> Dict[str, bool]:
    """
    Calculates the shift status (True/False) for each feature in two DataFrames.

    Args:
        baseline_features (pd.DataFrame): The reference DataFrame.
        monitoring_features (pd.DataFrame): The new DataFrame to compare.
        threshold (float): The significance threshold.

    Returns:
        dict: A dictionary {feature_name: True/False} indicating if there is a shift.
    """
    shift_report = {}
    
    for col_name in baseline_features.columns:
        if col_name in monitoring_features.columns:
            # Check if the distribution of this column has changed
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
    Calculates the p-values of the KS test for each feature between two DataFrames.

    Args:
        reference_set (pd.DataFrame): The reference DataFrame.
        comparison_set (pd.DataFrame): The new DataFrame to compare.

    Returns:
        dict: A dictionary {feature_name: p_value}.
    """
    pvalue_map = {}
    
    for feature in reference_set.columns:
        if feature in comparison_set.columns:
            
            ref_data = reference_set[feature].dropna()
            comp_data = comparison_set[feature].dropna()
            
            if ref_data.empty or comp_data.empty:
                p_val = np.nan # Assign NaN if there is no data to compare
            else:
                try:
                    _, p_val = ks_2samp(ref_data, comp_data)
                except Exception:
                    p_val = np.nan # Assign NaN if test fails
                    
            pvalue_map[feature] = p_val
            
    return pvalue_map