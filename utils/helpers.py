import pandas as pd
import numpy as np
from typing import Any, Dict, List


def detect_column_type(series: pd.Series) -> str:
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return 'empty'
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    if pd.api.types.is_numeric_dtype(series):
        if set(clean_series.unique()).issubset({0, 1, True, False}):
            return 'binary'
        if series.nunique() == len(series):
            return 'id'
        if series.nunique() < 20:
            return 'discrete_numeric'
        return 'continuous_numeric'
    
    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.05:
        return 'categorical'
    elif unique_ratio < 0.5:
        return 'high_cardinality_categorical'
    else:
        return 'text'


def format_number(num: float, decimals: int = 2) -> str:
    if abs(num) >= 1000000:
        return f"{num/1000000:.{decimals}f}M"
    elif abs(num) >= 1000:
        return f"{num/1000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def calculate_effect_size(group1: pd.Series, group2: pd.Series, 
                         method: str = 'cohens_d') -> float:
    if method == 'cohens_d':
        pooled_std = np.sqrt(
            ((len(group1) - 1) * group1.std()**2 + 
             (len(group2) - 1) * group2.std()**2) / 
            (len(group1) + len(group2) - 2)
        )
        return (group1.mean() - group2.mean()) / pooled_std
    
    elif method == 'glass_delta':
        return (group1.mean() - group2.mean()) / group2.std()
    
    elif method == 'hedges_g':
        cohens_d = calculate_effect_size(group1, group2, 'cohens_d')
        n = len(group1) + len(group2)
        correction = 1 - (3 / (4 * n - 9))
        return cohens_d * correction
    
    return 0.0


def interpret_effect_size(effect_size: float, metric: str = 'cohens_d') -> str:
    abs_es = abs(effect_size)
    
    if metric in ['cohens_d', 'hedges_g', 'glass_delta']:
        if abs_es < 0.2:
            return 'negligible'
        elif abs_es < 0.5:
            return 'small'
        elif abs_es < 0.8:
            return 'medium'
        else:
            return 'large'
    
    elif metric == 'cramers_v':
        if abs_es < 0.1:
            return 'negligible'
        elif abs_es < 0.3:
            return 'small'
        elif abs_es < 0.5:
            return 'medium'
        else:
            return 'large'
    
    elif metric == 'correlation':
        if abs_es < 0.3:
            return 'weak'
        elif abs_es < 0.7:
            return 'moderate'
        else:
            return 'strong'
    
    return 'unknown'


def generate_summary_stats(data: pd.DataFrame, 
                          columns: List[str] = None) -> Dict[str, Any]:

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {}
    
    for col in columns:
        if col not in data.columns:
            continue
        
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        summary[col] = {
            'count': len(col_data),
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'q25': float(col_data.quantile(0.25)),
            'q75': float(col_data.quantile(0.75)),
            'missing': int(data[col].isna().sum()),
            'missing_pct': float((data[col].isna().sum() / len(data)) * 100)
        }
    
    return summary


def validate_ab_test_data(data: pd.DataFrame, group_col: str, 
                         metric_col: str) -> Dict[str, Any]:
    issues = []
    warnings = []
    
    if group_col not in data.columns:
        issues.append(f"Group column '{group_col}' not found")
    
    if metric_col not in data.columns:
        issues.append(f"Metric column '{metric_col}' not found")
    
    if issues:
        return {'valid': False, 'issues': issues, 'warnings': warnings}
    
    n_groups = data[group_col].nunique()
    if n_groups < 2:
        issues.append("Need at least 2 groups for comparison")
    
    group_sizes = data.groupby(group_col).size()
    min_size = group_sizes.min()
    
    if min_size < 30:
        warnings.append(f"Small sample size detected (min={min_size}). Results may not be reliable.")
    
    missing_groups = data[group_col].isna().sum()
    missing_metric = data[metric_col].isna().sum()
    
    if missing_groups > 0:
        warnings.append(f"{missing_groups} missing values in group column")
    
    if missing_metric > 0:
        warnings.append(f"{missing_metric} missing values in metric column")
    
    max_size = group_sizes.max()
    imbalance_ratio = max_size / min_size
    
    if imbalance_ratio > 2:
        warnings.append(f"Groups are imbalanced (ratio={imbalance_ratio:.2f})")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'n_groups': n_groups,
        'group_sizes': group_sizes.to_dict(),
        'total_samples': len(data)
    }

