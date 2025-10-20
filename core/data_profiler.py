import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats


class DataProfiler:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.profile: Dict[str, Any] = {}
        
    def profile_data(self) -> Dict[str, Any]:
        self.profile = {
            'overview': self._get_overview(),
            'columns': self._profile_columns(),
            'correlations': self._analyze_correlations(),
            'data_quality': self._assess_data_quality(),
            'recommendations': []
        }
        
        self.profile['recommendations'] = self._generate_recommendations()
        
        return self.profile
    
    def _get_overview(self) -> Dict[str, Any]:
        return {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicate_rows': self.data.duplicated().sum(),
            'total_missing_values': self.data.isna().sum().sum()
        }
    
    def _profile_columns(self) -> Dict[str, Dict[str, Any]]:
        columns_profile = {}
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            profile = {
                'name': col,
                'dtype': str(col_data.dtype),
                'inferred_type': self._infer_column_type(col_data),
                'missing_count': col_data.isna().sum(),
                'missing_percentage': (col_data.isna().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                profile.update(self._numeric_stats(col_data))
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                profile.update(self._categorical_stats(col_data))
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                profile.update(self._datetime_stats(col_data))
            
            columns_profile[col] = profile
        
        return columns_profile
    
    def _infer_column_type(self, series: pd.Series) -> str:
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
    
    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {}
        
        stats_dict = {
            'mean': float(clean_series.mean()),
            'median': float(clean_series.median()),
            'std': float(clean_series.std()),
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'q25': float(clean_series.quantile(0.25)),
            'q75': float(clean_series.quantile(0.75)),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis())
        }
        
        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((clean_series < lower_bound) | (clean_series > upper_bound)).sum()
        
        stats_dict['outliers_count'] = int(outliers)
        stats_dict['outliers_percentage'] = float((outliers / len(clean_series)) * 100)
        
        if len(clean_series) < 5000:
            try:
                _, p_value = stats.shapiro(clean_series)
                stats_dict['normality_p_value'] = float(p_value)
                stats_dict['is_normal'] = p_value > 0.05
            except:
                stats_dict['normality_p_value'] = None
                stats_dict['is_normal'] = None
        
        return stats_dict
    
    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {}
        
        value_counts = clean_series.value_counts()
        
        return {
            'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'most_common_percentage': float((value_counts.iloc[0] / len(clean_series)) * 100) if len(value_counts) > 0 else 0,
            'categories': list(value_counts.index[:10])
        }
    
    def _datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {}
        
        return {
            'min_date': str(clean_series.min()),
            'max_date': str(clean_series.max()),
            'date_range_days': (clean_series.max() - clean_series.min()).days
        }
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'available': False, 'message': 'Not enough numeric columns'}
        
        corr_matrix = self.data[numeric_cols].corr()
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'available': True,
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_corr
        }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        issues = []
        
        missing_threshold = 0.3
        high_missing = []
        for col in self.data.columns:
            missing_pct = (self.data[col].isna().sum() / len(self.data))
            if missing_pct > missing_threshold:
                high_missing.append({
                    'column': col,
                    'missing_percentage': float(missing_pct * 100)
                })
        
        if high_missing:
            issues.append({
                'type': 'high_missing_values',
                'severity': 'high',
                'columns': high_missing
            })
        
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            issues.append({
                'type': 'duplicate_rows',
                'severity': 'medium',
                'count': int(duplicates),
                'percentage': float((duplicates / len(self.data)) * 100)
            })
        
        constant_cols = [col for col in self.data.columns if self.data[col].nunique() <= 1]
        if constant_cols:
            issues.append({
                'type': 'constant_columns',
                'severity': 'low',
                'columns': constant_cols
            })
        
        quality_score = 100
        quality_score -= len(high_missing) * 10
        quality_score -= (duplicates / len(self.data)) * 20
        quality_score -= len(constant_cols) * 5
        quality_score = max(0, quality_score)
        
        return {
            'quality_score': float(quality_score),
            'issues': issues,
            'needs_cleaning': len(issues) > 0
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        recommendations = []
        
        if self.profile['data_quality']['needs_cleaning']:
            for issue in self.profile['data_quality']['issues']:
                if issue['type'] == 'high_missing_values':
                    recommendations.append({
                        'category': 'data_cleaning',
                        'action': 'handle_missing_values',
                        'priority': 'high',
                        'description': f"Обработать пропущенные значения в столбцах с высоким процентом пропусков"
                    })
                elif issue['type'] == 'duplicate_rows':
                    recommendations.append({
                        'category': 'data_cleaning',
                        'action': 'remove_duplicates',
                        'priority': 'medium',
                        'description': f"Удалить {issue['count']} дубликатов строк"
                    })
        
        numeric_cols = [col for col, prof in self.profile['columns'].items() 
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        categorical_cols = [col for col, prof in self.profile['columns'].items() 
                           if prof['inferred_type'] == 'categorical']
        binary_cols = [col for col, prof in self.profile['columns'].items() 
                      if prof['inferred_type'] == 'binary']
        
        if binary_cols and numeric_cols:
            recommendations.append({
                'category': 'statistical_analysis',
                'action': 'ab_test',
                'priority': 'high',
                'description': f"Провести A/B тестирование: группы по {binary_cols[0]}, метрики {', '.join(numeric_cols[:3])}"
            })
        
        if len(numeric_cols) >= 2:
            recommendations.append({
                'category': 'statistical_analysis',
                'action': 'correlation_analysis',
                'priority': 'medium',
                'description': f"Анализ корреляций между числовыми переменными"
            })
        
        if len(categorical_cols) >= 2:
            recommendations.append({
                'category': 'statistical_analysis',
                'action': 'chi_square_test',
                'priority': 'medium',
                'description': f"Тест независимости хи-квадрат для категориальных переменных"
            })
        
        return recommendations

