import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    def __init__(self, data: pd.DataFrame, profile: Dict[str, Any]):
        self.data = data.copy()
        self.profile = profile
        self.cleaning_log: List[Dict[str, Any]] = []
        
    def auto_clean(self, aggressive: bool = False) -> pd.DataFrame:
        if self.profile['overview']['duplicate_rows'] > 0:
            self._remove_duplicates()
        
        for issue in self.profile['data_quality']['issues']:
            if issue['type'] == 'constant_columns':
                self._remove_constant_columns(issue['columns'])
        
        self._handle_missing_values(aggressive)
        
        if aggressive:
            self._handle_outliers()
        
        return self.data
    
    def _remove_duplicates(self):
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        
        self.cleaning_log.append({
            'action': 'remove_duplicates',
            'rows_removed': before - after,
            'description': f'Удалено {before - after} дубликатов строк'
        })
    
    def _remove_constant_columns(self, columns: List[str]):
        self.data = self.data.drop(columns=columns)
        
        self.cleaning_log.append({
            'action': 'remove_constant_columns',
            'columns_removed': columns,
            'description': f'Удалено {len(columns)} столбцов с константными значениями'
        })
    
    def _handle_missing_values(self, aggressive: bool = False):
        for col, col_profile in self.profile['columns'].items():
            if col not in self.data.columns:
                continue
                
            missing_pct = col_profile['missing_percentage']
            
            if missing_pct == 0:
                continue
            
            if missing_pct > 70:
                self.data = self.data.drop(columns=[col])
                self.cleaning_log.append({
                    'action': 'drop_column',
                    'column': col,
                    'reason': f'Слишком много пропусков ({missing_pct:.1f}%)',
                    'description': f'Удален столбец {col} из-за {missing_pct:.1f}% пропусков'
                })
                continue
            
            col_type = col_profile['inferred_type']
            
            if col_type in ['continuous_numeric', 'discrete_numeric']:
                if missing_pct < 5:
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    method = 'median'
                elif aggressive and missing_pct < 30:
                    self._knn_impute([col])
                    method = 'KNN'
                else:
                    mean_val = self.data[col].mean()
                    self.data[col].fillna(mean_val, inplace=True)
                    method = 'mean'
                
                self.cleaning_log.append({
                    'action': 'impute_numeric',
                    'column': col,
                    'method': method,
                    'description': f'Заполнены пропуски в {col} методом {method}'
                })
            
            elif col_type in ['categorical', 'high_cardinality_categorical']:
                mode_val = self.data[col].mode()
                if len(mode_val) > 0:
                    self.data[col].fillna(mode_val[0], inplace=True)
                    self.cleaning_log.append({
                        'action': 'impute_categorical',
                        'column': col,
                        'method': 'mode',
                        'description': f'Заполнены пропуски в {col} модой'
                    })
            
            elif col_type == 'binary':
                mode_val = self.data[col].mode()
                if len(mode_val) > 0:
                    self.data[col].fillna(mode_val[0], inplace=True)
                    self.cleaning_log.append({
                        'action': 'impute_binary',
                        'column': col,
                        'method': 'mode',
                        'description': f'Заполнены пропуски в {col} модой'
                    })
    
    def _knn_impute(self, columns: List[str]):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
        
        imputer = KNNImputer(n_neighbors=5)
        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
    
    def _handle_outliers(self):
        for col, col_profile in self.profile['columns'].items():
            if col not in self.data.columns:
                continue
            
            col_type = col_profile.get('inferred_type')
            if col_type not in ['continuous_numeric']:
                continue
            
            outliers_pct = col_profile.get('outliers_percentage', 0)
            if outliers_pct > 5:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                
                self.cleaning_log.append({
                    'action': 'cap_outliers',
                    'column': col,
                    'bounds': [float(lower_bound), float(upper_bound)],
                    'description': f'Ограничены выбросы в {col} ({outliers_pct:.1f}%)'
                })
    
    def get_cleaning_log(self) -> List[Dict[str, Any]]:
        return self.cleaning_log
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.data

