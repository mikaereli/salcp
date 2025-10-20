import pandas as pd
from typing import Dict, List, Any, Optional


class MethodRecommender:
    def __init__(self, data: pd.DataFrame, profile: Dict[str, Any]):
        self.data = data
        self.profile = profile
        
    def recommend_methods(self, analysis_goal: str = 'explore') -> List[Dict[str, Any]]:
        recommendations = []
        
        if analysis_goal == 'explore':
            recommendations.extend(self._recommend_exploratory())
        elif analysis_goal == 'compare_groups':
            recommendations.extend(self._recommend_comparison())
        elif analysis_goal == 'relationship':
            recommendations.extend(self._recommend_relationship())
        elif analysis_goal == 'predict':
            recommendations.extend(self._recommend_predictive())
        else:
            recommendations.extend(self._auto_recommend())
        
        recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
        
        return recommendations
    
    def _recommend_exploratory(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        recommendations.append({
            'method': 'descriptive_statistics',
            'name': 'Описательная статистика',
            'priority': 'high',
            'reason': 'Базовый анализ для понимания данных',
            'description': 'Расчет основных статистик: среднее, медиана, дисперсия, квартили',
            'applicable_columns': list(self.data.select_dtypes(include=['number']).columns)
        })
        
        numeric_cols = [col for col, prof in self.profile['columns'].items() 
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        
        if numeric_cols:
            recommendations.append({
                'method': 'distribution_analysis',
                'name': 'Анализ распределений',
                'priority': 'high',
                'reason': f'Обнаружено {len(numeric_cols)} числовых столбцов',
                'description': 'Визуализация и тесты на нормальность распределения',
                'applicable_columns': numeric_cols
            })
        
        if len(numeric_cols) >= 2:
            recommendations.append({
                'method': 'correlation_analysis',
                'name': 'Корреляционный анализ',
                'priority': 'medium',
                'reason': 'Множество числовых переменных для анализа взаимосвязей',
                'description': 'Матрица корреляций и тепловая карта',
                'applicable_columns': numeric_cols
            })
        
        outlier_cols = [col for col, prof in self.profile['columns'].items()
                       if prof.get('outliers_percentage', 0) > 5]
        
        if outlier_cols:
            recommendations.append({
                'method': 'outlier_detection',
                'name': 'Обнаружение выбросов',
                'priority': 'medium',
                'reason': f'Обнаружены выбросы в {len(outlier_cols)} столбцах',
                'description': 'Идентификация и анализ аномальных значений',
                'applicable_columns': outlier_cols
            })
        
        return recommendations
    
    def _recommend_comparison(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        categorical_cols = [col for col, prof in self.profile['columns'].items()
                          if prof['inferred_type'] in ['categorical', 'binary']]
        
        numeric_cols = [col for col, prof in self.profile['columns'].items()
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        
        if not categorical_cols or not numeric_cols:
            return recommendations
        
        binary_cols = [col for col, prof in self.profile['columns'].items()
                      if prof['inferred_type'] == 'binary']
        
        if binary_cols and numeric_cols:
            for group_col in binary_cols:
                recommendations.append({
                    'method': 'ab_test',
                    'name': 'A/B тестирование',
                    'priority': 'high',
                    'reason': f'Бинарная группировка по {group_col}',
                    'description': f'Сравнение двух групп по метрикам: {", ".join(numeric_cols[:3])}',
                    'parameters': {
                        'group_column': group_col,
                        'metric_columns': numeric_cols[:5]
                    }
                })
        
        if binary_cols:
            for group_col in binary_cols:
                normal_cols = [col for col in numeric_cols 
                             if self.profile['columns'][col].get('is_normal', False)]
                
                if normal_cols:
                    recommendations.append({
                        'method': 't_test',
                        'name': 'T-тест Стьюдента',
                        'priority': 'high',
                        'reason': 'Нормально распределенные данные с двумя группами',
                        'description': 'Параметрический тест для сравнения средних двух групп',
                        'parameters': {
                            'group_column': group_col,
                            'metric_columns': normal_cols
                        }
                    })
                
                non_normal_cols = [col for col in numeric_cols 
                                 if not self.profile['columns'][col].get('is_normal', True)]
                
                if non_normal_cols:
                    recommendations.append({
                        'method': 'mann_whitney',
                        'name': 'Тест Манна-Уитни',
                        'priority': 'high',
                        'reason': 'Ненормально распределенные данные с двумя группами',
                        'description': 'Непараметрический тест для сравнения двух групп',
                        'parameters': {
                            'group_column': group_col,
                            'metric_columns': non_normal_cols
                        }
                    })
        
        multi_group_cols = [col for col, prof in self.profile['columns'].items()
                           if prof['inferred_type'] == 'categorical' and 
                           2 < prof['unique_count'] < 10]
        
        if multi_group_cols and numeric_cols:
            recommendations.append({
                'method': 'anova',
                'name': 'Дисперсионный анализ (ANOVA)',
                'priority': 'high',
                'reason': 'Сравнение более двух групп',
                'description': 'Тест для сравнения средних нескольких групп',
                'parameters': {
                    'group_columns': multi_group_cols,
                    'metric_columns': numeric_cols
                }
            })
        
        return recommendations
    
    def _recommend_relationship(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        numeric_cols = [col for col, prof in self.profile['columns'].items()
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        
        categorical_cols = [col for col, prof in self.profile['columns'].items()
                          if prof['inferred_type'] in ['categorical']]
        
        if len(numeric_cols) >= 2:
            recommendations.append({
                'method': 'pearson_correlation',
                'name': 'Корреляция Пирсона',
                'priority': 'high',
                'reason': 'Анализ линейных взаимосвязей между числовыми переменными',
                'description': 'Измерение силы и направления линейной связи',
                'applicable_columns': numeric_cols
            })
            
            recommendations.append({
                'method': 'spearman_correlation',
                'name': 'Корреляция Спирмена',
                'priority': 'medium',
                'reason': 'Анализ монотонных взаимосвязей (не обязательно линейных)',
                'description': 'Ранговая корреляция для нелинейных зависимостей',
                'applicable_columns': numeric_cols
            })
        
        if len(categorical_cols) >= 2:
            recommendations.append({
                'method': 'chi_square',
                'name': 'Критерий хи-квадрат',
                'priority': 'high',
                'reason': 'Анализ связи между категориальными переменными',
                'description': 'Тест независимости для категориальных данных',
                'applicable_columns': categorical_cols
            })
        
        if len(numeric_cols) >= 2:
            recommendations.append({
                'method': 'linear_regression',
                'name': 'Линейная регрессия',
                'priority': 'medium',
                'reason': 'Моделирование зависимости между переменными',
                'description': 'Построение модели для предсказания и анализа влияния факторов',
                'applicable_columns': numeric_cols
            })
        
        return recommendations
    
    def _recommend_predictive(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        numeric_cols = [col for col, prof in self.profile['columns'].items()
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        
        binary_cols = [col for col, prof in self.profile['columns'].items()
                      if prof['inferred_type'] == 'binary']
        
        categorical_cols = [col for col, prof in self.profile['columns'].items()
                          if prof['inferred_type'] == 'categorical']
        
        if numeric_cols:
            recommendations.append({
                'method': 'regression_model',
                'name': 'Регрессионная модель',
                'priority': 'high',
                'reason': 'Предсказание числовых значений',
                'description': 'Линейная или полиномиальная регрессия для прогнозирования',
                'potential_targets': numeric_cols
            })
        
        if binary_cols or categorical_cols:
            recommendations.append({
                'method': 'classification_model',
                'name': 'Классификационная модель',
                'priority': 'high',
                'reason': 'Предсказание категорий',
                'description': 'Логистическая регрессия или другие классификаторы',
                'potential_targets': binary_cols + categorical_cols
            })
        
        return recommendations
    
    def _auto_recommend(self) -> List[Dict[str, Any]]:
        recommendations = []
        
        recommendations.extend(self._recommend_exploratory())
        
        categorical_cols = [col for col, prof in self.profile['columns'].items()
                          if prof['inferred_type'] in ['categorical', 'binary']]
        numeric_cols = [col for col, prof in self.profile['columns'].items()
                       if prof['inferred_type'] in ['continuous_numeric', 'discrete_numeric']]
        
        if categorical_cols and numeric_cols:
            recommendations.extend(self._recommend_comparison())
        
        if len(numeric_cols) >= 2 or len(categorical_cols) >= 2:
            recommendations.extend(self._recommend_relationship())
        
        return recommendations
    
    def recommend_test_for_columns(self, col1: str, col2: str) -> Dict[str, Any]:
        prof1 = self.profile['columns'].get(col1)
        prof2 = self.profile['columns'].get(col2)
        
        if not prof1 or not prof2:
            return {'error': 'Column not found in profile'}
        
        type1 = prof1['inferred_type']
        type2 = prof2['inferred_type']
        
        if type1 in ['continuous_numeric', 'discrete_numeric'] and type2 in ['categorical', 'binary']:
            if type2 == 'binary':
                return {
                    'recommended_test': 't_test',
                    'alternative': 'mann_whitney',
                    'reason': 'Сравнение числовой переменной между двумя группами'
                }
            else:
                return {
                    'recommended_test': 'anova',
                    'alternative': 'kruskal_wallis',
                    'reason': 'Сравнение числовой переменной между несколькими группами'
                }
        
        if type1 in ['continuous_numeric', 'discrete_numeric'] and \
           type2 in ['continuous_numeric', 'discrete_numeric']:
            return {
                'recommended_test': 'pearson_correlation',
                'alternative': 'spearman_correlation',
                'reason': 'Анализ взаимосвязи между двумя числовыми переменными'
            }
        
        if type1 in ['categorical', 'binary'] and type2 in ['categorical', 'binary']:
            return {
                'recommended_test': 'chi_square',
                'alternative': 'fisher_exact',
                'reason': 'Анализ связи между категориальными переменными'
            }
        
        return {
            'recommended_test': None,
            'reason': 'Не удалось подобрать подходящий тест для данных типов переменных'
        }

