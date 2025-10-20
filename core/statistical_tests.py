import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest


class StatisticalTests:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def t_test(self, group_col: str, value_col: str, 
               group1: Any, group2: Any) -> Dict[str, Any]:
        sample1 = self.data[self.data[group_col] == group1][value_col].dropna()
        sample2 = self.data[self.data[group_col] == group2][value_col].dropna()
        
        _, p_norm1 = stats.shapiro(sample1) if len(sample1) < 5000 else (None, None)
        _, p_norm2 = stats.shapiro(sample2) if len(sample2) < 5000 else (None, None)
        
        _, p_var = stats.levene(sample1, sample2)
        equal_var = p_var > 0.05
        
        t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
        
        cohens_d = (sample1.mean() - sample2.mean()) / np.sqrt(
            ((len(sample1) - 1) * sample1.std()**2 + 
             (len(sample2) - 1) * sample2.std()**2) / 
            (len(sample1) + len(sample2) - 2)
        )
        
        return {
            'test': 't-test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float(cohens_d),
            'effect_interpretation': self._interpret_cohens_d(cohens_d),
            'group1': {
                'name': str(group1),
                'n': len(sample1),
                'mean': float(sample1.mean()),
                'std': float(sample1.std())
            },
            'group2': {
                'name': str(group2),
                'n': len(sample2),
                'mean': float(sample2.mean()),
                'std': float(sample2.std())
            },
            'assumptions': {
                'normality_group1_p': float(p_norm1) if p_norm1 else None,
                'normality_group2_p': float(p_norm2) if p_norm2 else None,
                'equal_variance_p': float(p_var),
                'equal_variance': equal_var
            },
            'interpretation': self._interpret_result(p_value, group1, group2)
        }
    
    def mann_whitney_test(self, group_col: str, value_col: str,
                         group1: Any, group2: Any) -> Dict[str, Any]:
        sample1 = self.data[self.data[group_col] == group1][value_col].dropna()
        sample2 = self.data[self.data[group_col] == group2][value_col].dropna()
        
        u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'group1': {
                'name': str(group1),
                'n': len(sample1),
                'median': float(sample1.median())
            },
            'group2': {
                'name': str(group2),
                'n': len(sample2),
                'median': float(sample2.median())
            },
            'interpretation': f"Медианы {'различаются' if p_value < 0.05 else 'не различаются'} статистически значимо (p={p_value:.4f})"
        }
    
    def anova(self, group_col: str, value_col: str) -> Dict[str, Any]:
        groups = []
        group_names = []
        
        for name, group in self.data.groupby(group_col):
            values = group[value_col].dropna()
            if len(values) > 0:
                groups.append(values)
                group_names.append(str(name))
        
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for ANOVA'}
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        posthoc = None
        if p_value < 0.05:
            posthoc = self._tukey_hsd(group_col, value_col)
        
        group_stats = []
        for name, values in zip(group_names, groups):
            group_stats.append({
                'name': name,
                'n': len(values),
                'mean': float(values.mean()),
                'std': float(values.std())
            })
        
        return {
            'test': 'One-way ANOVA',
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'group_statistics': group_stats,
            'posthoc': posthoc,
            'interpretation': f"Есть {'статистически значимые' if p_value < 0.05 else 'незначимые'} различия между группами (p={p_value:.4f})"
        }
    
    def _tukey_hsd(self, group_col: str, value_col: str) -> Dict[str, Any]:
        tukey_result = pairwise_tukeyhsd(
            self.data[value_col].dropna(),
            self.data.loc[self.data[value_col].notna(), group_col]
        )
        
        comparisons = []
        for i in range(len(tukey_result.summary().data) - 1):
            row = tukey_result.summary().data[i + 1]
            comparisons.append({
                'group1': str(row[0]),
                'group2': str(row[1]),
                'mean_diff': float(row[2]),
                'p_value': float(row[3]),
                'significant': row[5] == 'True'
            })
        
        return {
            'method': 'Tukey HSD',
            'comparisons': comparisons
        }
    
    def chi_square_test(self, col1: str, col2: str) -> Dict[str, Any]:
        contingency_table = pd.crosstab(self.data[col1], self.data[col2])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'test': 'Chi-square',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'effect_size': float(cramers_v),
            'effect_interpretation': self._interpret_cramers_v(cramers_v),
            'contingency_table': contingency_table.to_dict(),
            'interpretation': f"Переменные {'зависимы' if p_value < 0.05 else 'независимы'} (p={p_value:.4f})"
        }
    
    def correlation_test(self, col1: str, col2: str, 
                        method: str = 'pearson') -> Dict[str, Any]:
        data1 = self.data[col1].dropna()
        data2 = self.data[col2].dropna()
        
        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx]
        data2 = data2.loc[common_idx]
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(data1, data2)
        else:
            corr, p_value = stats.spearmanr(data1, data2)
        
        return {
            'test': f'{method.capitalize()} correlation',
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'strength': self._interpret_correlation(corr),
            'interpretation': f"Корреляция {self._interpret_correlation(corr)} (r={corr:.3f}, p={p_value:.4f})"
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_cramers_v(self, v: float) -> str:
        if v < 0.1:
            return 'negligible'
        elif v < 0.3:
            return 'small'
        elif v < 0.5:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_correlation(self, r: float) -> str:
        abs_r = abs(r)
        if abs_r < 0.3:
            return 'weak'
        elif abs_r < 0.7:
            return 'moderate'
        else:
            return 'strong'
    
    def _interpret_result(self, p_value: float, group1: Any, group2: Any) -> str:
        if p_value < 0.05:
            return f"Обнаружены статистически значимые различия между {group1} и {group2} (p={p_value:.4f})"
        else:
            return f"Статистически значимых различий между {group1} и {group2} не обнаружено (p={p_value:.4f})"

