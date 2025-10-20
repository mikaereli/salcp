import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest


class ABTestEngine:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def run_ab_test(self, group_col: str, metric_col: str,
                   control_group: Any = None,
                   alpha: float = 0.05,
                   metric_type: str = 'continuous') -> Dict[str, Any]:
        groups = self.data[group_col].unique()
        n_groups = len(groups)
        
        if n_groups < 2:
            return {'error': 'Need at least 2 groups for A/B test'}
        
        if control_group is None:
            control_group = groups[0]
        
        if n_groups == 2:
            if metric_type == 'continuous':
                results = self._two_sample_test(group_col, metric_col, 
                                               control_group, list(groups)[1], alpha)
            else:
                results = self._proportion_test(group_col, metric_col,
                                               control_group, list(groups)[1], alpha)
        else:
            if metric_type == 'continuous':
                results = self._multigroup_test(group_col, metric_col, 
                                               control_group, alpha)
            else:
                results = self._multigroup_proportion_test(group_col, metric_col,
                                                          control_group, alpha)
        
        results['practical_significance'] = self._assess_practical_significance(
            group_col, metric_col, control_group
        )
        
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _two_sample_test(self, group_col: str, metric_col: str,
                        control: Any, variant: Any, alpha: float) -> Dict[str, Any]:
        control_data = self.data[self.data[group_col] == control][metric_col].dropna()
        variant_data = self.data[self.data[group_col] == variant][metric_col].dropna()
        
        _, p_norm_c = stats.shapiro(control_data) if len(control_data) < 5000 else (None, 0.5)
        _, p_norm_v = stats.shapiro(variant_data) if len(variant_data) < 5000 else (None, 0.5)
        
        is_normal = (p_norm_c > 0.05 and p_norm_v > 0.05) if p_norm_c and p_norm_v else False
        
        if is_normal:
            _, p_var = stats.levene(control_data, variant_data)
            equal_var = p_var > 0.05
            
            t_stat, p_value = stats.ttest_ind(control_data, variant_data, equal_var=equal_var)
            test_used = 't-test'
            
            pooled_std = np.sqrt(
                ((len(control_data) - 1) * control_data.std()**2 + 
                 (len(variant_data) - 1) * variant_data.std()**2) / 
                (len(control_data) + len(variant_data) - 2)
            )
            effect_size = (variant_data.mean() - control_data.mean()) / pooled_std
        else:
            u_stat, p_value = stats.mannwhitneyu(control_data, variant_data, 
                                                 alternative='two-sided')
            test_used = 'Mann-Whitney U'
            t_stat = u_stat
            effect_size = None
        
        control_mean = control_data.mean()
        variant_mean = variant_data.mean()
        lift_absolute = variant_mean - control_mean
        lift_relative = (lift_absolute / control_mean) * 100 if control_mean != 0 else 0
        
        if is_normal:
            se_diff = np.sqrt(control_data.var()/len(control_data) + 
                            variant_data.var()/len(variant_data))
            ci_lower = lift_absolute - 1.96 * se_diff
            ci_upper = lift_absolute + 1.96 * se_diff
        else:
            ci_lower, ci_upper = None, None
        
        if is_normal and effect_size:
            power_analysis = TTestIndPower()
            actual_power = power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=len(control_data),
                ratio=len(variant_data)/len(control_data),
                alpha=alpha
            )
        else:
            actual_power = None
        
        return {
            'test_type': test_used,
            'n_groups': 2,
            'control_group': str(control),
            'variant_groups': [str(variant)],
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < alpha,
            'alpha': alpha,
            'effect_size': float(effect_size) if effect_size else None,
            'groups': {
                str(control): {
                    'n': len(control_data),
                    'mean': float(control_mean),
                    'std': float(control_data.std()),
                    'median': float(control_data.median())
                },
                str(variant): {
                    'n': len(variant_data),
                    'mean': float(variant_mean),
                    'std': float(variant_data.std()),
                    'median': float(variant_data.median())
                }
            },
            'lift': {
                'absolute': float(lift_absolute),
                'relative_percent': float(lift_relative),
                'confidence_interval': [float(ci_lower), float(ci_upper)] if ci_lower else None
            },
            'statistical_power': float(actual_power) if actual_power else None,
            'assumptions': {
                'normality': is_normal,
                'equal_variance': equal_var if is_normal else None
            }
        }
    
    def _proportion_test(self, group_col: str, metric_col: str,
                        control: Any, variant: Any, alpha: float) -> Dict[str, Any]:
        control_data = self.data[self.data[group_col] == control][metric_col]
        variant_data = self.data[self.data[group_col] == variant][metric_col]
        
        control_successes = control_data.sum()
        variant_successes = variant_data.sum()
        
        control_n = len(control_data)
        variant_n = len(variant_data)
        
        control_prop = control_successes / control_n
        variant_prop = variant_successes / variant_n
        
        counts = np.array([variant_successes, control_successes])
        nobs = np.array([variant_n, control_n])
        z_stat, p_value = proportions_ztest(counts, nobs)
        
        lift_absolute = variant_prop - control_prop
        lift_relative = (lift_absolute / control_prop) * 100 if control_prop != 0 else 0
        
        se = np.sqrt(control_prop * (1-control_prop) / control_n + 
                    variant_prop * (1-variant_prop) / variant_n)
        ci_lower = lift_absolute - 1.96 * se
        ci_upper = lift_absolute + 1.96 * se
        
        return {
            'test_type': 'Z-test for proportions',
            'n_groups': 2,
            'control_group': str(control),
            'variant_groups': [str(variant)],
            'statistic': float(z_stat),
            'p_value': float(p_value),
            'is_significant': p_value < alpha,
            'alpha': alpha,
            'groups': {
                str(control): {
                    'n': control_n,
                    'successes': int(control_successes),
                    'proportion': float(control_prop)
                },
                str(variant): {
                    'n': variant_n,
                    'successes': int(variant_successes),
                    'proportion': float(variant_prop)
                }
            },
            'lift': {
                'absolute': float(lift_absolute),
                'relative_percent': float(lift_relative),
                'confidence_interval': [float(ci_lower), float(ci_upper)]
            }
        }
    
    def _multigroup_test(self, group_col: str, metric_col: str,
                        control: Any, alpha: float) -> Dict[str, Any]:
        groups_data = {}
        for name, group in self.data.groupby(group_col):
            values = group[metric_col].dropna()
            if len(values) > 0:
                groups_data[str(name)] = values
        
        f_stat, p_value = stats.f_oneway(*groups_data.values())
        
        control_mean = groups_data[str(control)].mean()
        groups_info = {}
        
        for name, values in groups_data.items():
            mean = values.mean()
            lift_abs = mean - control_mean
            lift_rel = (lift_abs / control_mean) * 100 if control_mean != 0 else 0
            
            groups_info[name] = {
                'n': len(values),
                'mean': float(mean),
                'std': float(values.std()),
                'lift_absolute': float(lift_abs),
                'lift_relative_percent': float(lift_rel)
            }
        
        return {
            'test_type': 'ANOVA',
            'n_groups': len(groups_data),
            'control_group': str(control),
            'variant_groups': [k for k in groups_data.keys() if k != str(control)],
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'is_significant': p_value < alpha,
            'alpha': alpha,
            'groups': groups_info
        }
    
    def _multigroup_proportion_test(self, group_col: str, metric_col: str,
                                   control: Any, alpha: float) -> Dict[str, Any]:
        contingency_data = []
        group_names = []
        
        for name, group in self.data.groupby(group_col):
            successes = group[metric_col].sum()
            failures = len(group) - successes
            contingency_data.append([successes, failures])
            group_names.append(str(name))
        
        contingency_table = np.array(contingency_data)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        control_idx = group_names.index(str(control))
        control_prop = contingency_data[control_idx][0] / sum(contingency_data[control_idx])
        
        groups_info = {}
        for i, name in enumerate(group_names):
            n = sum(contingency_data[i])
            successes = contingency_data[i][0]
            prop = successes / n
            lift_abs = prop - control_prop
            lift_rel = (lift_abs / control_prop) * 100 if control_prop != 0 else 0
            
            groups_info[name] = {
                'n': n,
                'successes': successes,
                'proportion': float(prop),
                'lift_absolute': float(lift_abs),
                'lift_relative_percent': float(lift_rel)
            }
        
        return {
            'test_type': 'Chi-square test',
            'n_groups': len(group_names),
            'control_group': str(control),
            'variant_groups': [k for k in group_names if k != str(control)],
            'statistic': float(chi2),
            'p_value': float(p_value),
            'is_significant': p_value < alpha,
            'alpha': alpha,
            'groups': groups_info
        }
    
    def _assess_practical_significance(self, group_col: str, metric_col: str,
                                      control: Any) -> Dict[str, Any]:
        control_data = self.data[self.data[group_col] == control][metric_col].dropna()
        control_mean = control_data.mean()
        
        practical_threshold = 0.05
        
        variants = []
        for variant in self.data[group_col].unique():
            if variant == control:
                continue
            
            variant_data = self.data[self.data[group_col] == variant][metric_col].dropna()
            variant_mean = variant_data.mean()
            
            relative_change = abs(variant_mean - control_mean) / control_mean if control_mean != 0 else 0
            
            variants.append({
                'variant': str(variant),
                'relative_change': float(relative_change),
                'is_practically_significant': relative_change >= practical_threshold
            })
        
        return {
            'threshold': practical_threshold,
            'variants': variants
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if results['is_significant']:
            if 'lift' in results:
                if results['lift']['relative_percent'] > 0:
                    recommendations.append(
                        f"ОК! Вариант показывает статистически значимое улучшение на "
                        f"{results['lift']['relative_percent']:.2f}%. Рекомендуется внедрение."
                    )
                else:
                    recommendations.append(
                        f"? Вариант показывает статистически значимое ухудшение на "
                        f"{abs(results['lift']['relative_percent']):.2f}%. Не рекомендуется внедрение."
                    )
            
            if results.get('statistical_power'):
                power = results['statistical_power']
                if power < 0.8:
                    recommendations.append(
                        f"⚠️ Статистическая мощность теста низкая ({power:.2%}). "
                        f"Рекомендуется увеличить размер выборки."
                    )
        else:
            recommendations.append(
                "НЕТ! Статистически значимых различий не обнаружено. "
                "Рекомендуется продолжить тест или пересмотреть гипотезу."
            )
            
            if 'groups' in results:
                min_sample = min(g['n'] for g in results['groups'].values())
                if min_sample < 100:
                    recommendations.append(
                        f"!? Размер выборки мал (минимум {min_sample}). "
                        f"Рекомендуется собрать больше данных."
                    )
        
        return recommendations
    
    def calculate_sample_size(self, baseline_rate: float, 
                            mde: float,
                            alpha: float = 0.05,
                            power: float = 0.8) -> Dict[str, Any]:
        effect_size = mde
        
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0
        )
        
        return {
            'sample_size_per_group': int(np.ceil(sample_size)),
            'total_sample_size': int(np.ceil(sample_size * 2)),
            'baseline_rate': baseline_rate,
            'mde': mde,
            'alpha': alpha,
            'power': power
        }

