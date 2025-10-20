from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QGroupBox, QLabel, QTextEdit, QComboBox, QFormLayout,
                             QMessageBox, QScrollArea, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd
from typing import Optional, Dict
import json

from core.statistical_tests import StatisticalTests
from core.ab_testing import ABTestEngine


class AnalysisView(QWidget):
    def __init__(self):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None
        self.data_profile: Optional[Dict] = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        left_panel = self.create_controls_panel()
        
        right_panel = self.create_results_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
    def create_controls_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc_group = QGroupBox("Описательная статистика")
        desc_layout = QVBoxLayout()
        
        desc_btn = QPushButton("Показать описательную статистику")
        desc_btn.clicked.connect(self.show_descriptive_stats)
        desc_layout.addWidget(desc_btn)
        
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)
        
        ab_group = QGroupBox("A/B Тестирование")
        ab_layout = QFormLayout()
        
        self.ab_group_col = QComboBox()
        ab_layout.addRow("Столбец с группами:", self.ab_group_col)
        
        self.ab_metric_col = QComboBox()
        ab_layout.addRow("Метрика:", self.ab_metric_col)
        
        self.ab_metric_type = QComboBox()
        self.ab_metric_type.addItems(['continuous', 'binary'])
        ab_layout.addRow("Тип метрики:", self.ab_metric_type)
        
        ab_btn = QPushButton("Выполнить A/B тест")
        ab_btn.clicked.connect(self.run_ab_test)
        ab_layout.addRow(ab_btn)
        
        ab_group.setLayout(ab_layout)
        layout.addWidget(ab_group)
        
        corr_group = QGroupBox("Корреляционный анализ")
        corr_layout = QVBoxLayout()
        
        self.corr_method = QComboBox()
        self.corr_method.addItems(['pearson', 'spearman', 'kendall'])
        corr_layout.addWidget(QLabel("Метод:"))
        corr_layout.addWidget(self.corr_method)
        
        corr_btn = QPushButton("Показать корреляции")
        corr_btn.clicked.connect(self.show_correlations)
        corr_layout.addWidget(corr_btn)
        
        corr_group.setLayout(corr_layout)
        layout.addWidget(corr_group)
        
        ttest_group = QGroupBox("T-Test (сравнение 2 групп)")
        ttest_layout = QFormLayout()
        
        self.ttest_group_col = QComboBox()
        ttest_layout.addRow("Столбец с группами:", self.ttest_group_col)
        
        self.ttest_value_col = QComboBox()
        ttest_layout.addRow("Столбец со значениями:", self.ttest_value_col)
        
        self.ttest_group1 = QComboBox()
        ttest_layout.addRow("Группа 1:", self.ttest_group1)
        
        self.ttest_group2 = QComboBox()
        ttest_layout.addRow("Группа 2:", self.ttest_group2)
        
        ttest_btn = QPushButton("Выполнить T-Test")
        ttest_btn.clicked.connect(self.run_ttest)
        ttest_layout.addRow(ttest_btn)
        
        ttest_group.setLayout(ttest_layout)
        layout.addWidget(ttest_group)
        
        anova_group = QGroupBox("ANOVA (сравнение 3+ групп)")
        anova_layout = QFormLayout()
        
        self.anova_group_col = QComboBox()
        anova_layout.addRow("Столбец с группами:", self.anova_group_col)
        
        self.anova_value_col = QComboBox()
        anova_layout.addRow("Столбец со значениями:", self.anova_value_col)
        
        anova_btn = QPushButton("Выполнить ANOVA")
        anova_btn.clicked.connect(self.run_anova)
        anova_layout.addRow(anova_btn)
        
        anova_group.setLayout(anova_layout)
        layout.addWidget(anova_group)
        
        chi_group = QGroupBox("Chi-Square (категориальные данные)")
        chi_layout = QFormLayout()
        
        self.chi_col1 = QComboBox()
        chi_layout.addRow("Столбец 1:", self.chi_col1)
        
        self.chi_col2 = QComboBox()
        chi_layout.addRow("Столбец 2:", self.chi_col2)
        
        chi_btn = QPushButton("Выполнить Chi-Square")
        chi_btn.clicked.connect(self.run_chi_square)
        chi_layout.addRow(chi_btn)
        
        chi_group.setLayout(chi_layout)
        layout.addWidget(chi_group)
        
        layout.addStretch()
        
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(scroll)
        
        return container
        
    def create_results_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("Результаты анализа")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText("Выберите анализ и нажмите соответствующую кнопку")
        layout.addWidget(self.results_text)
        
        return widget
        
    def set_data(self, data: pd.DataFrame, profile: Dict):
        self.data = data
        self.data_profile = profile
        
        all_cols = list(data.columns)
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        categorical_cols = [col for col, prof in profile['columns'].items()
                           if prof['inferred_type'] in ['categorical', 'binary']]
        
        self.ab_group_col.clear()
        self.ab_group_col.addItems(categorical_cols)
        self.ab_metric_col.clear()
        self.ab_metric_col.addItems(numeric_cols)
        
        self.ttest_group_col.clear()
        self.ttest_group_col.addItems(categorical_cols)
        self.ttest_value_col.clear()
        self.ttest_value_col.addItems(numeric_cols)
        
        self.anova_group_col.clear()
        self.anova_group_col.addItems(categorical_cols)
        self.anova_value_col.clear()
        self.anova_value_col.addItems(numeric_cols)
        
        self.chi_col1.clear()
        self.chi_col1.addItems(categorical_cols)
        self.chi_col2.clear()
        self.chi_col2.addItems(categorical_cols)
        
        self.ttest_group_col.currentTextChanged.connect(self.update_ttest_groups)
        if categorical_cols:
            self.update_ttest_groups(categorical_cols[0])
        
    def update_ttest_groups(self, group_col: str):
        if not self.data is None and group_col in self.data.columns:
            unique_groups = self.data[group_col].unique().tolist()
            self.ttest_group1.clear()
            self.ttest_group1.addItems([str(g) for g in unique_groups])
            self.ttest_group2.clear()
            self.ttest_group2.addItems([str(g) for g in unique_groups])
            if len(unique_groups) > 1:
                self.ttest_group2.setCurrentIndex(1)
        
    def show_descriptive_stats(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        stats = self.data[numeric_cols].describe()
        
        result_text = "=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===\n\n"
        result_text += stats.to_string()
        
        self.results_text.setPlainText(result_text)
        
    def run_ab_test(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        group_col = self.ab_group_col.currentText()
        metric_col = self.ab_metric_col.currentText()
        metric_type = self.ab_metric_type.currentText()
        
        if not group_col or not metric_col:
            QMessageBox.warning(self, "Ошибка", "Выберите столбцы для анализа")
            return
        
        try:
            engine = ABTestEngine(self.data)
            result = engine.run_ab_test(group_col, metric_col, metric_type=metric_type)
            
            result_text = self.format_ab_test_result(result)
            self.results_text.setPlainText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка выполнения A/B теста:\n{str(e)}")
    
    def show_correlations(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        method = self.corr_method.currentText()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Недостаточно данных", 
                              "Нужно минимум 2 числовых столбца")
            return
        
        corr_matrix = self.data[numeric_cols].corr(method=method)
        
        result_text = f"=== КОРРЕЛЯЦИОННАЯ МАТРИЦА ({method.upper()}) ===\n\n"
        result_text += corr_matrix.to_string()
        result_text += "\n\n=== СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7) ===\n\n"
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    result_text += f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}\n"
        
        self.results_text.setPlainText(result_text)
        
    def run_ttest(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        group_col = self.ttest_group_col.currentText()
        value_col = self.ttest_value_col.currentText()
        group1 = self.ttest_group1.currentText()
        group2 = self.ttest_group2.currentText()
        
        if not all([group_col, value_col, group1, group2]):
            QMessageBox.warning(self, "Ошибка", "Заполните все поля")
            return
        
        if group1 == group2:
            QMessageBox.warning(self, "Ошибка", "Выберите разные группы")
            return
        
        try:
            tests = StatisticalTests(self.data)
            result = tests.t_test(group_col, value_col, group1, group2)
            
            result_text = self.format_ttest_result(result)
            self.results_text.setPlainText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка выполнения T-теста:\n{str(e)}")
    
    def run_anova(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        group_col = self.anova_group_col.currentText()
        value_col = self.anova_value_col.currentText()
        
        if not all([group_col, value_col]):
            QMessageBox.warning(self, "Ошибка", "Выберите столбцы")
            return
        
        try:
            tests = StatisticalTests(self.data)
            result = tests.anova(group_col, value_col)
            
            result_text = self.format_anova_result(result)
            self.results_text.setPlainText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка выполнения ANOVA:\n{str(e)}")
    
    def run_chi_square(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        col1 = self.chi_col1.currentText()
        col2 = self.chi_col2.currentText()
        
        if not all([col1, col2]):
            QMessageBox.warning(self, "Ошибка", "Выберите столбцы")
            return
        
        if col1 == col2:
            QMessageBox.warning(self, "Ошибка", "Выберите разные столбцы")
            return
        
        try:
            tests = StatisticalTests(self.data)
            result = tests.chi_square_test(col1, col2)
            
            result_text = self.format_chi_square_result(result)
            self.results_text.setPlainText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка выполнения Chi-square:\n{str(e)}")
    
    def format_ab_test_result(self, result: Dict) -> str:
        text = "=== РЕЗУЛЬТАТЫ A/B ТЕСТИРОВАНИЯ ===\n\n"
        text += f"Тест: {result['test_type']}\n"
        text += f"Количество групп: {result['n_groups']}\n"
        text += f"Контрольная группа: {result['control_group']}\n\n"
        
        text += "--- СТАТИСТИКА ПО ГРУППАМ ---\n\n"
        for group_name, group_stats in result['groups'].items():
            text += f"{group_name}:\n"
            text += f"  Размер выборки: {group_stats['n']}\n"
            if 'mean' in group_stats:
                text += f"  Среднее: {group_stats['mean']:.4f}\n"
                text += f"  Ст. отклонение: {group_stats['std']:.4f}\n"
            if 'proportion' in group_stats:
                text += f"  Пропорция: {group_stats['proportion']:.4f}\n"
            text += "\n"
        
        text += "--- РЕЗУЛЬТАТЫ ТЕСТА ---\n\n"
        text += f"Статистика: {result['statistic']:.4f}\n"
        text += f"P-value: {result['p_value']:.4f}\n"
        text += f"Уровень значимости (alpha): {result['alpha']}\n"
        text += f"Статистически значимо: {'ДА' if result['is_significant'] else 'НЕТ'}\n\n"
        
        if 'lift' in result:
            text += "--- РАЗМЕР ЭФФЕКТА ---\n\n"
            text += f"Абсолютный Lift: {result['lift']['absolute']:.4f}\n"
            text += f"Относительный Lift: {result['lift']['relative_percent']:.2f}%\n\n"
        
        if 'recommendations' in result:
            text += "--- РЕКОМЕНДАЦИИ ---\n\n"
            for rec in result['recommendations']:
                text += f"  {rec}\n"
        
        return text
    
    def format_ttest_result(self, result: Dict) -> str:
        text = "=== РЕЗУЛЬТАТЫ T-ТЕСТА ===\n\n"
        text += f"Тест: {result['test']}\n\n"
        
        text += "--- ГРУППЫ ---\n\n"
        for key in ['group1', 'group2']:
            g = result[key]
            text += f"{g['name']}:\n"
            text += f"  N: {g['n']}\n"
            text += f"  Среднее: {g['mean']:.4f}\n"
            text += f"  Ст. откл.: {g['std']:.4f}\n\n"
        
        text += "--- РЕЗУЛЬТАТЫ ---\n\n"
        text += f"Статистика: {result['statistic']:.4f}\n"
        text += f"P-value: {result['p_value']:.4f}\n"
        text += f"Значимо: {'ДА' if result['significant'] else 'НЕТ'}\n"
        text += f"Cohen's d: {result['effect_size']:.4f} ({result['effect_interpretation']})\n\n"
        
        text += f"ИНТЕРПРЕТАЦИЯ: {result['interpretation']}\n"
        
        return text
    
    def format_anova_result(self, result: Dict) -> str:
        text = "=== РЕЗУЛЬТАТЫ ANOVA ===\n\n"
        text += f"F-статистика: {result['statistic']:.4f}\n"
        text += f"P-value: {result['p_value']:.4f}\n"
        text += f"Значимо: {'ДА' if result['significant'] else 'НЕТ'}\n\n"
        
        text += "--- ГРУППЫ ---\n\n"
        for g in result['group_statistics']:
            text += f"{g['name']}:\n"
            text += f"  N: {g['n']}\n"
            text += f"  Среднее: {g['mean']:.4f}\n"
            text += f"  Ст. откл.: {g['std']:.4f}\n\n"
        
        text += f"ИНТЕРПРЕТАЦИЯ: {result['interpretation']}\n"
        
        return text
    
    def format_chi_square_result(self, result: Dict) -> str:
        text = "=== РЕЗУЛЬТАТЫ CHI-SQUARE ТЕСТА ===\n\n"
        text += f"Chi-square статистика: {result['statistic']:.4f}\n"
        text += f"P-value: {result['p_value']:.4f}\n"
        text += f"Степени свободы: {result['degrees_of_freedom']}\n"
        text += f"Значимо: {'ДА' if result['significant'] else 'НЕТ'}\n"
        text += f"Cramér's V: {result['effect_size']:.4f} ({result['effect_interpretation']})\n\n"
        
        text += f"ИНТЕРПРЕТАЦИЯ: {result['interpretation']}\n"
        
        return text

