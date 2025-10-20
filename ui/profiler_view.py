from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QGroupBox, QLabel, QScrollArea, QGridLayout,
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QPushButton, QListWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from typing import Optional, Dict
import json


class ProfilerView(QWidget):
    def __init__(self):
        super().__init__()
        self.profile: Optional[Dict] = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.overview_widget = self.create_overview_widget()
        self.tabs.addTab(self.overview_widget, "Обзор")
        
        self.columns_widget = self.create_columns_widget()
        self.tabs.addTab(self.columns_widget, "Столбцы")
        
        self.quality_widget = self.create_quality_widget()
        self.tabs.addTab(self.quality_widget, "Качество данных")
        
        self.recommendations_widget = self.create_recommendations_widget()
        self.tabs.addTab(self.recommendations_widget, "Рекомендации")
        
    def create_overview_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        overview_group = QGroupBox("Общая информация")
        overview_layout = QGridLayout()
        overview_group.setLayout(overview_layout)
        
        self.overview_labels = {
            'rows': QLabel("-"),
            'columns': QLabel("-"),
            'memory': QLabel("-"),
            'duplicates': QLabel("-"),
            'missing': QLabel("-")
        }
        
        row = 0
        for key, label in self.overview_labels.items():
            title = {
                'rows': 'Строк:',
                'columns': 'Столбцов:',
                'memory': 'Память:',
                'duplicates': 'Дубликатов:',
                'missing': 'Пропусков:'
            }[key]
            
            title_label = QLabel(title)
            font = QFont()
            font.setBold(True)
            title_label.setFont(font)
            
            overview_layout.addWidget(title_label, row, 0)
            overview_layout.addWidget(label, row, 1)
            row += 1
        
        layout.addWidget(overview_group)
        
        sample_group = QGroupBox("Характеристики выборки")
        sample_layout = QGridLayout()
        sample_group.setLayout(sample_layout)
        
        self.sample_labels = {
            'numeric_cols': QLabel("-"),
            'categorical_cols': QLabel("-"),
            'binary_cols': QLabel("-"),
            'text_cols': QLabel("-"),
            'datetime_cols': QLabel("-")
        }
        
        row = 0
        for key, label in self.sample_labels.items():
            title = {
                'numeric_cols': 'Числовых столбцов:',
                'categorical_cols': 'Категориальных столбцов:',
                'binary_cols': 'Бинарных столбцов:',
                'text_cols': 'Текстовых столбцов:',
                'datetime_cols': 'Дата/время столбцов:'
            }[key]
            
            title_label = QLabel(title)
            font = QFont()
            font.setBold(True)
            title_label.setFont(font)
            
            sample_layout.addWidget(title_label, row, 0)
            sample_layout.addWidget(label, row, 1)
            row += 1
        
        layout.addWidget(sample_group)
        
        analysis_group = QGroupBox("Статистический анализ выборки")
        analysis_layout = QVBoxLayout()
        analysis_group.setLayout(analysis_layout)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(200)
        self.analysis_text.setPlainText("Загрузите данные для просмотра статистического анализа")
        analysis_layout.addWidget(self.analysis_text)
        
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        
        return widget
    
    def create_columns_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.columns_table = QTableWidget()
        self.columns_table.setColumnCount(7)
        self.columns_table.setHorizontalHeaderLabels([
            "Столбец", "Тип", "Распознанный тип", "Уникальных", 
            "Пропусков (%)", "Статистики", "Проблемы"
        ])
        self.columns_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.columns_table)
        
        return widget
    
    def create_quality_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.quality_score_label = QLabel("Оценка качества: -")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.quality_score_label.setFont(font)
        layout.addWidget(self.quality_score_label)
        
        issues_group = QGroupBox("Обнаруженные проблемы")
        issues_layout = QVBoxLayout()
        issues_group.setLayout(issues_layout)
        
        self.issues_list = QListWidget()
        issues_layout.addWidget(self.issues_list)
        
        layout.addWidget(issues_group)
        
        return widget
    
    def create_recommendations_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("Рекомендуемые действия на основе анализа данных:")
        layout.addWidget(info_label)
        
        self.recommendations_list = QListWidget()
        layout.addWidget(self.recommendations_list)
        
        return widget
    
    def set_profile(self, profile: Dict):
        self.profile = profile
        self.update_ui()
        
    def update_ui(self):
        if not self.profile:
            return
        
        overview = self.profile.get('overview', {})
        self.overview_labels['rows'].setText(f"{overview.get('rows', 0):,}")
        self.overview_labels['columns'].setText(f"{overview.get('columns', 0)}")
        self.overview_labels['memory'].setText(f"{overview.get('memory_usage_mb', 0):.2f} МБ")
        self.overview_labels['duplicates'].setText(f"{overview.get('duplicate_rows', 0):,}")
        self.overview_labels['missing'].setText(f"{overview.get('total_missing_values', 0):,}")
        
        columns = self.profile.get('columns', {})
        
        numeric_count = len([c for c, p in columns.items() 
                           if p['inferred_type'] in ['continuous_numeric', 'discrete_numeric']])
        categorical_count = len([c for c, p in columns.items() 
                               if p['inferred_type'] in ['categorical', 'high_cardinality_categorical']])
        binary_count = len([c for c, p in columns.items() 
                          if p['inferred_type'] == 'binary'])
        text_count = len([c for c, p in columns.items() 
                        if p['inferred_type'] == 'text'])
        datetime_count = len([c for c, p in columns.items() 
                            if p['inferred_type'] == 'datetime'])
        
        self.sample_labels['numeric_cols'].setText(str(numeric_count))
        self.sample_labels['categorical_cols'].setText(str(categorical_count))
        self.sample_labels['binary_cols'].setText(str(binary_count))
        self.sample_labels['text_cols'].setText(str(text_count))
        self.sample_labels['datetime_cols'].setText(str(datetime_count))
        
        analysis_text = "=== СТАТИСТИЧЕСКИЙ АНАЛИЗ ВЫБОРКИ ===\n\n"
        
        rows = overview.get('rows', 0)
        analysis_text += f"Размер выборки: n = {rows:,}\n"
        
        if rows < 30:
            analysis_text += "Оценка: ОЧЕНЬ МАЛАЯ (< 30) - непараметрические тесты предпочтительнее\n"
        elif rows < 100:
            analysis_text += "Оценка: Малая (30-100) - проверка предпосылок критична\n"
        elif rows < 1000:
            analysis_text += "Оценка: Средняя (100-1000) - достаточна для большинства тестов\n"
        else:
            analysis_text += "Оценка: Большая (> 1000) - параметрические тесты надежны\n"
        
        analysis_text += "\n--- ТЕСТЫ НА НОРМАЛЬНОСТЬ (Shapiro-Wilk) ---\n"
        
        normal_cols = []
        non_normal_cols = []
        
        for col_name, col_data in columns.items():
            if col_data['inferred_type'] in ['continuous_numeric', 'discrete_numeric']:
                is_normal = col_data.get('is_normal')
                p_val = col_data.get('normality_p_value')
                
                if is_normal is not None and p_val is not None:
                    if is_normal:
                        normal_cols.append((col_name, p_val))
                        analysis_text += f"  {col_name}: НОРМАЛЬНОЕ (p = {p_val:.4f})\n"
                    else:
                        non_normal_cols.append((col_name, p_val))
                        analysis_text += f"  {col_name}: НЕ НОРМАЛЬНОЕ (p = {p_val:.4f})\n"
        
        if not normal_cols and not non_normal_cols:
            analysis_text += "  Нет числовых столбцов для проверки\n"
        
        analysis_text += f"\nИтого: {len(normal_cols)} нормальных, {len(non_normal_cols)} не нормальных\n"
        
        analysis_text += "\n--- РЕКОМЕНДАЦИИ ПО ВЫБОРУ ТЕСТОВ ---\n"
        
        if non_normal_cols:
            analysis_text += f"• Для столбцов {', '.join([c[0] for c in non_normal_cols[:3]])}: "
            analysis_text += "использовать непараметрические тесты (Mann-Whitney, Kruskal-Wallis)\n"
        
        if normal_cols:
            analysis_text += f"• Для столбцов {', '.join([c[0] for c in normal_cols[:3]])}: "
            analysis_text += "можно использовать параметрические тесты (t-test, ANOVA)\n"
        
        if numeric_count > 0:
            analysis_text += "\n--- КВАНТИЛИ (первый числовой столбец) ---\n"
            first_numeric = None
            for col_name, col_data in columns.items():
                if col_data['inferred_type'] in ['continuous_numeric', 'discrete_numeric']:
                    first_numeric = (col_name, col_data)
                    break
            
            if first_numeric:
                col_name, col_data = first_numeric
                analysis_text += f"Столбец: {col_name}\n"
                if 'min' in col_data:
                    analysis_text += f"  Min: {col_data['min']:.2f}\n"
                if 'q25' in col_data:
                    analysis_text += f"  Q1 (25%): {col_data['q25']:.2f}\n"
                if 'median' in col_data:
                    analysis_text += f"  Median (50%): {col_data['median']:.2f}\n"
                if 'q75' in col_data:
                    analysis_text += f"  Q3 (75%): {col_data['q75']:.2f}\n"
                if 'max' in col_data:
                    analysis_text += f"  Max: {col_data['max']:.2f}\n"
                if 'mean' in col_data:
                    analysis_text += f"  Mean: {col_data['mean']:.2f}\n"
                if 'std' in col_data:
                    analysis_text += f"  Std: {col_data['std']:.2f}\n"
        
        corr_data = self.profile.get('correlations', {})
        if corr_data.get('available') and corr_data.get('strong_correlations'):
            analysis_text += "\n--- СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7) ---\n"
            for corr in corr_data['strong_correlations'][:5]:
                analysis_text += f"  {corr['col1']} <-> {corr['col2']}: r = {corr['correlation']:.3f}\n"
        
        self.analysis_text.setPlainText(analysis_text)
        
        columns = self.profile.get('columns', {})
        self.columns_table.setRowCount(len(columns))
        
        for i, (col_name, col_data) in enumerate(columns.items()):
            self.columns_table.setItem(i, 0, QTableWidgetItem(col_name))
            
            self.columns_table.setItem(i, 1, QTableWidgetItem(col_data.get('dtype', '-')))
            
            inferred = col_data.get('inferred_type', '-')
            self.columns_table.setItem(i, 2, QTableWidgetItem(inferred))
            
            unique = col_data.get('unique_count', 0)
            unique_pct = col_data.get('unique_percentage', 0)
            self.columns_table.setItem(i, 3, 
                QTableWidgetItem(f"{unique} ({unique_pct:.1f}%)"))
            
            missing_pct = col_data.get('missing_percentage', 0)
            self.columns_table.setItem(i, 4, QTableWidgetItem(f"{missing_pct:.1f}%"))
            
            stats_text = self._format_statistics(col_data)
            self.columns_table.setItem(i, 5, QTableWidgetItem(stats_text))
            
            issues = self._get_column_issues(col_data)
            self.columns_table.setItem(i, 6, QTableWidgetItem(issues))
        
        self.columns_table.resizeColumnsToContents()
        
        quality = self.profile.get('data_quality', {})
        score = quality.get('quality_score', 0)
        
        score_text = f"Оценка качества: {score:.1f}/100"
        if score >= 80:
            score_color = "green"
        elif score >= 60:
            score_color = "orange"
        else:
            score_color = "red"
        
        self.quality_score_label.setText(score_text)
        self.quality_score_label.setStyleSheet(f"color: {score_color};")
        
        self.issues_list.clear()
        for issue in quality.get('issues', []):
            severity_text = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}
            severity = severity_text.get(issue.get('severity', 'low'), '[?]')
            
            issue_text = f"{severity} {issue.get('type', 'unknown')}"
            if 'columns' in issue:
                cols = issue['columns']
                if isinstance(cols, list):
                    if len(cols) <= 3:
                        issue_text += f": {', '.join(str(c) if isinstance(c, str) else c.get('column', '') for c in cols)}"
                    else:
                        issue_text += f": {len(cols)} столбцов"
            elif 'count' in issue:
                issue_text += f": {issue['count']} строк ({issue.get('percentage', 0):.1f}%)"
            
            self.issues_list.addItem(issue_text)
        
        if not quality.get('issues'):
            self.issues_list.addItem("[OK] Проблем не обнаружено")
        
        self.recommendations_list.clear()
        for rec in self.profile.get('recommendations', []):
            priority_text = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}
            priority = priority_text.get(rec.get('priority', 'low'), '[?]')
            
            rec_text = f"{priority} {rec.get('description', 'No description')}"
            self.recommendations_list.addItem(rec_text)
        
        if not self.profile.get('recommendations'):
            self.recommendations_list.addItem("[OK] Дополнительных рекомендаций нет")
    
    def _format_statistics(self, col_data: Dict) -> str:
        if 'mean' in col_data:
            return f"μ={col_data['mean']:.2f}, σ={col_data.get('std', 0):.2f}"
        elif 'most_common' in col_data:
            return f"Частая: {col_data['most_common']}"
        return "-"
    
    def _get_column_issues(self, col_data: Dict) -> str:
        issues = []
        
        if col_data.get('missing_percentage', 0) > 30:
            issues.append("Много пропусков")
        
        if col_data.get('outliers_percentage', 0) > 10:
            issues.append("Много выбросов")
        
        if col_data.get('unique_count', 0) <= 1:
            issues.append("Константа")
        
        return ", ".join(issues) if issues else "-"

