from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QGroupBox, QLabel, QComboBox, QFormLayout,
                             QMessageBox, QScrollArea, QSplitter, QListWidget,
                             QTableWidget, QTableWidgetItem, QCheckBox, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from scipy import stats


class DataProcessingView(QWidget):
    
    data_changed = None
    
    def __init__(self):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.data_profile: Optional[Dict] = None
        self.parent_window = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        left_panel = self.create_controls_panel()
        
        right_panel = self.create_preview_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
    def create_controls_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("Выберите операции обработки данных:")
        info_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(info_label)
        
        missing_group = QGroupBox("1. Обработка пропущенных значений")
        missing_layout = QFormLayout()
        
        self.missing_column = QComboBox()
        missing_layout.addRow("Столбец:", self.missing_column)
        
        self.missing_method = QComboBox()
        self.missing_method.addItems([
            'Удалить строки с пропусками',
            'Заполнить средним (mean)',
            'Заполнить медианой (median)',
            'Заполнить модой (mode)',
            'Заполнить нулями',
            'Заполнить значением...',
            'Forward fill (предыдущее значение)',
            'Backward fill (следующее значение)',
            'KNN импутация (умная)'
        ])
        missing_layout.addRow("Метод:", self.missing_method)
        
        missing_btn_layout = QHBoxLayout()
        
        self.preview_missing_btn = QPushButton("Просмотр")
        self.preview_missing_btn.clicked.connect(self.preview_missing_handling)
        missing_btn_layout.addWidget(self.preview_missing_btn)
        
        self.apply_missing_btn = QPushButton("Применить")
        self.apply_missing_btn.clicked.connect(self.apply_missing_handling)
        missing_btn_layout.addWidget(self.apply_missing_btn)
        
        missing_layout.addRow(missing_btn_layout)
        
        missing_group.setLayout(missing_layout)
        layout.addWidget(missing_group)
        
        duplicates_group = QGroupBox("2. Обработка дубликатов")
        duplicates_layout = QVBoxLayout()
        
        self.duplicates_info = QLabel("Дубликаты: -")
        duplicates_layout.addWidget(self.duplicates_info)
        
        dup_btn_layout = QHBoxLayout()
        
        self.show_duplicates_btn = QPushButton("Показать дубликаты")
        self.show_duplicates_btn.clicked.connect(self.show_duplicates)
        dup_btn_layout.addWidget(self.show_duplicates_btn)
        
        self.remove_duplicates_btn = QPushButton("Удалить дубликаты")
        self.remove_duplicates_btn.clicked.connect(self.remove_duplicates)
        dup_btn_layout.addWidget(self.remove_duplicates_btn)
        
        duplicates_layout.addLayout(dup_btn_layout)
        
        duplicates_group.setLayout(duplicates_layout)
        layout.addWidget(duplicates_group)
        
        outliers_group = QGroupBox("3. Обработка выбросов")
        outliers_layout = QFormLayout()
        
        self.outliers_column = QComboBox()
        outliers_layout.addRow("Столбец:", self.outliers_column)
        
        self.outliers_method = QComboBox()
        self.outliers_method.addItems([
            'IQR метод (1.5 × IQR)',
            'IQR метод (3.0 × IQR)',
            'Z-score (|z| > 3)',
            'Персентили (1% и 99%)',
            'Персентили (5% и 95%)'
        ])
        outliers_layout.addRow("Метод определения:", self.outliers_method)
        
        self.outliers_action = QComboBox()
        self.outliers_action.addItems([
            'Показать статистику',
            'Удалить выбросы',
            'Заменить на границы (cap)',
            'Заменить на медиану',
            'Заменить на среднее'
        ])
        outliers_layout.addRow("Действие:", self.outliers_action)
        
        outliers_btn_layout = QHBoxLayout()
        
        self.detect_outliers_btn = QPushButton("Обнаружить")
        self.detect_outliers_btn.clicked.connect(self.detect_outliers)
        outliers_btn_layout.addWidget(self.detect_outliers_btn)
        
        self.apply_outliers_btn = QPushButton("Применить")
        self.apply_outliers_btn.clicked.connect(self.apply_outliers_handling)
        outliers_btn_layout.addWidget(self.apply_outliers_btn)
        
        outliers_layout.addRow(outliers_btn_layout)
        
        outliers_group.setLayout(outliers_layout)
        layout.addWidget(outliers_group)
        
        convert_group = QGroupBox("4. Преобразование типов данных")
        convert_layout = QFormLayout()
        
        self.convert_column = QComboBox()
        convert_layout.addRow("Столбец:", self.convert_column)
        
        self.convert_type = QComboBox()
        self.convert_type.addItems([
            'int (целое число)',
            'float (дробное число)',
            'string (текст)',
            'datetime (дата/время)',
            'category (категория)'
        ])
        convert_layout.addRow("Преобразовать в:", self.convert_type)
        
        self.convert_btn = QPushButton("Применить преобразование")
        self.convert_btn.clicked.connect(self.convert_dtype)
        convert_layout.addRow(self.convert_btn)
        
        convert_group.setLayout(convert_layout)
        layout.addWidget(convert_group)
        
        layout.addStretch()
        
        reset_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Сбросить все изменения")
        self.reset_btn.clicked.connect(self.reset_changes)
        self.reset_btn.setStyleSheet("background-color: #ffcccc;")
        reset_layout.addWidget(self.reset_btn)
        layout.addLayout(reset_layout)
        
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(scroll)
        
        return container
        
    def create_preview_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("Просмотр и результаты")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        self.stats_label = QLabel("Статистика изменений:")
        layout.addWidget(self.stats_label)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlainText("Выберите операцию и нажмите 'Просмотр' или 'Обнаружить'")
        layout.addWidget(self.preview_text)
        
        return widget
        
    def set_data(self, data: pd.DataFrame, profile: Dict):
        self.data = data.copy()
        self.original_data = data.copy()
        self.data_profile = profile
        
        all_cols = list(data.columns)
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        
        self.missing_column.clear()
        self.missing_column.addItem("(Все столбцы)")
        self.missing_column.addItems(all_cols)
        
        self.outliers_column.clear()
        self.outliers_column.addItems(numeric_cols)
        
        self.convert_column.clear()
        self.convert_column.addItems(all_cols)
        
        dup_count = data.duplicated().sum()
        dup_pct = (dup_count / len(data)) * 100 if len(data) > 0 else 0
        self.duplicates_info.setText(f"Дубликаты: {dup_count} строк ({dup_pct:.1f}%)")
        
        self.update_stats()
        
    def get_processed_data(self) -> pd.DataFrame:
        return self.data
        
    def preview_missing_handling(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        column = self.missing_column.currentText()
        method = self.missing_method.currentText()
        
        preview_text = "=== ПРОСМОТР ОБРАБОТКИ ПРОПУСКОВ ===\n\n"
        
        if column == "(Все столбцы)":
            cols_to_process = self.data.columns
        else:
            cols_to_process = [column]
        
        for col in cols_to_process:
            if col not in self.data.columns:
                continue
                
            missing_count = self.data[col].isna().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            
            preview_text += f"Столбец: {col}\n"
            preview_text += f"  Пропусков: {missing_count} ({missing_pct:.1f}%)\n"
            
            if missing_count == 0:
                preview_text += f"  Действие: нет пропусков\n\n"
                continue
            
            preview_text += f"  Метод: {method}\n"
            
            if 'Удалить строки' in method:
                preview_text += f"  Результат: будет удалено {missing_count} строк\n"
            elif 'средним' in method and pd.api.types.is_numeric_dtype(self.data[col]):
                mean_val = self.data[col].mean()
                preview_text += f"  Значение для заполнения: {mean_val:.2f}\n"
            elif 'медианой' in method and pd.api.types.is_numeric_dtype(self.data[col]):
                median_val = self.data[col].median()
                preview_text += f"  Значение для заполнения: {median_val:.2f}\n"
            elif 'модой' in method:
                mode_val = self.data[col].mode()
                if len(mode_val) > 0:
                    preview_text += f"  Значение для заполнения: {mode_val[0]}\n"
            
            preview_text += "\n"
        
        self.preview_text.setPlainText(preview_text)
        
    def apply_missing_handling(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        column = self.missing_column.currentText()
        method = self.missing_method.currentText()
        
        try:
            if column == "(Все столбцы)":
                cols_to_process = self.data.columns
            else:
                cols_to_process = [column]
            
            for col in cols_to_process:
                if col not in self.data.columns:
                    continue
                
                if 'Удалить строки' in method:
                    self.data = self.data.dropna(subset=[col])
                elif 'средним' in method:
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif 'медианой' in method:
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        self.data[col].fillna(self.data[col].median(), inplace=True)
                elif 'модой' in method:
                    mode_val = self.data[col].mode()
                    if len(mode_val) > 0:
                        self.data[col].fillna(mode_val[0], inplace=True)
                elif 'нулями' in method:
                    self.data[col].fillna(0, inplace=True)
                elif 'Forward fill' in method:
                    self.data[col].fillna(method='ffill', inplace=True)
                elif 'Backward fill' in method:
                    self.data[col].fillna(method='bfill', inplace=True)
                elif 'KNN' in method:
                    # Simple KNN imputation for numeric columns
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = self.data.select_dtypes(include=['number']).columns
                        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
            
            self.update_stats()
            self.notify_data_change()
            QMessageBox.information(self, "Успешно", "Пропуски обработаны")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки пропусков:\n{str(e)}")
    
    def show_duplicates(self):
        if self.data is None:
            return
        
        duplicates = self.data[self.data.duplicated(keep=False)]
        
        preview_text = "=== ДУБЛИКАТЫ ===\n\n"
        preview_text += f"Найдено дубликатов: {len(duplicates)} строк\n\n"
        
        if len(duplicates) > 0:
            preview_text += "Первые 10 дубликатов:\n"
            preview_text += duplicates.head(10).to_string()
        else:
            preview_text += "Дубликатов не найдено!"
        
        self.preview_text.setPlainText(preview_text)
    
    def remove_duplicates(self):
        if self.data is None:
            return
        
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        
        removed = before - after
        
        self.update_stats()
        self.notify_data_change()
        QMessageBox.information(self, "Успешно", f"Удалено {removed} дубликатов")
    
    def detect_outliers(self):
        if self.data is None:
            return
        
        column = self.outliers_column.currentText()
        method = self.outliers_method.currentText()
        
        if not column or column not in self.data.columns:
            return
        
        col_data = self.data[column].dropna()
        
        preview_text = f"=== ОБНАРУЖЕНИЕ ВЫБРОСОВ ===\n\n"
        preview_text += f"Столбец: {column}\n"
        preview_text += f"Метод: {method}\n\n"
        
        if 'IQR' in method:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            if '1.5' in method:
                multiplier = 1.5
            else:
                multiplier = 3.0
            
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outliers = col_data[outliers_mask]
            
            preview_text += f"Q1 (25%): {q1:.2f}\n"
            preview_text += f"Q3 (75%): {q3:.2f}\n"
            preview_text += f"IQR: {iqr:.2f}\n"
            preview_text += f"Нижняя граница: {lower_bound:.2f}\n"
            preview_text += f"Верхняя граница: {upper_bound:.2f}\n\n"
            
        elif 'Z-score' in method:
            z_scores = np.abs(stats.zscore(col_data))
            outliers_mask = z_scores > 3
            outliers = col_data[outliers_mask]
            
            preview_text += f"Порог |z-score|: 3\n\n"
            
        elif 'Персентили' in method:
            if '1%' in method:
                lower_pct, upper_pct = 1, 99
            else:
                lower_pct, upper_pct = 5, 95
            
            lower_bound = col_data.quantile(lower_pct / 100)
            upper_bound = col_data.quantile(upper_pct / 100)
            
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outliers = col_data[outliers_mask]
            
            preview_text += f"Нижняя граница ({lower_pct}%): {lower_bound:.2f}\n"
            preview_text += f"Верхняя граница ({upper_pct}%): {upper_bound:.2f}\n\n"
        
        outliers_count = len(outliers)
        outliers_pct = (outliers_count / len(col_data)) * 100
        
        preview_text += f"Обнаружено выбросов: {outliers_count} ({outliers_pct:.1f}%)\n\n"
        
        if outliers_count > 0:
            preview_text += "Статистика выбросов:\n"
            preview_text += f"  Min выброса: {outliers.min():.2f}\n"
            preview_text += f"  Max выброса: {outliers.max():.2f}\n"
            preview_text += f"  Mean выброса: {outliers.mean():.2f}\n\n"
            
            preview_text += "Первые 20 выбросов:\n"
            preview_text += str(outliers.head(20).tolist())
        
        self.preview_text.setPlainText(preview_text)
    
    def apply_outliers_handling(self):
        if self.data is None:
            return
        
        column = self.outliers_column.currentText()
        method = self.outliers_method.currentText()
        action = self.outliers_action.currentText()
        
        if 'Показать статистику' in action:
            self.detect_outliers()
            return
        
        try:
            col_data = self.data[column]

            if 'IQR' in method:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                multiplier = 1.5 if '1.5' in method else 3.0
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
                
            elif 'Z-score' in method:
                z_scores = np.abs(stats.zscore(col_data.dropna()))
                outliers_mask = pd.Series([False] * len(col_data), index=col_data.index)
                outliers_mask[col_data.notna()] = z_scores > 3
                lower_bound, upper_bound = None, None
                
            elif 'Персентили' in method:
                if '1%' in method:
                    lower_pct, upper_pct = 1, 99
                else:
                    lower_pct, upper_pct = 5, 95
                lower_bound = col_data.quantile(lower_pct / 100)
                upper_bound = col_data.quantile(upper_pct / 100)
                outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
            outliers_count = outliers_mask.sum()
            
            if 'Удалить' in action:
                self.data = self.data[~outliers_mask]
            elif 'границы' in action and lower_bound is not None:
                self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
            elif 'медиану' in action:
                median_val = col_data.median()
                self.data.loc[outliers_mask, column] = median_val
            elif 'среднее' in action:
                mean_val = col_data.mean()
                self.data.loc[outliers_mask, column] = mean_val
            
            self.update_stats()
            self.notify_data_change()
            QMessageBox.information(self, "Успешно", f"Обработано {outliers_count} выбросов")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки выбросов:\n{str(e)}")
    
    def convert_dtype(self):
        if self.data is None:
            return
        
        column = self.convert_column.currentText()
        target_type = self.convert_type.currentText()
        
        try:
            if 'int' in target_type:
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce').astype('Int64')
            elif 'float' in target_type:
                self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
            elif 'string' in target_type:
                self.data[column] = self.data[column].astype(str)
            elif 'datetime' in target_type:
                self.data[column] = pd.to_datetime(self.data[column], errors='coerce')
            elif 'category' in target_type:
                self.data[column] = self.data[column].astype('category')
            
            self.update_stats()
            self.notify_data_change()
            QMessageBox.information(self, "Успешно", f"Тип столбца {column} изменен на {target_type}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка преобразования типа:\n{str(e)}")
    
    def reset_changes(self):
        if self.original_data is None:
            return
        
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Сбросить все изменения и вернуться к исходным данным?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.data = self.original_data.copy()
            self.update_stats()
            self.notify_data_change()
            QMessageBox.information(self, "Сброшено", "Данные возвращены к исходному состоянию")
    
    def update_stats(self):
        if self.data is None or self.original_data is None:
            return
        
        stats_text = f"Исходные данные: {len(self.original_data):,} строк\n"
        stats_text += f"Текущие данные: {len(self.data):,} строк\n"
        stats_text += f"Удалено строк: {len(self.original_data) - len(self.data):,}\n\n"
        
        total_missing_original = self.original_data.isna().sum().sum()
        total_missing_current = self.data.isna().sum().sum()
        stats_text += f"Пропусков было: {total_missing_original:,}\n"
        stats_text += f"Пропусков стало: {total_missing_current:,}\n"
        stats_text += f"Заполнено: {total_missing_original - total_missing_current:,}"
        
        self.stats_label.setText(stats_text)
    
    def notify_data_change(self):
        if self.parent_window:
            self.parent_window.on_data_processed(self.data)

