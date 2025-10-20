from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QLabel, QPushButton, QLineEdit,
                             QComboBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt
import pandas as pd
from typing import Optional, Dict


class DataView(QWidget):
    def __init__(self):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Optional[Dict] = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.metadata_group = QGroupBox("Информация о данных")
        metadata_layout = QGridLayout()
        self.metadata_group.setLayout(metadata_layout)
        
        self.file_label = QLabel("Файл: -")
        self.rows_label = QLabel("Строк: -")
        self.cols_label = QLabel("Столбцов: -")
        self.size_label = QLabel("Размер: -")
        
        metadata_layout.addWidget(self.file_label, 0, 0)
        metadata_layout.addWidget(self.rows_label, 0, 1)
        metadata_layout.addWidget(self.cols_label, 0, 2)
        metadata_layout.addWidget(self.size_label, 0, 3)
        
        layout.addWidget(self.metadata_group)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Фильтр:"))
        
        self.filter_column = QComboBox()
        filter_layout.addWidget(self.filter_column)
        
        self.filter_operator = QComboBox()
        self.filter_operator.addItems(['содержит', '=', '≠', '>', '<', '≥', '≤'])
        filter_layout.addWidget(self.filter_operator)
        
        self.filter_value = QLineEdit()
        filter_layout.addWidget(self.filter_value)
        
        self.filter_btn = QPushButton("Применить")
        self.filter_btn.clicked.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_btn)
        
        self.reset_btn = QPushButton("Сбросить")
        self.reset_btn.clicked.connect(self.reset_filter)
        filter_layout.addWidget(self.reset_btn)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        rows_layout = QHBoxLayout()
        rows_layout.addWidget(QLabel("Показать строк:"))
        
        self.rows_limit = QComboBox()
        self.rows_limit.addItems(['100', '500', '1000', '5000', 'Все'])
        self.rows_limit.setCurrentText('1000')
        self.rows_limit.currentTextChanged.connect(self.on_rows_limit_changed)
        rows_layout.addWidget(self.rows_limit)
        
        rows_layout.addStretch()
        layout.addLayout(rows_layout)
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        self.info_label = QLabel("Загрузите данные для начала работы")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
    def set_data(self, data: pd.DataFrame, metadata: Dict):
        self.data = data.copy()
        self.metadata = metadata
        
        self.file_label.setText(f"Файл: {metadata.get('file_name', '-')}")
        self.rows_label.setText(f"Строк: {metadata.get('rows', 0):,}")
        self.cols_label.setText(f"Столбцов: {metadata.get('columns', 0)}")
        self.size_label.setText(f"Размер: {metadata.get('size_mb', 0):.2f} МБ")
        
        self.filter_column.clear()
        self.filter_column.addItems(self.data.columns.tolist())
        
        self.display_data(self.data)
        
    def on_rows_limit_changed(self):
        if self.data is not None:
            self.display_data(self.data)
    
    def display_data(self, data: pd.DataFrame):
        limit_text = self.rows_limit.currentText()
        if limit_text == 'Все':
            display_rows = len(data)
        else:
            display_rows = min(len(data), int(limit_text))
        
        self.table.setRowCount(display_rows)
        self.table.setColumnCount(len(data.columns))
        self.table.setHorizontalHeaderLabels(data.columns.tolist())
        
        for i in range(display_rows):
            for j, col in enumerate(data.columns):
                value = data.iloc[i, j]
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, j, item)
        
        self.table.resizeColumnsToContents()
        
        if len(data) > display_rows:
            self.info_label.setText(
                f"Показано {display_rows:,} из {len(data):,} строк"
            )
        else:
            self.info_label.setText(f"Всего строк: {len(data):,}")
    
    def apply_filter(self):
        if self.data is None:
            return
        
        column = self.filter_column.currentText()
        operator = self.filter_operator.currentText()
        value = self.filter_value.text()
        
        if not value:
            return
        
        try:
            filtered = self.data.copy()
            
            if operator == 'содержит':
                filtered = filtered[filtered[column].astype(str).str.contains(value, case=False)]
            elif operator == '=':
                filtered = filtered[filtered[column] == value]
            elif operator == '≠':
                filtered = filtered[filtered[column] != value]
            elif operator == '>':
                filtered = filtered[filtered[column] > float(value)]
            elif operator == '<':
                filtered = filtered[filtered[column] < float(value)]
            elif operator == '≥':
                filtered = filtered[filtered[column] >= float(value)]
            elif operator == '≤':
                filtered = filtered[filtered[column] <= float(value)]
            
            self.display_data(filtered)
            
        except Exception as e:
            self.info_label.setText(f"Ошибка фильтрации: {str(e)}")
    
    def reset_filter(self):
        if self.data is not None:
            self.filter_value.clear()
            self.display_data(self.data)

