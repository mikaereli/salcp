from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QGroupBox, QLabel, QComboBox, QFormLayout,
                             QMessageBox, QScrollArea, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from typing import Optional, Dict


class VisualizationView(QWidget):
    
    def __init__(self):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None
        self.data_profile: Optional[Dict] = None
        self.current_figure: Optional[Figure] = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        left_panel = self.create_controls_panel()
        
        right_panel = self.create_chart_panel()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650])
        
        main_layout.addWidget(splitter)
        
    def create_controls_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        hist_group = QGroupBox("Гистограмма")
        hist_layout = QFormLayout()
        
        self.hist_column = QComboBox()
        hist_layout.addRow("Столбец:", self.hist_column)
        
        self.hist_bins = QComboBox()
        self.hist_bins.addItems(['10', '20', '30', '50', 'auto'])
        self.hist_bins.setCurrentText('30')
        hist_layout.addRow("Bins:", self.hist_bins)
        
        hist_btn = QPushButton("Построить гистограмму")
        hist_btn.clicked.connect(self.plot_histogram)
        hist_layout.addRow(hist_btn)
        
        hist_group.setLayout(hist_layout)
        layout.addWidget(hist_group)
        
        scatter_group = QGroupBox("Диаграмма рассеяния")
        scatter_layout = QFormLayout()
        
        self.scatter_x = QComboBox()
        scatter_layout.addRow("X:", self.scatter_x)
        
        self.scatter_y = QComboBox()
        scatter_layout.addRow("Y:", self.scatter_y)
        
        self.scatter_hue = QComboBox()
        scatter_layout.addRow("Группировка (опционально):", self.scatter_hue)
        
        scatter_btn = QPushButton("Построить scatter plot")
        scatter_btn.clicked.connect(self.plot_scatter)
        scatter_layout.addRow(scatter_btn)
        
        scatter_group.setLayout(scatter_layout)
        layout.addWidget(scatter_group)
        
        box_group = QGroupBox("Box plot")
        box_layout = QFormLayout()
        
        self.box_x = QComboBox()
        box_layout.addRow("Категории (X):", self.box_x)
        
        self.box_y = QComboBox()
        box_layout.addRow("Значения (Y):", self.box_y)
        
        box_btn = QPushButton("Построить box plot")
        box_btn.clicked.connect(self.plot_boxplot)
        box_layout.addRow(box_btn)
        
        box_group.setLayout(box_layout)
        layout.addWidget(box_group)
        
        bar_group = QGroupBox("Столбчатая диаграмма")
        bar_layout = QFormLayout()
        
        self.bar_column = QComboBox()
        bar_layout.addRow("Столбец:", self.bar_column)
        
        bar_btn = QPushButton("Построить bar chart")
        bar_btn.clicked.connect(self.plot_barchar)
        bar_layout.addRow(bar_btn)
        
        bar_group.setLayout(bar_layout)
        layout.addWidget(bar_group)
        
        heatmap_group = QGroupBox("Тепловая карта корреляций")
        heatmap_layout = QFormLayout()
        
        self.heatmap_method = QComboBox()
        self.heatmap_method.addItems(['pearson', 'spearman'])
        heatmap_layout.addRow("Метод:", self.heatmap_method)
        
        heatmap_btn = QPushButton("Построить heatmap")
        heatmap_btn.clicked.connect(self.plot_heatmap)
        heatmap_layout.addRow(heatmap_btn)
        
        heatmap_group.setLayout(heatmap_layout)
        layout.addWidget(heatmap_group)
        
        layout.addStretch()
        
        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.clear_plot)
        layout.addWidget(clear_btn)
        
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(scroll)
        
        return container
        
    def create_chart_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        title = QLabel("Визуализация данных")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.info_label = QLabel("Выберите тип графика и нажмите кнопку")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        return widget
        
    def set_data(self, data: pd.DataFrame, profile: Dict):
        self.data = data
        self.data_profile = profile
        
        numeric_cols = list(data.select_dtypes(include=['number']).columns)
        categorical_cols = [col for col, prof in profile['columns'].items()
                           if prof['inferred_type'] in ['categorical', 'binary']]
        all_cols = list(data.columns)
        
        self.hist_column.clear()
        self.hist_column.addItems(numeric_cols)
        
        self.scatter_x.clear()
        self.scatter_x.addItems(numeric_cols)
        self.scatter_y.clear()
        self.scatter_y.addItems(numeric_cols)
        self.scatter_hue.clear()
        self.scatter_hue.addItem("")
        self.scatter_hue.addItems(categorical_cols)
        
        self.box_x.clear()
        self.box_x.addItems(categorical_cols)
        self.box_y.clear()
        self.box_y.addItems(numeric_cols)
        
        self.bar_column.clear()
        self.bar_column.addItems(categorical_cols)
        
    def plot_histogram(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        column = self.hist_column.currentText()
        bins = self.hist_bins.currentText()
        
        if not column:
            QMessageBox.warning(self, "Ошибка", "Выберите столбец")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            bins_value = 'auto' if bins == 'auto' else int(bins)
            
            ax.hist(self.data[column].dropna(), bins=bins_value, edgecolor='black', alpha=0.7)
            ax.set_xlabel(column)
            ax.set_ylabel('Частота')
            ax.set_title(f'Гистограмма: {column}')
            ax.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.info_label.setText(f"Гистограмма для {column}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика:\n{str(e)}")
    
    def plot_scatter(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        x_col = self.scatter_x.currentText()
        y_col = self.scatter_y.currentText()
        hue_col = self.scatter_hue.currentText()
        
        if not x_col or not y_col:
            QMessageBox.warning(self, "Ошибка", "Выберите столбцы X и Y")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            if hue_col:
                groups = self.data[hue_col].unique()
                for group in groups:
                    mask = self.data[hue_col] == group
                    ax.scatter(self.data.loc[mask, x_col], 
                              self.data.loc[mask, y_col],
                              label=str(group), alpha=0.6)
                ax.legend()
            else:
                ax.scatter(self.data[x_col], self.data[y_col], alpha=0.6)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            ax.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.info_label.setText(f"Scatter plot: {x_col} vs {y_col}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика:\n{str(e)}")
    
    def plot_boxplot(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        x_col = self.box_x.currentText()
        y_col = self.box_y.currentText()
        
        if not x_col or not y_col:
            QMessageBox.warning(self, "Ошибка", "Выберите столбцы")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            groups = self.data[x_col].unique()
            data_to_plot = [self.data[self.data[x_col] == g][y_col].dropna() for g in groups]
            
            bp = ax.boxplot(data_to_plot, labels=[str(g) for g in groups], patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Box Plot: {y_col} by {x_col}')
            ax.grid(True, alpha=0.3, axis='y')
            
            if len(groups) > 5:
                ax.tick_params(axis='x', rotation=45)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.info_label.setText(f"Box plot: {y_col} by {x_col}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика:\n{str(e)}")
    
    def plot_barchar(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        column = self.bar_column.currentText()
        
        if not column:
            QMessageBox.warning(self, "Ошибка", "Выберите столбец")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            value_counts = self.data[column].value_counts().head(20)
            
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel(column)
            ax.set_ylabel('Количество')
            ax.set_title(f'Bar Chart: {column}')
            ax.grid(True, alpha=0.3, axis='y')
            
            ax.tick_params(axis='x', rotation=45)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.info_label.setText(f"Bar chart для {column} (топ {len(value_counts)} категорий)")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика:\n{str(e)}")
    
    def plot_heatmap(self):
        if self.data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите данные")
            return
        
        method = self.heatmap_method.currentText()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "Недостаточно данных", 
                              "Нужно минимум 2 числовых столбца")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            corr_matrix = self.data[numeric_cols].corr(method=method)
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            self.figure.colorbar(im, ax=ax)
            
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'Correlation Heatmap ({method})')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            self.info_label.setText(f"Correlation heatmap ({method})")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика:\n{str(e)}")
    
    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()
        self.info_label.setText("График очищен")

