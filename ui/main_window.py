from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QFileDialog, QMessageBox,
                             QToolBar, QStatusBar, QSplitter, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
import pandas as pd
from typing import Optional

from ui.data_view import DataView
from ui.data_processing_view import DataProcessingView
from ui.analysis_view import AnalysisView
from ui.profiler_view import ProfilerView
from ui.visualization_view import VisualizationView
from core.data_loader import DataLoader
from core.data_profiler import DataProfiler
from core.recommender import MethodRecommender


class MainWindow(QMainWindow):
    
    data_loaded = pyqtSignal(pd.DataFrame, dict)
    
    def __init__(self):
        super().__init__()
        
        self.data: Optional[pd.DataFrame] = None
        self.data_profile: Optional[dict] = None
        self.data_loader = DataLoader()
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Statistical Analysis Low-Code Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        
        self.create_toolbar()
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.data_view = DataView()
        self.data_processing_view = DataProcessingView()
        self.profiler_view = ProfilerView()
        self.analysis_view = AnalysisView()
        self.visualization_view = VisualizationView()
        
        self.tab_widget.addTab(self.data_view, "Данные")
        self.tab_widget.addTab(self.profiler_view, "Профилирование")
        self.tab_widget.addTab(self.data_processing_view, "Обработка данных")
        self.tab_widget.addTab(self.analysis_view, "Статистический анализ")
        self.tab_widget.addTab(self.visualization_view, "Визуализация")
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")
        
        self.data_loaded.connect(self.on_data_loaded)
        
        self.data_processing_view.parent_window = self
        
    def create_toolbar(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        load_action = QAction("Загрузить CSV", self)
        load_action.triggered.connect(self.load_csv)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        clean_action = QAction("Очистить данные", self)
        clean_action.triggered.connect(self.clean_data)
        toolbar.addAction(clean_action)
        
        toolbar.addSeparator()
        
        help_action = QAction("Справка", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV файл",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.status_bar.showMessage("Загрузка данных...")
                
                self.data = self.data_loader.load_csv(file_path)
                metadata = self.data_loader.get_metadata()
                
                self.data_view.set_data(self.data, metadata)
                
                self.data_loaded.emit(self.data, metadata)
                
                self.status_bar.showMessage(
                    f"Загружено: {metadata['rows']} строк, {metadata['columns']} столбцов"
                )
                
                self.tab_widget.setCurrentIndex(0)
                
                self.profile_data()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Ошибка загрузки",
                    f"Не удалось загрузить файл:\n{str(e)}"
                )
                self.status_bar.showMessage("Ошибка загрузки")
    
    def profile_data(self):
        if self.data is None:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала загрузите данные"
            )
            return
        
        try:
            self.status_bar.showMessage("Профилирование данных...")
            
            profiler = DataProfiler(self.data)
            self.data_profile = profiler.profile_data()
            
            self.profiler_view.set_profile(self.data_profile)
            
            self.data_processing_view.set_data(self.data, self.data_profile)
            
            self.analysis_view.set_data(self.data, self.data_profile)
            
            self.visualization_view.set_data(self.data, self.data_profile)
            
            self.status_bar.showMessage("Профилирование завершено")
            
            self.tab_widget.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка профилирования",
                f"Не удалось профилировать данные:\n{str(e)}"
            )
            self.status_bar.showMessage("Ошибка профилирования")
    
    def clean_data(self):
        if self.data is None or self.data_profile is None:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Сначала загрузите и профилируйте данные"
            )
            return
        
        try:
            from core.data_cleaner import DataCleaner
            
            reply = QMessageBox.question(
                self,
                "Режим очистки",
                "Использовать агрессивный режим очистки?\n\n"
                "Да - более тщательная очистка (может изменить больше данных)\n"
                "Нет - консервативная очистка",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            aggressive = (reply == QMessageBox.StandardButton.Yes)
            
            self.status_bar.showMessage("Очистка данных...")
            
            cleaner = DataCleaner(self.data, self.data_profile)
            cleaned_data = cleaner.auto_clean(aggressive=aggressive)
            cleaning_log = cleaner.get_cleaning_log()
            
            if hasattr(self.data_processing_view, 'data') and self.data_processing_view.data is not None:
                self.data = self.data_processing_view.get_processed_data()
            else:
                self.data = cleaned_data
            
            self.data_view.set_data(self.data, self.data_loader.get_metadata())
            
            log_text = "Выполненные операции:\n\n"
            for entry in cleaning_log:
                log_text += f"• {entry['description']}\n"
            
            QMessageBox.information(
                self,
                "Очистка завершена",
                log_text
            )
            
            self.status_bar.showMessage("Очистка завершена")
            
            self.profile_data()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка очистки",
                f"Не удалось очистить данные:\n{str(e)}"
            )
            self.status_bar.showMessage("Ошибка очистки")
    
    def on_data_loaded(self, data: pd.DataFrame, metadata: dict):
        pass
    
    def on_data_processed(self, processed_data: pd.DataFrame):
        self.data = processed_data
        
        from core.data_profiler import DataProfiler
        profiler = DataProfiler(self.data)
        self.data_profile = profiler.profile_data()
        
        metadata = {
            'file_name': self.data_loader.metadata.get('file_name', 'processed_data'),
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'size_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        self.data_view.set_data(self.data, metadata)
        self.profiler_view.set_profile(self.data_profile)
        self.analysis_view.set_data(self.data, self.data_profile)
        self.visualization_view.set_data(self.data, self.data_profile)
        
        self.status_bar.showMessage("Данные обновлены после обработки")
    
    def show_help(self):
        help_text = """
        <h2>Statistical Analysis Low-Code Platform</h2>
        <h3>Быстрый старт:</h3>
        <ol>
            <li>Загрузите CSV файл через кнопку "Загрузить CSV"</li>
            <li>Данные автоматически профилируются</li>
            <li>Используйте вкладку "Обработка данных" для детальной очистки</li>
            <li>Перейдите во вкладку "Статистический анализ"</li>
            <li>Выберите нужный анализ и нажмите соответствующую кнопку</li>
            <li>Используйте вкладку "Визуализация" для построения графиков</li>
        </ol>
        
        <h3>Возможности:</h3>
        <ul>
            <li>Автоматическое профилирование данных</li>
            <li>Умная очистка данных</li>
            <li>A/B/C/n тестирование</li>
            <li>Статистические тесты (t-test, ANOVA, Chi-square)</li>
            <li>Корреляционный анализ</li>
            <li>Интерактивные графики и диаграммы</li>
            <li>Рекомендации статистических методов</li>
        </ul>
        
        <h3>Вкладки приложения:</h3>
        <ul>
            <li><b>Данные</b> - просмотр и фильтрация таблицы</li>
            <li><b>Обработка данных</b> - детальная работа с пропусками, дубликатами и выбросами</li>
            <li><b>Профилирование</b> - автоматический анализ данных и рекомендации</li>
            <li><b>Статистический анализ</b> - тесты и сравнения (A/B, t-test, ANOVA, и др.)</li>
            <li><b>Визуализация</b> - графики и диаграммы</li>
        </ul>
        
        <h3>Обработка данных:</h3>
        <ul>
            <li>Пропуски: 9 методов заполнения (mean, median, mode, KNN и др.)</li>
            <li>Дубликаты: просмотр и удаление</li>
            <li>Выбросы: 5 методов обнаружения, 5 способов обработки</li>
            <li>Типы данных: преобразование между типами</li>
        </ul>
        """
        
        QMessageBox.about(self, "Справка", help_text)

