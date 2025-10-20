from typing import Any, Dict, List, Optional
from enum import Enum
import pandas as pd


class BlockCategory(Enum):
    DATA = "data"
    CLEANING = "cleaning"
    ANALYSIS = "analysis"
    TESTING = "testing"
    VISUALIZATION = "visualization"
    EXPORT = "export"


class BlockType(Enum):
    LOAD_CSV = "load_csv"
    FILTER_ROWS = "filter_rows"
    SELECT_COLUMNS = "select_columns"
    
    HANDLE_MISSING = "handle_missing"
    REMOVE_DUPLICATES = "remove_duplicates"
    REMOVE_OUTLIERS = "remove_outliers"
    
    DESCRIPTIVE_STATS = "descriptive_stats"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    
    T_TEST = "t_test"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    AB_TEST = "ab_test"
    
    PLOT_HISTOGRAM = "plot_histogram"
    PLOT_SCATTER = "plot_scatter"
    PLOT_BOX = "plot_box"
    
    EXPORT_CSV = "export_csv"
    EXPORT_REPORT = "export_report"


class BlockParameter:
    def __init__(self, name: str, param_type: str, required: bool = True,
                 default: Any = None, options: Optional[List] = None,
                 description: str = ""):
        self.name = name
        self.param_type = param_type  # 'string', 'int', 'float', 'bool', 'column', 'columns'
        self.required = required
        self.default = default
        self.options = options
        self.description = description


class Block:
    
    def __init__(self, block_id: str, block_type: BlockType):
        self.block_id = block_id
        self.block_type = block_type
        self.parameters: Dict[str, Any] = {}
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        
    def set_parameter(self, name: str, value: Any):
        self.parameters[name] = value
        
    def set_input(self, name: str, value: Any):
        self.inputs[name] = value
        
    def execute(self) -> Dict[str, Any]:
        raise NotImplementedError
        
    def validate(self) -> bool:
        return True


class LoadCSVBlock(Block):
    
    PARAMETERS = [
        BlockParameter("file_path", "file", required=True, 
                      description="Путь к CSV файлу")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.LOAD_CSV)
        
    def execute(self) -> Dict[str, Any]:
        from core.data_loader import DataLoader
        
        file_path = self.parameters.get('file_path')
        loader = DataLoader()
        data = loader.load_csv(file_path)
        
        return {
            'data': data,
            'metadata': loader.get_metadata()
        }


class FilterRowsBlock(Block):
    
    PARAMETERS = [
        BlockParameter("column", "column", required=True,
                      description="Столбец для фильтрации"),
        BlockParameter("operator", "string", required=True,
                      options=['==', '!=', '>', '<', '>=', '<=', 'contains'],
                      description="Оператор сравнения"),
        BlockParameter("value", "string", required=True,
                      description="Значение для сравнения")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.FILTER_ROWS)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data')
        column = self.parameters.get('column')
        operator = self.parameters.get('operator')
        value = self.parameters.get('value')
        
        if operator == '==':
            filtered = data[data[column] == value]
        elif operator == '!=':
            filtered = data[data[column] != value]
        elif operator == '>':
            filtered = data[data[column] > float(value)]
        elif operator == '<':
            filtered = data[data[column] < float(value)]
        elif operator == '>=':
            filtered = data[data[column] >= float(value)]
        elif operator == '<=':
            filtered = data[data[column] <= float(value)]
        elif operator == 'contains':
            filtered = data[data[column].astype(str).str.contains(str(value))]
        else:
            filtered = data
        
        return {'data': filtered}


class SelectColumnsBlock(Block):
    
    PARAMETERS = [
        BlockParameter("columns", "columns", required=True,
                      description="Столбцы для выбора")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.SELECT_COLUMNS)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data')
        columns = self.parameters.get('columns')
        
        selected = data[columns]
        
        return {'data': selected}


class HandleMissingBlock(Block):
    
    PARAMETERS = [
        BlockParameter("method", "string", required=True,
                      options=['drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'],
                      description="Метод обработки пропусков"),
        BlockParameter("columns", "columns", required=False,
                      description="Столбцы для обработки (пусто = все)")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.HANDLE_MISSING)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data').copy()
        method = self.parameters.get('method')
        columns = self.parameters.get('columns')
        
        if not columns:
            columns = data.columns
        
        if method == 'drop':
            data = data.dropna(subset=columns)
        elif method == 'mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col].fillna(data[col].mean(), inplace=True)
        elif method == 'median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col].fillna(data[col].median(), inplace=True)
        elif method == 'mode':
            for col in columns:
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)
        elif method == 'forward_fill':
            data[columns] = data[columns].fillna(method='ffill')
        elif method == 'backward_fill':
            data[columns] = data[columns].fillna(method='bfill')
        
        return {'data': data}


class RemoveDuplicatesBlock(Block):

    PARAMETERS = [
        BlockParameter("columns", "columns", required=False,
                      description="Столбцы для проверки дубликатов (пусто = все)")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.REMOVE_DUPLICATES)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data')
        columns = self.parameters.get('columns')
        
        if columns:
            cleaned = data.drop_duplicates(subset=columns)
        else:
            cleaned = data.drop_duplicates()
        
        return {'data': cleaned}


class TTestBlock(Block):
    
    PARAMETERS = [
        BlockParameter("group_column", "column", required=True,
                      description="Столбец с группами"),
        BlockParameter("value_column", "column", required=True,
                      description="Столбец со значениями"),
        BlockParameter("group1", "string", required=True,
                      description="Первая группа"),
        BlockParameter("group2", "string", required=True,
                      description="Вторая группа")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.T_TEST)
        
    def execute(self) -> Dict[str, Any]:
        from core.statistical_tests import StatisticalTests
        
        data = self.inputs.get('data')
        tests = StatisticalTests(data)
        
        result = tests.t_test(
            self.parameters['group_column'],
            self.parameters['value_column'],
            self.parameters['group1'],
            self.parameters['group2']
        )
        
        return {'result': result}


class ABTestBlock(Block):
    
    PARAMETERS = [
        BlockParameter("group_column", "column", required=True,
                      description="Столбец с группами"),
        BlockParameter("metric_column", "column", required=True,
                      description="Столбец с метрикой"),
        BlockParameter("metric_type", "string", required=True,
                      options=['continuous', 'binary'],
                      default='continuous',
                      description="Тип метрики")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.AB_TEST)
        
    def execute(self) -> Dict[str, Any]:
        from core.ab_testing import ABTestEngine
        
        data = self.inputs.get('data')
        engine = ABTestEngine(data)
        
        result = engine.run_ab_test(
            self.parameters['group_column'],
            self.parameters['metric_column'],
            metric_type=self.parameters.get('metric_type', 'continuous')
        )
        
        return {'result': result}


class DescriptiveStatsBlock(Block):
    
    PARAMETERS = [
        BlockParameter("columns", "columns", required=False,
                      description="Столбцы для анализа (пусто = все числовые)")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.DESCRIPTIVE_STATS)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data')
        columns = self.parameters.get('columns')
        
        if not columns:
            columns = data.select_dtypes(include=['number']).columns
        
        stats = data[columns].describe()
        
        return {'result': stats}


class CorrelationBlock(Block):
    
    PARAMETERS = [
        BlockParameter("method", "string", required=True,
                      options=['pearson', 'spearman', 'kendall'],
                      default='pearson',
                      description="Метод корреляции"),
        BlockParameter("columns", "columns", required=False,
                      description="Столбцы для анализа (пусто = все числовые)")
    ]
    
    def __init__(self, block_id: str):
        super().__init__(block_id, BlockType.CORRELATION)
        
    def execute(self) -> Dict[str, Any]:
        data = self.inputs.get('data')
        method = self.parameters.get('method', 'pearson')
        columns = self.parameters.get('columns')
        
        if not columns:
            columns = data.select_dtypes(include=['number']).columns
        
        corr_matrix = data[columns].corr(method=method)
        
        return {'result': corr_matrix}


BLOCK_REGISTRY = {
    BlockType.LOAD_CSV: LoadCSVBlock,
    BlockType.FILTER_ROWS: FilterRowsBlock,
    BlockType.SELECT_COLUMNS: SelectColumnsBlock,
    BlockType.HANDLE_MISSING: HandleMissingBlock,
    BlockType.REMOVE_DUPLICATES: RemoveDuplicatesBlock,
    BlockType.T_TEST: TTestBlock,
    BlockType.AB_TEST: ABTestBlock,
    BlockType.DESCRIPTIVE_STATS: DescriptiveStatsBlock,
    BlockType.CORRELATION: CorrelationBlock,
}


def create_block(block_type: BlockType, block_id: str) -> Block:
    block_class = BLOCK_REGISTRY.get(block_type)
    if block_class:
        return block_class(block_id)
    raise ValueError(f"Unknown block type: {block_type}")

