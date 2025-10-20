import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import chardet


class DataLoader:
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            self.data = pd.read_csv(file_path, encoding=encoding, **kwargs)
            
            self.metadata = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'encoding': encoding,
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'size_mb': Path(file_path).stat().st_size / (1024 * 1024)
            }
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def get_data(self) -> Optional[pd.DataFrame]:
        return self.data
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
    
    def clear_data(self):
        self.data = None
        self.metadata = {}

