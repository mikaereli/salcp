from typing import Dict, List, Any, Optional
from workflow.blocks import Block, BlockType, create_block
import uuid


class Connection:
    def __init__(self, from_block: str, from_output: str, 
                 to_block: str, to_input: str):
        self.from_block = from_block
        self.from_output = from_output
        self.to_block = to_block
        self.to_input = to_input


class Workflow:
    
    def __init__(self, name: str = "Untitled Workflow"):
        self.name = name
        self.blocks: Dict[str, Block] = {}
        self.connections: List[Connection] = []
        
    def add_block(self, block_type: BlockType, block_id: Optional[str] = None) -> str:
        if block_id is None:
            block_id = f"{block_type.value}_{uuid.uuid4().hex[:8]}"
        
        block = create_block(block_type, block_id)
        self.blocks[block_id] = block
        
        return block_id
    
    def remove_block(self, block_id: str):
        if block_id in self.blocks:
            self.connections = [
                conn for conn in self.connections
                if conn.from_block != block_id and conn.to_block != block_id
            ]
            del self.blocks[block_id]
    
    def connect(self, from_block: str, from_output: str,
                to_block: str, to_input: str):
        if from_block not in self.blocks or to_block not in self.blocks:
            raise ValueError("Block not found in workflow")
        
        connection = Connection(from_block, from_output, to_block, to_input)
        self.connections.append(connection)
    
    def disconnect(self, from_block: str, to_block: str):
        self.connections = [
            conn for conn in self.connections
            if not (conn.from_block == from_block and conn.to_block == to_block)
        ]
    
    def get_block(self, block_id: str) -> Optional[Block]:
        return self.blocks.get(block_id)
    
    def set_block_parameter(self, block_id: str, param_name: str, value: Any):
        block = self.blocks.get(block_id)
        if block:
            block.set_parameter(param_name, value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'blocks': [
                {
                    'id': block_id,
                    'type': block.block_type.value,
                    'parameters': block.parameters
                }
                for block_id, block in self.blocks.items()
            ],
            'connections': [
                {
                    'from_block': conn.from_block,
                    'from_output': conn.from_output,
                    'to_block': conn.to_block,
                    'to_input': conn.to_input
                }
                for conn in self.connections
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Workflow':
        workflow = cls(data.get('name', 'Untitled Workflow'))
        
        for block_data in data.get('blocks', []):
            block_id = workflow.add_block(
                BlockType(block_data['type']),
                block_data['id']
            )
            for param_name, param_value in block_data.get('parameters', {}).items():
                workflow.set_block_parameter(block_id, param_name, param_value)
        
        for conn_data in data.get('connections', []):
            workflow.connect(
                conn_data['from_block'],
                conn_data['from_output'],
                conn_data['to_block'],
                conn_data['to_input']
            )
        
        return workflow


class WorkflowEngine:
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.execution_log: List[Dict[str, Any]] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def execute(self) -> Dict[str, Any]:
        self.execution_log = []
        self.results = {}
        
        execution_order = self._topological_sort()
        
        if not execution_order:
            return {
                'status': 'error',
                'message': 'Workflow has cycles or is invalid'
            }
        
        try:
            for block_id in execution_order:
                self._execute_block(block_id)
            
            return {
                'status': 'success',
                'results': self.results,
                'execution_log': self.execution_log
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'execution_log': self.execution_log
            }
    
    def _execute_block(self, block_id: str):
        block = self.workflow.blocks[block_id]
        for conn in self.workflow.connections:
            if conn.to_block == block_id:
                from_result = self.results.get(conn.from_block, {})
                input_value = from_result.get(conn.from_output)
                block.set_input(conn.to_input, input_value)
        
        self.execution_log.append({
            'block_id': block_id,
            'block_type': block.block_type.value,
            'status': 'executing'
        })
        
        try:
            result = block.execute()
            self.results[block_id] = result
            
            self.execution_log[-1]['status'] = 'success'
            self.execution_log[-1]['result_keys'] = list(result.keys())
        except Exception as e:
            self.execution_log[-1]['status'] = 'error'
            self.execution_log[-1]['error'] = str(e)
            raise
    
    def _topological_sort(self) -> List[str]:
        graph = {block_id: [] for block_id in self.workflow.blocks}
        in_degree = {block_id: 0 for block_id in self.workflow.blocks}
        
        for conn in self.workflow.connections:
            graph[conn.from_block].append(conn.to_block)
            in_degree[conn.to_block] += 1
        
        queue = [block_id for block_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            block_id = queue.pop(0)
            result.append(block_id)
            
            for neighbor in graph[block_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.workflow.blocks):
            return []
        
        return result
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        return self.execution_log
    
    def get_block_result(self, block_id: str) -> Optional[Dict[str, Any]]:
        return self.results.get(block_id)

