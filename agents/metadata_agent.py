import logging
from typing import Any, Dict, List, Optional
import time
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from prometheus_client import Counter, Gauge

from .base_agent import BaseVectorAgent

class MetadataSchema(BaseModel):
    """
    Schema for metadata management
    """
    key: str = Field(..., description="Metadata key")
    value: Any = Field(..., description="Metadata value")
    type: str = Field(..., description="Type of metadata")

class MetadataAgent(BaseVectorAgent):
    """
    Specialized agent for managing vector database metadata
    """
    
    def __init__(
        self, 
        llm: Optional[BaseLanguageModel] = None,
        database_connection: Any = None
    ):
        """
        Initialize Metadata Agent
        
        Args:
            llm (BaseLanguageModel, optional): Language model for reasoning
            database_connection (Any): Connection to vector database
        """
        super().__init__(llm, name="MetadataAgent")
        self.database_connection = database_connection

        self.metadata_update_counter = Counter(
            'metadata_update_total', 
            'Total metadata update operations'
        )
        self.metadata_filter_latency = Gauge(
            'metadata_filter_latency_seconds', 
            'Latency of metadata filtering operations'
        )
    
    def create_tools(self) -> List[BaseTool]:
        """
        Create tools for metadata management
        
        Returns:
            List of tools for metadata operations
        """
        return [
            Tool(
                name="add_metadata",
                func=self.add_metadata,
                description="Add or update metadata for a vector embedding"
            ),
            Tool(
                name="filter_metadata",
                func=self.filter_metadata,
                description="Filter vector embeddings based on metadata"
            ),
            Tool(
                name="delete_metadata",
                func=self.delete_metadata,
                description="Delete specific metadata for a vector embedding"
            )
        ]
    
    def add_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add or update metadata for a vector embedding
        
        Args:
            metadata (Dict): Metadata to be added/updated
        
        Returns:
            Dict with operation result
        """
        start_time = time.time()
        try:
            validated_metadata = {
                k: MetadataSchema(key=k, value=v, type=type(v).__name__)
                for k, v in metadata.items()
            }
            
            result = self.database_connection.update_metadata(validated_metadata)
            
            processing_time = time.time() - start_time
            self.log_metrics("add_metadata", processing_time)
            self.metadata_update_counter.inc()
            
            return {
                "status": "success",
                "metadata": result,
                "processing_time": processing_time
            }
        
        except Exception as e:
            self.logger.error(f"Metadata addition failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def filter_metadata(
        self, 
        filters: Dict[str, Any], 
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Filter vector embeddings based on metadata
        
        Args:
            filters (Dict): Metadata filtering conditions
            max_results (int): Maximum number of results to return
        
        Returns:
            List of matching vector embeddings
        """
        start_time = time.time()
        try:
            results = self.database_connection.filter_vectors(
                filters, 
                limit=max_results
            )
            
            processing_time = time.time() - start_time
            self.metadata_filter_latency.set(processing_time)
            self.log_metrics("filter_metadata", processing_time)
            
            return {
                "status": "success",
                "results": results,
                "processing_time": processing_time,
                "result_count": len(results)
            }
        
        except Exception as e:
            self.logger.error(f"Metadata filtering failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def delete_metadata(self, vector_id: str, metadata_key: str) -> Dict[str, Any]:
        """
        Delete specific metadata for a vector embedding
        
        Args:
            vector_id (str): Unique identifier for the vector
            metadata_key (str): Metadata key to be deleted
        
        Returns:
            Dict with operation result
        """
        start_time = time.time()
        try:
            result = self.database_connection.delete_metadata(
                vector_id, 
                metadata_key
            )
            
            processing_time = time.time() - start_time
            self.log_metrics("delete_metadata", processing_time)
            self.metadata_update_counter.inc()
            
            return {
                "status": "success",
                "deleted_key": metadata_key,
                "processing_time": processing_time
            }
        
        except Exception as e:
            self.logger.error(f"Metadata deletion failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def process(self, input_data: Dict[str, Any]) -> Any:
        """
        Process metadata-related operations
        
        Args:
            input_data (Dict): Input data for metadata processing
        
        Returns:
            Processed metadata operation result
        """
        operation = input_data.get('operation')
        
        if operation == 'add':
            return self.add_metadata(input_data.get('metadata', {}))
        elif operation == 'filter':
            return self.filter_metadata(
                input_data.get('filters', {}), 
                input_data.get('max_results', 100)
            )
        elif operation == 'delete':
            return self.delete_metadata(
                input_data.get('vector_id'), 
                input_data.get('metadata_key')
            )
        else:
            self.logger.warning(f"Unsupported operation: {operation}")
            return {
                "status": "error",
                "message": "Unsupported metadata operation"
            }
