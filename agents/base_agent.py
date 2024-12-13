import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from prometheus_client import Counter, Gauge

class BaseVectorAgent(ABC):
    """
    Abstract base class for vector database agents
    Provides common functionality and monitoring capabilities
    """
    
    def __init__(
        self, 
        llm: Optional[BaseLanguageModel] = None, 
        name: str = "BaseAgent"
    ):
        """
        Initialize the base agent
        
        Args:
            llm (BaseLanguageModel, optional): Language model for reasoning
            name (str): Unique identifier for the agent
        """
        self.name = name
        self.llm = llm
        self.logger = logging.getLogger(f"agent.{name}")
        
        # Prometheus metrics
        self.request_counter = Counter(
            f'{name.lower()}_requests_total', 
            f'Total requests processed by {name} agent'
        )
        self.processing_time = Gauge(
            f'{name.lower()}_processing_time_seconds', 
            f'Processing time for {name} agent operations'
        )
    
    def log_metrics(self, method_name: str, processing_time: float):
        """
        Log prometheus metrics for agent operations
        
        Args:
            method_name (str): Name of the method being tracked
            processing_time (float): Time taken to process
        """
        self.request_counter.inc()
        self.processing_time.set(processing_time)
        self.logger.info(f"{self.name} - {method_name} completed in {processing_time:.2f}s")
    
    @abstractmethod
    def create_tools(self) -> List[BaseTool]:
        """
        Create tools specific to this agent
        
        Returns:
            List of tools the agent can use
        """
        pass
    
    def create_agent_workflow(self):
        """
        Create a LangGraph agent workflow
        
        Returns:
            Configured agent workflow
        """
        tools = self.create_tools()
        return create_react_agent(
            tools=tools,
            llm=self.llm
        )
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Abstract method for processing input data
        
        Args:
            input_data (Any): Input data to be processed
        
        Returns:
            Processed output
        """
        pass
