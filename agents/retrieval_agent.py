import time
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from db_setup import DatabaseManager

from .base_agent import BaseVectorAgent

class RetrievalAgent(BaseVectorAgent):
    """
    Specialized agent for semantic search and document retrieval
    """
    
    def __init__(
        self, 
        llm=None, 
        embedding_model=None,
        db_manager=None,
        k: int = 5
    ):
        """
        Initialize Retrieval Agent
        
        Args:
            llm (BaseLanguageModel, optional): Reasoning language model
            embedding_model (Optional): Embedding generation model
            db_manager (DatabaseManager, optional): Database management instance
            k (int, optional): Number of results to retrieve
        """
        super().__init__(llm, name="RetrievalAgent")
        
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.db_manager = db_manager or DatabaseManager()
        self.k = k
    
    def create_tools(self):
        """
        Create tools for semantic search and retrieval
        
        Returns:
            List of tools for the agent
        """
        return [
            self.semantic_search,
            self.filtered_search,
            self.context_aware_retrieval
        ]
    
    @tool
    def semantic_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on vector database
        
        Args:
            query (str): Search query
            k (int, optional): Number of results to retrieve
        
        Returns:
            List of matching documents
        """
        start_time = time.time()
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self.db_manager.semantic_search(
                query_embedding, 
                k=k or self.k
            )
            
            processed_results = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "source_type": doc.source_type
                } for doc in results
            ]
            
            processing_time = time.time() - start_time
            self.log_metrics('semantic_search', processing_time)
            
            return processed_results
        
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise
    
    @tool
    def filtered_search(
        self, 
        query: str, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with additional metadata filtering
        
        Args:
            query (str): Search query
            filters (Dict): Metadata filtering conditions
        
        Returns:
            List of matching documents
        """
        start_time = time.time()
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self.db_manager.semantic_search(
                query_embedding, 
                filters=filters
            )
            
            processed_results = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "source_type": doc.source_type
                } for doc in results
            ]
            
            processing_time = time.time() - start_time
            self.log_metrics('filtered_search', processing_time)
            
            return processed_results
        
        except Exception as e:
            self.logger.error(f"Filtered search failed: {e}")
            raise
    
    @tool
    def context_aware_retrieval(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform context-aware semantic search
        
        Args:
            query (str): Search query
            context (Dict, optional): Additional context for search
        
        Returns:
            List of context-aware matching documents
        """
        start_time = time.time()
        
        try:
            enhanced_query = (
                f"{query} " + 
                " ".join([f"{k}: {v}" for k, v in (context or {}).items()])
            )
            
            query_embedding = self.embedding_model.embed_query(enhanced_query)
            
            results = self.db_manager.semantic_search(query_embedding)
            
            processed_results = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": self._calculate_relevance(doc, context)
                } for doc in results
            ]
            
            processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            processing_time = time.time() - start_time
            self.log_metrics('context_aware_retrieval', processing_time)
            
            return processed_results
        
        except Exception as e:
            self.logger.error(f"Context-aware retrieval failed: {e}")
            raise
    
    def _calculate_relevance(
        self, 
        document, 
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate document relevance based on context
        
        Args:
            document: Document to evaluate
            context (Dict, optional): Contextual information
        
        Returns:
            Relevance score
        """
        if not context:
            return 1.0
        
        relevance = 1.0
        for key, value in context.items():
            doc_value = document.metadata.get(key)
            if doc_value == value:
                relevance *= 1.2
            elif doc_value is not None:
                relevance *= 0.8
        
        return min(relevance, 1.0)
    
    async def process(self, input_data):
        """
        Process input data for retrieval
        
        Args:
            input_data (Dict): Search parameters
        
        Returns:
            Search results
        """
        query = input_data.get('query')
        filters = input_data.get('filters')
        context = input_data.get('context')
        
        if filters:
            return self.filtered_search(query, filters)
        elif context:
            return self.context_aware_retrieval(query, context)
        else:
            return self.semantic_search(query)
