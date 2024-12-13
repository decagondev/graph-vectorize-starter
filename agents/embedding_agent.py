import time
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from db_setup import DatabaseManager

from .base_agent import BaseVectorAgent

class EmbeddingAgent(BaseVectorAgent):
    """
    Specialized agent for generating and managing vector embeddings
    """
    
    def __init__(
        self, 
        llm=None, 
        embedding_model=None,
        db_manager=None
    ):
        """
        Initialize Embedding Agent
        
        Args:
            llm (BaseLanguageModel, optional): Reasoning language model
            embedding_model (Optional): Embedding generation model
            db_manager (DatabaseManager, optional): Database management instance
        """
        super().__init__(llm, name="EmbeddingAgent")
        
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.db_manager = db_manager or DatabaseManager()
    
    def create_tools(self):
        """
        Create tools for embedding generation and management
        
        Returns:
            List of tools for the agent
        """
        return [
            self.generate_embedding,
            self.bulk_embed_documents,
            self.update_document_embedding
        ]
    
    @tool
    def generate_embedding(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate embedding for a single text
        
        Args:
            text (str): Text to embed
            metadata (Dict, optional): Additional document metadata
        
        Returns:
            Dict containing embedding and metadata
        """
        start_time = time.time()
        
        try:
            embedding = self.embedding_model.embed_query(text)
            
            doc_id = self.db_manager.insert_document(
                content=text,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            processing_time = time.time() - start_time
            self.log_metrics('generate_embedding', processing_time)
            
            return {
                "embedding": embedding,
                "document_id": doc_id,
                "metadata": metadata
            }
        
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    @tool
    def bulk_embed_documents(
        self, 
        documents: List[str], 
        metadata_list: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple documents
        
        Args:
            documents (List[str]): List of documents to embed
            metadata_list (List[Dict], optional): Metadata for each document
        
        Returns:
            List of embedding results
        """
        start_time = time.time()
        
        metadata_list = metadata_list or [{} for _ in documents]
        
        results = []
        for doc, metadata in zip(documents, metadata_list):
            result = self.generate_embedding(doc, metadata)
            results.append(result)
        
        processing_time = time.time() - start_time
        self.log_metrics('bulk_embed_documents', processing_time)
        
        return results
    
    @tool
    def update_document_embedding(
        self, 
        document_id: int, 
        new_text: str
    ) -> Dict[str, Any]:
        """
        Update embedding for an existing document
        
        Args:
            document_id (int): ID of the document to update
            new_text (str): New text content
        
        Returns:
            Updated document information
        """
        start_time = time.time()
        
        try:
            new_embedding = self.embedding_model.embed_query(new_text)
          
            
            processing_time = time.time() - start_time
            self.log_metrics('update_document_embedding', processing_time)
            
            return {
                "document_id": document_id,
                "new_embedding": new_embedding,
                "status": "updated"
            }
        
        except Exception as e:
            self.logger.error(f"Embedding update failed: {e}")
            raise
    
    async def process(self, input_data):
        """
        Process input data for embedding
        
        Args:
            input_data (Union[str, List[str]]): Input text(s) to process
        
        Returns:
            Embedding results
        """
        if isinstance(input_data, str):
            return self.generate_embedding(input_data)
        elif isinstance(input_data, list):
            return self.bulk_embed_documents(input_data)
        else:
            raise ValueError("Invalid input type for embedding")
