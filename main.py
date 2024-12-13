import asyncio
import logging
from typing import List, Dict, Any

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from agents.embedding_agent import EmbeddingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.metadata_agent import MetadataAgent

from db_setup import initialize_database
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('graphvectorize')

load_dotenv()

class GraphVectorizeApp:
    """
    Core application class for GraphVectorize multi-agent vector database system
    """
    
    def __init__(self, 
                 embedding_model: str = 'text-embedding-3-small',
                 llm_provider: str = 'openai'):
        """
        Initialize the GraphVectorize application
        
        Args:
            embedding_model (str): Embedding model to use
            llm_provider (str): Language model provider (openai or anthropic)
        """
        self.db_connection = initialize_database()
        
        if llm_provider == 'openai':
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.2,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif llm_provider == 'anthropic':
            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.2,
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.embedding_agent = EmbeddingAgent(
            llm=self.llm, 
            database_connection=self.db_connection,
            embedding_model=embedding_model
        )
        
        self.retrieval_agent = RetrievalAgent(
            llm=self.llm, 
            database_connection=self.db_connection
        )
        
        self.metadata_agent = MetadataAgent(
            llm=self.llm, 
            database_connection=self.db_connection
        )
        
        self.embedding_workflow = self.embedding_agent.create_agent_workflow()
        self.retrieval_workflow = self.retrieval_agent.create_agent_workflow()
        self.metadata_workflow = self.metadata_agent.create_agent_workflow()
    
    def create_vector_processing_graph(self):
        """
        Create a LangGraph state graph for coordinating vector processing
        
        Returns:
            Configured StateGraph for vector operations
        """
        class VectorProcessingState(TypedDict):
            input: str
            documents: List[Dict[str, Any]]
            embeddings: List[List[float]]
            metadata: Dict[str, Any]
            error: Optional[str]
        
        graph = StateGraph(VectorProcessingState)
        
        graph.add_node("embedding", self._run_embedding_agent)
        graph.add_node("metadata", self._run_metadata_agent)
        graph.add_node("retrieval", self._run_retrieval_agent)
        
        graph.set_entry_point("embedding")
        graph.add_edge("embedding", "metadata")
        graph.add_edge("metadata", "retrieval")
        graph.add_conditional_edges(
            "retrieval", 
            self._routing_logic,
            {
                "continue": END,
                "retry": "embedding"
            }
        )
        
        return graph.compile()
    
    def _run_embedding_agent(self, state: VectorProcessingState):
        """
        Run embedding generation and storage
        
        Args:
            state (VectorProcessingState): Current processing state
        
        Returns:
            Updated state after embedding generation
        """
        try:
            embeddings = self.embedding_agent.process({
                'documents': state['input']
            })
            
            return {
                **state,
                'embeddings': embeddings['embeddings']
            }
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                **state,
                'error': str(e)
            }
    
    def _run_metadata_agent(self, state: VectorProcessingState):
        """
        Manage metadata for generated embeddings
        
        Args:
            state (VectorProcessingState): Current processing state
        
        Returns:
            Updated state with metadata
        """
        try:
            metadata = self.metadata_agent.process({
                'operation': 'add',
                'metadata': {
                    'source': state['input'],
                    'embeddings_count': len(state['embeddings'])
                }
            })
            
            return {
                **state,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Metadata processing failed: {e}")
            return {
                **state,
                'error': str(e)
            }
    
    def _run_retrieval_agent(self, state: VectorProcessingState):
        """
        Perform vector retrieval and semantic search
        
        Args:
            state (VectorProcessingState): Current processing state
        
        Returns:
            Final processing state
        """
        try:
            results = self.retrieval_agent.process({
                'embeddings': state['embeddings'],
                'metadata': state['metadata']
            })
            
            return {
                **state,
                'documents': results.get('retrieved_documents', [])
            }
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                **state,
                'error': str(e)
            }
    
    def _routing_logic(self, state: VectorProcessingState):
        """
        Determine next steps based on processing state
        
        Args:
            state (VectorProcessingState): Current processing state
        
        Returns:
            Routing decision
        """
        if state.get('error'):
            return 'retry'
        return 'continue'
    
    async def process_documents(self, documents: List[str]):
        """
        Main document processing method
        
        Args:
            documents (List[str]): Documents to process
        
        Returns:
            Processing results
        """
        workflow = self.create_vector_processing_graph()
        
        results = []
        for doc in documents:
            result = await workflow.ainvoke({
                'input': doc
            })
            results.append(result)
        
        return results
    
    async def semantic_search(self, query: str, top_k: int = 5):
        """
        Perform semantic search across vector database
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
        
        Returns:
            List of semantic search results
        """
        return await self.retrieval_agent.semantic_search(query, top_k)

def main():
    """
    Main entry point for GraphVectorize application
    """
    app = GraphVectorizeApp()
    
    async def run_example():
        documents = [
            "Machine learning is transforming industries.",
            "Vector databases enable semantic search capabilities.",
            "Multi-agent systems improve AI workflow efficiency."
        ]
        
        results = await app.process_documents(documents)
        logger.info(f"Processed {len(results)} documents")
        
        search_results = await app.semantic_search(
            "AI technological advancements", 
            top_k=2
        )
        logger.info(f"Semantic search results: {search_results}")
    
    asyncio.run(run_example())

if __name__ == "__main__":
    main()
