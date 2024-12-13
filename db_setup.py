import os
from typing import List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    'postgresql://graphuser:graphpassword@localhost:5432/vectordb'
)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class VectorDocument(Base):
    """
    SQLAlchemy model for storing vector embeddings with rich metadata
    
    Supports:
    - Document content
    - Vector embedding
    - Comprehensive metadata
    - Source tracking
    """
    __tablename__ = 'vector_documents'

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    
    metadata = Column(JSON, nullable=True)
    
    source_type = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    
    document_type = Column(String, nullable=True)
    tags = Column(JSON, nullable=True)
    
    created_at = Column(String, nullable=True)
    last_accessed = Column(String, nullable=True)

class DatabaseManager:
    """
    Comprehensive database management class for vector operations
    
    Provides methods for:
    - Database initialization
    - Vector document insertion
    - Semantic search
    - Metadata filtering
    """
    
    def __init__(self, database_url: str = DATABASE_URL):
        """
        Initialize database connection and session
        
        Args:
            database_url (str): PostgreSQL connection string
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def initialize_database(self):
        """
        Create all tables and extensions
        Ensures pgvector extension is available
        """
        try:
            with self.engine.connect() as conn:
                conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            Base.metadata.create_all(bind=self.engine)
            print("🚀 Database initialized successfully!")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
    
    def insert_document(
        self, 
        content: str, 
        embedding: List[float], 
        metadata: Dict[str, Any] = None,
        source_type: str = None,
        document_type: str = None,
        tags: List[str] = None
    ):
        """
        Insert a document with its vector embedding
        
        Args:
            content (str): Document text content
            embedding (List[float]): Vector embedding
            metadata (Dict, optional): Additional document metadata
            source_type (str, optional): Origin of document
            document_type (str, optional): Classification of document
            tags (List[str], optional): Searchable tags
        
        Returns:
            int: Inserted document ID
        """
        from datetime import datetime
        
        session = self.SessionLocal()
        try:
            document = VectorDocument(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                source_type=source_type,
                document_type=document_type,
                tags=tags,
                created_at=datetime.utcnow().isoformat(),
                last_accessed=datetime.utcnow().isoformat()
            )
            
            session.add(document)
            session.commit()
            session.refresh(document)
            
            return document.id
        except Exception as e:
            session.rollback()
            print(f"❌ Document insertion failed: {e}")
            return None
        finally:
            session.close()
    
    def semantic_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filters: Dict[str, Any] = None
    ):
        """
        Perform semantic search with optional metadata filtering
        
        Args:
            query_embedding (List[float]): Query vector embedding
            k (int): Number of results to return
            filters (Dict, optional): Metadata filtering conditions
        
        Returns:
            List[VectorDocument]: Matching documents
        """
        from sqlalchemy import and_
        
        session = self.SessionLocal()
        try:
            query = session.query(VectorDocument).order_by(
                VectorDocument.embedding.l2_distance(query_embedding)
            ).limit(k)
            
            if filters:
                filter_conditions = [
                    getattr(VectorDocument, key) == value 
                    for key, value in filters.items()
                ]
                query = query.filter(and_(*filter_conditions))
            
            return query.all()
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []
        finally:
            session.close()

def setup_database():
    """
    Convenience function to set up database
    Can be called during application startup
    """
    db_manager = DatabaseManager()
    db_manager.initialize_database()

if __name__ == "__main__":
    setup_database()
