# 🌐 GraphVectorize: Multi-Agent Vector Database System

## 📝 Overview

GraphVectorize is a project demonstrating the power of LangGraph for creating a sophisticated, multi-agent vector database system. This solution leverages advanced AI agents to manage, process, and retrieve vector embeddings with intelligent coordination.

## 🌟 Key Features

- 🤖 **Multi-Agent Architecture**: Intelligent agents for different vector database operations
- 🔍 **Semantic Search**: Advanced contextual retrieval and similarity matching
- 🧠 **Adaptive Reasoning**: Agents collaborate to optimize vector storage and querying
- 🐘 **PostgreSQL Integration**: Robust vector storage using `pgvector`
- 🐳 **Dockerized Environment**: Simplified deployment and setup
- 📊 **Flexible Metadata Handling**: Intelligent metadata management and filtering

## 🏗️ Project Structure

```
📁 graphvectorize/
│
├── 🐳 Dockerfile              # Container configuration
├── 📝 docker-compose.yml       # Multi-container Docker setup
├── 📋 requirements.txt         # Python dependencies
├── 🐍 main.py                  # Core application logic
├── 🤖 agents/                  # Agent implementation directory
│   ├── retrieval_agent.py      # Document retrieval specialist
│   ├── embedding_agent.py      # Embedding generation agent
│   └── metadata_agent.py       # Metadata management agent
├── 🗄️ db_setup.py              # Database initialization script
└── 📖 README.md                # Project documentation
```

## 🛠️ Prerequisites

- 💻 Docker and Docker Compose
- 🐍 Python 3.11+
- 🔑 OpenAI API Key
- 🌐 LangGraph
- 🐙 GitHub Account (for contributing)

## 🚀 Quick Start Guide

### 1. Fork the Repository 🍴

1. Visit the repository: https://github.com/YourOrg/graphvectorize
2. Click the **Fork** button in the top-right corner
3. Choose your personal GitHub account

### 2. Clone Your Forked Repository 📦

```bash
# Replace {your-username} with your GitHub username
git clone https://github.com/{your-username}/graphvectorize.git
cd graphvectorize

# Add upstream remote
git remote add upstream https://github.com/YourOrg/graphvectorize.git
```

### 3. Set Up Environment Variables 🔧

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://yourusername:yourpassword@postgres:5432/vectordb
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 4. Build and Launch 🚢

```bash
docker-compose up --build
```

## 💡 Usage Examples

### Creating a Multi-Agent Vector Workflow

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from graphvectorize.agents import (
    EmbeddingAgent, 
    RetrievalAgent, 
    MetadataAgent
)

# Define agent tools
@tool
def generate_embeddings(documents):
    """Generate vector embeddings for documents"""
    return embedding_agent.process(documents)

@tool
def semantic_search(query, k=3):
    """Perform semantic search on vector database"""
    return retrieval_agent.search(query, k)

@tool
def manage_metadata(document, metadata_actions):
    """Manage document metadata"""
    return metadata_agent.process(document, metadata_actions)

# Initialize agents
embedding_agent = EmbeddingAgent()
retrieval_agent = RetrievalAgent()
metadata_agent = MetadataAgent()

# Create collaborative workflow
workflow = create_react_agent(
    tools=[
        generate_embeddings, 
        semantic_search, 
        manage_metadata
    ],
    llm=ChatOpenAI(model="gpt-4")
)
```

### Advanced Agent Coordination

```python
# Example of a complex document processing workflow
def process_research_documents(documents):
    # Embedding generation
    embeddings = workflow.invoke({
        "input": "Generate embeddings for these research documents",
        "documents": documents
    })
    
    # Metadata enrichment
    enriched_docs = workflow.invoke({
        "input": "Enrich metadata for these documents",
        "documents": embeddings
    })
    
    # Semantic indexing
    workflow.invoke({
        "input": "Index these documents in our vector database",
        "documents": enriched_docs
    })
```

## 🔬 Advanced Configurations

### Custom Agent Capabilities

- **Embedding Agent**: Support for multiple embedding models
- **Retrieval Agent**: Advanced query expansion and re-ranking
- **Metadata Agent**: Intelligent metadata extraction and management

### Extensible Architecture

```python
class CustomAgent:
    def __init__(self, strategies=None):
        """Easily extend agent capabilities"""
        self.strategies = strategies or []
    
    def add_strategy(self, strategy):
        """Dynamically add processing strategies"""
        self.strategies.append(strategy)
```

## 🤝 Contributing Workflow

(Similar to the previous README's contribution guidelines)

## 🛡️ Troubleshooting

- **Agent Coordination Issues**
  - Verify LangGraph configuration
  - Check inter-agent communication protocols

- **Vector Database Challenges**
  - Ensure PostgreSQL vector extension is properly configured
  - Validate embedding dimensions

## 🚀 Future Roadmap

- [ ] Multi-model support
- [ ] Advanced agent reasoning capabilities
- [ ] Distributed agent architectures
- [ ] Enhanced observability and tracing

## 📜 License

MIT License - See `LICENSE` file for details.

## 💌 Disclaimer

This project demonstrates multi-agent vector database interactions. Adapt carefully for production use and comply with AI service providers' terms.

## 🌐 Connect

- **Project Link**: [GitHub Repository](https://github.com/decagondev/graph-vectorize-starter)
- **Issues**: [Project Issues](https://github.com/decagondev/graph-vectorize-starter/issues)

---

**Empower Your AI Workflows!** 🚀🤖
