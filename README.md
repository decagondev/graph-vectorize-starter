# 🌐 GraphVectorize: Multi-Agent Vector Database System

## 📝 Overview

GraphVectorize is an innovative project demonstrating the power of LangGraph for creating a sophisticated, multi-agent vector database system. This solution leverages advanced AI agents to manage, process, and retrieve vector embeddings with intelligent coordination.

## 🌟 Key Features

- 🤖 **Multi-Agent Architecture**: Intelligent agents for different vector database operations
- 🔍 **Semantic Search**: Advanced contextual retrieval and similarity matching
- 🧠 **Adaptive Reasoning**: Agents collaborate to optimize vector storage and querying
- 🐘 **PostgreSQL Integration**: Robust vector storage using `pgvector`
- 🐳 **Dockerized Environment**: Simplified deployment and setup
- 📊 **Flexible Metadata Handling**: Intelligent metadata management and filtering
- 🔭 **Comprehensive Monitoring**: Integrated Prometheus and Grafana dashboards

## 🏗️ Project Structure

```
📁 graphvectorize/
│
├── 🐳 Dockerfile              # Container configuration
├── 📝 docker-compose.yml       # Multi-container Docker setup
├── 📋 pyproject.toml           # Poetry dependency management
├── 📊 prometheus.yml           # Prometheus monitoring configuration
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
DATABASE_URL=postgresql://graphuser:graphpassword@postgres:5432/vectordb
REDIS_URL=redis://redis:6379/0
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 4. Build and Launch 🚢

```bash
# Start all services
docker-compose up --build

# Or run specific services
docker-compose up app postgres redis
```

## 🔬 Infrastructure Components

### Docker Composition

Our `docker-compose.yml` provides a robust multi-service architecture:

- **Main Application**: Python-based LangGraph service
- **PostgreSQL**: Vector database with `pgvector` extension
- **Redis**: Caching and task queuing
- **Adminer**: Database management interface
- **Prometheus**: System and application monitoring
- **Grafana**: Visualization and dashboarding

### Dependency Management

We use Poetry for sophisticated dependency management:
- `pyproject.toml` defines project dependencies
- Supports development and production environments
- Ensures consistent package versions across setups

### Monitoring Stack

- **Prometheus**: Collects metrics from:
  - Application performance
  - Database statistics
  - Redis cache metrics
- **Grafana**: Creates interactive dashboards for real-time insights

## 💡 Usage Examples

(Previous usage examples remain the same)

## 🚀 Monitoring and Observability

### Accessing Dashboards

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000`
- **Adminer**: `http://localhost:8080`

### Metrics Tracking

Configure custom metrics in your agents:

```python
from prometheus_client import Counter, Gauge

# Example agent metrics
EMBEDDING_REQUESTS = Counter(
    'agent_embedding_requests_total', 
    'Total embedding generation requests'
)
VECTOR_SEARCH_LATENCY = Gauge(
    'agent_vector_search_latency_seconds', 
    'Latency of semantic search operations'
)
```

## 🤝 Contributing Workflow

(Previous contribution guidelines remain the same)

## 🛡️ Troubleshooting

- **Docker Composition**
  - Verify all service dependencies
  - Check container logs with `docker-compose logs <service>`

- **Performance Monitoring**
  - Review Prometheus metrics
  - Analyze Grafana dashboards
  - Adjust resource allocations in `docker-compose.yml`

## 🚀 Future Roadmap

- [ ] Multi-model support
- [ ] Advanced agent reasoning capabilities
- [ ] Distributed agent architectures
- [ ] Enhanced observability and tracing
- [ ] Custom Grafana dashboards

## 📜 License

MIT License - See `LICENSE` file for details.

## 💌 Disclaimer

This project demonstrates multi-agent vector database interactions. Adapt carefully for production use and comply with AI service providers' terms.

## 🌐 Connect

- **Project Link**: [GitHub Repository](https://github.com/decagondev/graph-vectorize-starter)
- **Issues**: [Project Issues](https://github.com/decagondev/graph-vectorize-starter/issues)

---

**Empower Your AI Workflows!** 🚀🤖
