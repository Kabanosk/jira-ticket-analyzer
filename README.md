# Jira Ticket Analyzer

Automated Jira ticket analysis system built with LangGraph, FastAPI and PostgreSQL. Applies local LLMs to classify, analyze and recommend actions for incoming engineering tickets.

## Architecture

```
POST /tickets
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   classify  │────▶│   analyze   │────▶│   recommend     │
│             │     │             │     │                 │
│ • labels    │     │ • priority  │     │ • recommended   │
│ • work_type │     │ • similar   │     │   action        │
│             │     │   tickets   │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
      │                    │                    │
      └────────────────────┴────────────────────┘
                           │
       PostgreSQL checkpoint (state persistence)
```

## Stack

- **LangGraph** - stateful agent workflow with PostgreSQL checkpointing
- **FastAPI + Pydantic** - REST API with typed request/response models
- **pgvector** - vector similarity search for finding related past tickets
- **Ollama** - local LLM inference (gemma3:1b) and embeddings (nomic-embed-text)
- **Docker Compose** - PostgreSQL + pgvector container

## Quickstart

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- Docker + Docker Compose
- Python 3.14+ with [uv](https://github.com/astral-sh/uv)

### Pull models

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### Setup

```bash
# Clone and install dependencies
git clone https://github.com/Kabanosk/jira-ticket-analyzer
cd jira-ticket-analyzer
uv sync

# Configure environment variables
cp .env.example .env

# Start PostgreSQL
docker compose up -d

# Create mock historical tickets for vector search
uv run create_tickets.py

# Start the API
uv run uvicorn app.main:app --reload
```

## API

### Analyze a ticket

```bash
curl -X POST http://localhost:8000/tickets -H "Content-Type: application/json" -d '{
    "ticket_id": "A0809-623",
    "title": "[RAG] Hybrid search",
    "description": "Find a better option to search info in docs when user prompt direct words"
  }'
```

### Get ticket analysis state

```bash
curl http://localhost:8000/tickets/A0809-623
```

### Retry failed workflow

```bash
curl -X POST http://localhost:8000/tickets/A0809-623/retry
```
