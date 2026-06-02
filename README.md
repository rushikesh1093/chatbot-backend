# Agrofarm RAG Backend (FastAPI)

This backend adds Retrieval-Augmented Generation (RAG) for the Agrofarm chatbot.

## Architecture

React frontend -> FastAPI backend -> RAG pipeline

RAG pipeline:
1. Embed product documents using `sentence-transformers`
2. Store vectors in FAISS index
3. Retrieve top matching chunks for each question
4. Send context + question to OpenRouter LLM
5. Return grounded answer + sources

## Folder Layout

- `main.py`: FastAPI app and endpoints
- `rag.py`: document ingestion, chunking, embedding, retrieval
- `data/`: knowledge base files (`.txt`, `.csv`, `.pdf`)
- `vector_store/`: generated FAISS index + chunk metadata
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
copy .env.example .env
```

Then set:
- `OPENROUTER_API_KEY`
- optional: `OPENROUTER_MODEL`

4. Start server:

```bash
uvicorn app:app --reload --port 9000
```

## Ingest Knowledge Base

After placing docs in `data/`, build the vector index:

```bash
curl -X POST http://localhost:9000/ingest
```

or call `/ingest` from Postman.

## API Endpoints

- `GET /health`
- `POST /ingest`
- `POST /chat`

### `POST /chat` body

```json
{
  "question": "Which product helps root rot in nursery?",
  "language": "en",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"}
  ]
}
```
