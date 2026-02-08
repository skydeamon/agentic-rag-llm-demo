
# RAG FastAPI + LangGraph Agent Scaffold

**Purpose.** This repository gives you a production-leaning scaffold to demonstrate two things:

1) A **RAG FastAPI service** that returns answers with **citations** from your curated corpus
2) A **basic LangGraph tool‑use flow** that chains retrieval + reasoning + formatting with simple guardrails

It is designed to be:
- **Traceable** (citations, input/output logging)
- **Testable** (eval hooks, contract tests)
- **Portable** (Dockerfile, `.env`‑driven config)

---

## Architecture (High level)

```
clients → FastAPI
          ├── /v1/rag/query  (RAG: retrieve → synthesize → cite)
          └── /v1/agent/solve (LangGraph: plan → retrieve → answer → validate)

RAG store: FAISS (local) — built from /data/sample_docs via /scripts/ingest.sh
Embeddings: sentence‑transformers (default) or provider embeddings
LLM: provider‑agnostic (OpenAI/Anthropic/HF via env)
Observability: simple request/response logs and validation errors
```

## Quickstart

```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set provider/API keys as needed

# 2) Ingest example docs into FAISS
bash scripts/ingest.sh

# 3) Run API
bash scripts/dev_run.sh
# open http://localhost:8000/docs
```

### Docker
```bash
docker build -t cerebria-llm-demo -f docker/Dockerfile .
docker run --rm -p 8000:8000 --env-file .env cerebria-llm-demo
```

## Endpoints
- `POST /v1/rag/query` → `{ query: str, top_k?: int }` → `{ answer, citations:[{id, score, source, chunk}] }`
- `POST /v1/agent/solve` → `{ question: str }` → `{ answer, steps, citations }`
- `GET  /health` → health probe

## Project Layout
```
app/
  main.py               # FastAPI bootstrap & routers
  config.py             # env & settings
  api/rag.py            # /v1/rag endpoints
  api/agent.py          # /v1/agent endpoints
  rag/ingest.py         # build FAISS index from data/
  rag/retriever.py      # retrieval wrapper
  agent/graph.py        # basic LangGraph flow
  safety/validators.py  # output schema & simple guardrails

data/sample_docs/      # seed corpus (replace with your content)
scripts/ingest.sh      # run ingestion
scripts/dev_run.sh     # launch server

tests/                 # pytest skeletons
```

## Configuration
Set **`.env`**:
- `PROVIDER=openai|anthropic|huggingface`
- `MODEL_NAME` → e.g., `gpt-4o-mini`
- `EMBEDDINGS_PROVIDER=sentence-transformers|provider`
- `EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- `VECTOR_STORE_PATH=.vector_store/faiss`

## Evaluation Hooks (lightweight)
- Deterministic output schema (Pydantic) for both endpoints
- Simple format validation + presence of citations
- Placeholders for factual checks and regression sets in `tests/`

## Safety Notes
- Mask/strip PII at ingestion if needed
- Enforce max context and deterministic prompt structure
- Add **fallbacks** when retrieval returns low confidence
- Log inputs/outputs with correlation IDs for auditability

## Extending
- Swap FAISS for pgvector/Weaviate/Pinecone
- Add rerankers (e.g., cross‑encoder) after initial retrieval
- Add offline eval suites for factuality, completeness, and toxicity
- Introduce caching (semantic + response) and cost/latency monitoring

---

**License:** MIT (for scaffold). Replace with your own for production.
