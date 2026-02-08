
from fastapi import FastAPI
from .api.rag import router as rag_router
from .api.agent import router as agent_router

app = FastAPI(title="Agentic LLM Demo", version="0.1.0")

@app.get('/health')
def health():
    return {"status": "ok"}

app.include_router(rag_router, prefix='/v1/rag')
app.include_router(agent_router, prefix='/v1/agent')
