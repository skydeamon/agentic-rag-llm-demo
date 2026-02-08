import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .api.rag import router as rag_router
from .api.agent import router as agent_router
from .api.status import router as status_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic LLM Demo", version="0.1.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"path={request.url.path} method={request.method} duration={duration:.4f}s status={response.status_code}")
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)}
    )

@app.get('/health')
def health():
    return {"status": "ok"}

app.include_router(rag_router, prefix='/v1/rag')
app.include_router(agent_router, prefix='/v1/agent')
app.include_router(status_router, prefix='/v1/rag')
