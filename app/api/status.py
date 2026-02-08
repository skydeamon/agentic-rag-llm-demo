
from fastapi import APIRouter
from ..rag.retriever import RagService
from ..config import settings
import os

router = APIRouter()

@router.get('/status')
async def get_rag_status():
    """Returns the status of the RAG system and vector store."""
    faiss_path = settings.vector_store_path
    index_exists = os.path.exists(faiss_path)
    meta_exists = os.path.exists(faiss_path + ".meta.pkl")
    
    return {
        "vector_store": {
            "path": faiss_path,
            "index_loaded": index_exists,
            "metadata_loaded": meta_exists,
        },
        "config": {
            "provider": settings.provider,
            "model_name": settings.model_name,
            "embeddings_model": settings.embeddings_model
        }
    }
