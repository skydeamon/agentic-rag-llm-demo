
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..rag.retriever import RagService
from ..safety.validators import AnswerPayload

router = APIRouter()
rag = RagService()

class RagRequest(BaseModel):
    query: str
    top_k: int = 4

@router.post('/query', response_model=AnswerPayload)
async def query_rag(req: RagRequest):
    try:
        return rag.answer(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
