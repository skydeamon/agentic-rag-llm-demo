from fastapi import APIRouter, HTTPException
from typing import List, Optional
from ..rag.retriever import RagService
from ..models import AnswerPayload, RagRequest

router = APIRouter()
rag = RagService()

@router.post('/query', response_model=AnswerPayload)
async def query_rag(req: RagRequest):
    try:
        return rag.answer(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
