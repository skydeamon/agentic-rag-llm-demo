
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..agent.graph import AgentFlow
from ..safety.validators import AgentAnswer

router = APIRouter()
flow = AgentFlow()

class AgentRequest(BaseModel):
    question: str

@router.post('/solve', response_model=AgentAnswer)
async def solve(req: AgentRequest):
    try:
        return flow.run(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
