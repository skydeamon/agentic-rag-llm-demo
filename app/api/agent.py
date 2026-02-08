from fastapi import APIRouter, HTTPException
from ..agent.graph import AgentFlow
from ..models import AgentAnswer, AgentRequest

router = APIRouter()
flow = AgentFlow()

@router.post('/solve', response_model=AgentAnswer)
async def solve(req: AgentRequest):
    try:
        return flow.run(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
