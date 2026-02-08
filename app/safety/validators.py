
from pydantic import BaseModel, Field
from typing import List, Optional, Any

class Citation(BaseModel):
    id: int
    score: float
    source: str
    chunk: str

class AnswerPayload(BaseModel):
    answer: str = Field(min_length=1)
    citations: List[Citation]

class AgentStep(BaseModel):
    step: int
    action: str
    details: str

class AgentAnswer(BaseModel):
    answer: str
    steps: List[AgentStep]
    citations: List[Citation]
