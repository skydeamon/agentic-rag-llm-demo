
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# --- RAG Models ---

class Citation(BaseModel):
    id: int
    score: float
    source: str
    chunk: str

class AnswerPayload(BaseModel):
    answer: str = Field(min_length=1, examples=["Coronary physiology is the study of blood flow..."])
    citations: List[Citation] = Field(default_factory=list)

class RagRequest(BaseModel):
    query: str = Field(..., examples=["What is coronary physiology?"])
    top_k: int = Field(default=4, ge=1, le=20, examples=[5])

class RagResponse(AnswerPayload):
    pass

# --- Agent Models ---

class AgentRequest(BaseModel):
    question: str = Field(..., examples=["Help me with my research on heart blood flow."])

class AgentStep(BaseModel):
    step: int
    action: str
    details: str

class AgentAnswer(BaseModel):
    answer: str
    steps: List[AgentStep]
    citations: List[Citation] = Field(default_factory=list)
