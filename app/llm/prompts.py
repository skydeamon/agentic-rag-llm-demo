
from langchain_core.prompts import PromptTemplate

# RAG Synthesis Prompt
RAG_TEMPLATE = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context below.
If the answer is not in the context, say "I don't have enough information to answer that."

CRITICAL: You MUST include inline citations for every claim you make using the format [Source: filename].

Context:
{context}

Question:
{question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_TEMPLATE
)

# Agent Planning Prompt (Simple ReAct style or similar)
AGENT_PLAN_TEMPLATE = """You are an intelligent agent.
Your goal is to answer the user's question: {question}

You have access to the following steps:
1. Plan: Decide what information you need.
2. Retrieve: Search for documents.
3. Answer: Synthesize the final answer.
4. Validate: Check if the answer is complete.

Current Context:
{context}

What is your next step directly? Output JSON format: {{"step": "...", "reason": "..."}}
"""

AGENT_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=AGENT_PLAN_TEMPLATE
)
