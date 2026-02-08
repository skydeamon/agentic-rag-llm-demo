
from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_rag_status():
    response = client.get("/v1/rag/status")
    assert response.status_code == 200
    data = response.json()
    assert "vector_store" in data
    assert "config" in data

def test_rag_query_invalid():
    # Test invalid request format
    response = client.post("/v1/rag/query", json={"invalid": "field"})
    assert response.status_code == 422

def test_rag_query_mocked():
    # We could mock RagService inside the endpoint if we wanted 100% isolation,
    # but since unit tests already cover the service, here we check endpoint wiring.
    # If API keys are missing, the endpoint might return 200 with "LLM provider not configured" 
    # message based on our current RagService logic.
    response = client.post("/v1/rag/query", json={"query": "test query", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
