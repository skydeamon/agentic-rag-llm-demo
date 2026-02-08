
from app.rag.retriever import RagService

def test_retrieve_after_ingest():
    svc = RagService()
    res = svc.retrieve("What is coronary physiology?", top_k=2)
    assert len(res) > 0
