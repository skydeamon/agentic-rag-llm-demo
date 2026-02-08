import os
import shutil
import tempfile
import pytest
from app.rag.retriever import RagService
from app.rag.ingest import Ingestor
from app.models import AnswerPayload, Citation

@pytest.fixture
def retriever():
    # Create temp directories for docs and index
    test_docs_dir = tempfile.mkdtemp()
    test_index_dir = tempfile.mkdtemp()
    index_path = os.path.join(test_index_dir, 'index')

    try:
        # Create dummy documents
        with open(os.path.join(test_docs_dir, "doc1.txt"), "w") as f:
            f.write("Coronary physiology is the study of blood flow to the heart muscle.")
        with open(os.path.join(test_docs_dir, "doc2.txt"), "w") as f:
            f.write("The capital of France is Paris.")

        # Build index
        ingestor = Ingestor(store_path=index_path)
        ingestor.build(docs_root=test_docs_dir)

        # Mock LLM Provider
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Mocked LLM Answer"

        # Initialize service with temp index and mock LLM
        service = RagService(store_path=index_path, llm_provider=mock_llm)
        yield service, mock_llm

    finally:
        # Cleanup
        shutil.rmtree(test_docs_dir)
        shutil.rmtree(test_index_dir)

def test_preprocess_query(retriever):
    svc, _ = retriever
    # Test whitespace stripping and newline removal
    raw_query = "  What is \n coronary   physiology?  \n"
    expected = "What is   coronary   physiology?"
    processed = svc.preprocess_query(raw_query)
    assert processed == expected

def test_preprocess_query_length_limit(retriever):
    svc, _ = retriever
    # Test truncation
    long_query = "a" * 2000
    processed = svc.preprocess_query(long_query)
    assert len(processed) == 1000

def test_retrieve_returns_results(retriever):
    svc, _ = retriever
    res = svc.retrieve("coronary physiology", top_k=2)
    assert len(res) > 0
    # Check that we retrieved the relevant doc
    # doc1 has "Coronary physiology"
    assert "Coronary" in res[0]['chunk']
    assert isinstance(res[0], dict)
    assert 'score' in res[0]
    assert 'chunk' in res[0]
    assert 'source' in res[0]

def test_answer_returns_payload(retriever):
    svc, mock_llm = retriever
    # Mock LLM response that mentions only doc1
    mock_llm.generate.return_value = "According to [Source: doc1.txt], coronary physiology is vital."
    
    payload = svc.answer("coronary physiology", top_k=2)
    
    assert isinstance(payload, AnswerPayload)
    assert "coronary physiology is vital" in payload.answer
    # Should only have 1 citation because only doc1.txt was mentioned
    assert len(payload.citations) == 1
    assert "doc1.txt" in payload.citations[0].source

def test_answer_handles_llm_error(retriever):
    svc, mock_llm = retriever
    mock_llm.generate.side_effect = Exception("API Timeout")
    
    payload = svc.answer("coronary physiology", top_k=2)
    
    # Should degrade gracefully
    assert "technical issue" in payload.answer
    assert "API Timeout" in payload.answer
    assert len(payload.citations) == 0

def test_answer_no_mentions_yields_no_citations(retriever):
    svc, mock_llm = retriever
    # LLM provides answer but no formatting [Source: ...]
    mock_llm.generate.return_value = "I think it's study of heart."
    
    payload = svc.answer("coronary physiology", top_k=2)
    
    assert len(payload.citations) == 0

