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

        # Initialize service with temp index
        service = RagService(store_path=index_path)
        yield service

    finally:
        # Cleanup
        shutil.rmtree(test_docs_dir)
        shutil.rmtree(test_index_dir)

def test_preprocess_query(retriever):
    # Test whitespace stripping and newline removal
    raw_query = "  What is \n coronary   physiology?  \n"
    expected = "What is   coronary   physiology?"
    processed = retriever.preprocess_query(raw_query)
    assert processed == expected

def test_preprocess_query_length_limit(retriever):
    # Test truncation
    long_query = "a" * 2000
    processed = retriever.preprocess_query(long_query)
    assert len(processed) == 1000

def test_retrieve_returns_results(retriever):
    res = retriever.retrieve("coronary physiology", top_k=2)
    assert len(res) > 0
    # Check that we retrieved the relevant doc
    # doc1 has "Coronary physiology"
    assert "Coronary" in res[0]['chunk']
    assert isinstance(res[0], dict)
    assert 'score' in res[0]
    assert 'chunk' in res[0]
    assert 'source' in res[0]

def test_answer_returns_payload(retriever):
    # Test high-level answer method returns Pydantic model
    payload = retriever.answer("coronary physiology", top_k=2)
    assert isinstance(payload, AnswerPayload)
    assert len(payload.citations) > 0
    assert isinstance(payload.citations[0], Citation)
    assert len(payload.answer) > 0
