
import pytest
import os
import shutil
from langchain_core.documents import Document
from app.rag.ingest import Ingestor
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TestIngestor:
    @pytest.fixture
    def setup_data(self):
        # Create a temp directory for test data
        test_dir = "data/test_temp_docs"
        os.makedirs(test_dir, exist_ok=True)
        yield test_dir
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    def test_text_splitter_behavior(self):
        # TASK-021: Test text splitter
        text = "a" * 1000  # 1000 chars
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.create_documents([text])
        
        # 1000 chars should be split into 2 chunks (800 + 200 overlap logic)
        assert len(chunks) >= 2
        assert len(chunks[0].page_content) <= 800

    def test_load_text_file(self, setup_data):
        # Load a simple text file
        filepath = os.path.join(setup_data, "test.txt")
        with open(filepath, "w") as f:
            f.write("Hello world")
        
        ing = Ingestor()
        docs = ing.load_docs(setup_data)
        
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"
        # Check relative path in metadata
        assert docs[0].metadata["source"] == "test.txt"
        assert docs[0].metadata["type"] == "text"

    def test_load_pdf_file_mock(self, setup_data, monkeypatch):
        # Mock pypdf reader since we don't want to rely on a real binary PDF in unit tests
        # or we can create a minimal valid PDF if needed, but mocking is easier for logic test.
        import sys
        from unittest.mock import MagicMock
        
        # We need to mock pypdf before it's imported in the function
        # But ingest.py does 'from pypdf import PdfReader' inside the loop.
        # So we mock sys.modules
        
        mock_pdf_module = MagicMock()
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF Content"
        mock_reader.pages = [mock_page]
        mock_pdf_module.PdfReader.return_value = mock_reader
        
        with monkeypatch.context() as m:
            m.setitem(sys.modules, 'pypdf', mock_pdf_module)
            
            # Create a dummy .pdf file (content doesn't matter as we mock the reader)
            filepath = os.path.join(setup_data, "test.pdf")
            with open(filepath, "wb") as f:
                f.write(b"%PDF-1.4...")
            
            ing = Ingestor()
            # We need to ensure logic enters the 'if pdf' block
            docs = ing.load_docs(setup_data)
            
            # Since we wrote a real file, os.walk finds it.
            # pypdf.PdfReader is mocked.
            
            assert len(docs) == 1
            assert docs[0].page_content == "PDF Content"
            assert docs[0].metadata["type"] == "pdf"

    def test_corrupt_file_handling(self, setup_data):
        # Create a file that would crash utf-8 reading if not handled
        filepath = os.path.join(setup_data, "bad.txt")
        with open(filepath, "wb") as f:
            f.write(b'\x80\x81\xff') # Invalid utf-8
            
        ing = Ingestor()
        docs = ing.load_docs(setup_data)
        
        # Should catch error and return empty list (or list without this file)
        # ingest.py uses 'errors=ignore' for text files, so it might actually return garbage string.
        # Let's verify it doesn't Crash.
        assert isinstance(docs, list)
