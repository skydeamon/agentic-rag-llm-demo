
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle

class Ingestor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', store_path: str = '.vector_store/faiss'):
        self.embedder = SentenceTransformer(model_name)
        self.store_path = store_path
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

    def load_docs(self, root: str = 'data/sample_docs') -> List[Document]:
        docs = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    # Metadata enrichment (TASK-022)
                    rel_path = os.path.relpath(p, root)
                    
                    if fn.lower().endswith('.pdf'):
                        from pypdf import PdfReader
                        reader = PdfReader(p)
                        text = ""
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text() or ""
                            text += page_text
                            # Optional: Note page numbers in metadata if we were splitting by page
                        
                        # Just store one doc per file for now, simple approach
                        docs.append(Document(page_content=text, metadata={"source": rel_path, "type": "pdf"}))
                    else:
                        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        docs.append(Document(page_content=text, metadata={"source": rel_path, "type": "text"}))
                        
                except Exception as e:
                    # Error Handling (TASK-019)
                    print(f"Failed to process file {p}: {e}")
                    # Continue to next file instead of crashing
                    continue
        return docs

    def build(self, docs_root: str = 'data/sample_docs'):
        import tempfile
        import shutil
        import numpy as np
        import faiss

        print(f"Loading documents from {docs_root}...")
        try:
            docs = self.load_docs(docs_root)
            if not docs:
                print("No documents found. Splitting skipped.")
                return 0

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
            chunks = splitter.split_documents(docs)
            print(f"Split into {len(chunks)} chunks.")

            # Create embeddings
            print("Generating embeddings...")
            vectors = [self.embedder.encode(d.page_content) for d in chunks]
            
            if not vectors:
                print("No content to embed.")
                return 0
                
            xb = np.vstack(vectors).astype('float32')
            
            # Validation (TASK-026)
            if xb.shape[1] != 384:
                raise ValueError(f"Embedding dimension mismatch. Expected 384, got {xb.shape[1]}")
            print(f"Embeddings generated with shape {xb.shape}")

            # Atomic Write (TASK-029)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_index_path = os.path.join(tmpdir, 'index.faiss')
                tmp_meta_path = os.path.join(tmpdir, 'index.pkl')
                
                index = faiss.IndexFlatL2(xb.shape[1])
                index.add(xb)
                
                faiss.write_index(index, tmp_index_path)
                with open(tmp_meta_path, 'wb') as f:
                    pickle.dump(chunks, f)
                
                # Move to final location
                shutil.move(tmp_index_path, self.store_path)
                shutil.move(tmp_meta_path, self.store_path + '.meta.pkl')
                print(f"Index successfully saved to {self.store_path}")

            return len(chunks)

        except Exception as e:
            print(f"Critical error during ingestion: {e}")
            raise e
