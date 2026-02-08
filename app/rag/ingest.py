
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                docs.append(Document(page_content=text, metadata={"source": p}))
        return docs

    def build(self, docs_root: str = 'data/sample_docs'):
        docs = self.load_docs(docs_root)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        # create embeddings
        vectors = [self.embedder.encode(d.page_content) for d in chunks]
        # FAISS expects numpy arrays; langchain FAISS wrapper can build from embeddings, but we keep minimal here
        import numpy as np
        import faiss
        xb = np.vstack(vectors).astype('float32')
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)
        # persist: save index + mapped chunks
        faiss.write_index(index, self.store_path)
        with open(self.store_path + '.meta.pkl', 'wb') as f:
            pickle.dump(chunks, f)
        return len(chunks)

if __name__ == '__main__':
    ing = Ingestor()
    n = ing.build()
    print(f'Built FAISS index with {n} chunks')
