
import faiss, os, pickle
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from ..models import AnswerPayload, Citation

class RagService:
    def __init__(self, store_path: str = '.vector_store/faiss', embed_model: str = 'all-MiniLM-L6-v2'):
        if not os.path.exists(store_path):
            raise RuntimeError('Vector store not found. Run ingestion first.')
        self.index = faiss.read_index(store_path)
        with open(store_path + '.meta.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        self.embedder = SentenceTransformer(embed_model)

    def preprocess_query(self, query: str) -> str:
        # Preprocessing: strip whitespace, remove newlines, limit length
        q = query.strip().replace('\n', ' ')
        return q[:1000]

    def retrieve(self, query: str, top_k: int = 4):
        clean_query = self.preprocess_query(query)
        q = self.embedder.encode(clean_query).astype('float32')
        D, I = self.index.search(np.expand_dims(q, 0), top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            doc = self.chunks[int(idx)]
            results.append({
                'id': int(idx),
                'score': float(score),
                'source': doc.metadata.get('source','unknown'),
                'chunk': doc.page_content[:800]
            })
        return results

    def synthesize(self, query: str, ctx: List[Dict]) -> str:
        # Placeholder: deterministic stitcher with simple template
        # Replace with provider LLM call and prompt template
        bullets = '\n'.join([f"- ({c['score']:.2f}) {c['chunk'][:200].replace('\n',' ')}" for c in ctx])
        answer = f"Answer (draft) for: '{query}'.\nKey retrieved evidence:\n{bullets}\n\n(Replace with LLM call and cite sources.)"
        return answer

    def answer(self, query: str, top_k: int = 4) -> AnswerPayload:
        ctx = self.retrieve(query, top_k=top_k)
        text = self.synthesize(query, ctx)
        cits = [Citation(**c) for c in ctx]
        return AnswerPayload(answer=text, citations=cits)
