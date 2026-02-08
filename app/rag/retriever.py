
import faiss, os, pickle
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from ..models import AnswerPayload, Citation
from ..llm.provider import get_llm, LLMProvider
from ..llm.prompts import RAG_PROMPT
from ..config import settings

class RagService:
    def __init__(self, store_path: str = '.vector_store/faiss', embed_model: str = 'all-MiniLM-L6-v2', llm_provider: Optional[LLMProvider] = None):
        if not os.path.exists(store_path):
            raise RuntimeError('Vector store not found. Run ingestion first.')
        self.index = faiss.read_index(store_path)
        with open(store_path + '.meta.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        self.embedder = SentenceTransformer(embed_model)
        
        # Initialize LLM Provider (allow injection for testing)
        if llm_provider:
            self.llm = llm_provider
        else:
            try:
                self.llm = get_llm(settings.provider, settings.model_name)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM provider: {e}")
                self.llm = None

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
        # Format context for prompt
        context_str = ""
        for c in ctx:
            context_str += f"[Source: {c['source']}]\n{c['chunk']}\n\n"
        
        # If no context found
        if not context_str.strip():
            context_str = "No relevant documents found."
            
        # Generate prompt
        prompt = RAG_PROMPT.format(context=context_str, question=query)
        
        # Generate answer
        if self.llm:
            return self.llm.generate(prompt)
        else:
            return "LLM provider not configured. Please set API keys."

    def answer(self, query: str, top_k: int = 4) -> AnswerPayload:
        try:
            ctx = self.retrieve(query, top_k=top_k)
            text = self.synthesize(query, ctx)
            
            # Citation Filtering (TASK-055)
            # Find which sources were actually mentioned in the LLM text
            import re
            mentioned_sources = set(re.findall(r'\[Source:\s*(.*?)\]', text))
            
            final_citations = []
            for c in ctx:
                # Normalizing source name for comparison (assuming filenames)
                source_name = os.path.basename(c['source'])
                if source_name in mentioned_sources or c['source'] in mentioned_sources:
                    final_citations.append(Citation(**c))
            
            # Fallback: if no citations found in text but we have context, 
            # maybe the LLM forgot formatting. For transparency, we could include all.
            # But let's stick to strict if mentioned, otherwise empty for "no citations confirmed".
            # Or if LLM says "I don't know", the list will be filtered.
            
            return AnswerPayload(answer=text, citations=final_citations)

        except Exception as e:
            # Graceful Degradation (TASK-058)
            print(f"Error in RAG answer flow: {e}")
            return AnswerPayload(
                answer=f"I encountered a technical issue while processing your request. Error details: {str(e)}",
                citations=[]
            )
