
import time
import statistics
import timeit
from app.rag.retriever import RagService
import os

def benchmark():
    print("Initializing RAG Service...")
    # Ensure we use an existing index or creating a temp one if not found (but for benchmark we usually want the real one)
    # Check if default index exists, if not, warn.
    if not os.path.exists('.vector_store/faiss'):
         print("WARNING: Default vector store not found. Benchmark might fail or need ingestion first.")
         print("Run 'bash scripts/ingest.sh' first.")
         return

    try:
        svc = RagService()
    except Exception as e:
        print(f"Failed to init RAG: {e}")
        return

    queries = [
        "What is the capital of France?",
        "Explain coronary physiology",
        "How do transformers work?",
        "Optimization techniques for RAG",
        "Python GIL details",
        "Deep learning history",
        "FastAPI middleware",
        "LangGraph state management",
        "Vector database comparison",
        "Embedding models benchmark"
    ] * 5  # 50 queries

    print(f"Running {len(queries)} queries...")
    latencies = []
    
    start_total = time.time()
    for q in queries:
        t0 = time.time()
        svc.retrieve(q, top_k=4)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
    end_total = time.time()

    total_time = end_total - start_total
    avg_lat = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] # 95th percentile
    throughput = len(queries) / total_time

    print(f"\n--- Benchmark Results ---")
    print(f"Total Queries: {len(queries)}")
    print(f"Total Time:    {total_time:.2f} s")
    print(f"Throughput:    {throughput:.2f} QPS")
    print(f"Avg Latency:   {avg_lat:.2f} ms")
    print(f"P95 Latency:   {p95:.2f} ms")
    print(f"-------------------------")

if __name__ == "__main__":
    benchmark()
