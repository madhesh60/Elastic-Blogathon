#!/usr/bin/env python3
"""
Hybrid Search Implementation using Elasticsearch
Combines BM25 (Keyword) and Vector embeddings (Semantic)
Based on blog.md
"""

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

def hybrid_search(es, index_name, query, model):
    """Perform hybrid search in Elasticsearch"""
    
    query_vector = model.encode(query).tolist()
    
    response = es.search(
        index=index_name,
        query={
            "bool": {
                "should": [
                    {"match": {"text": query}}
                ]
            }
        },
        knn={
            "field": "vector",
            "query_vector": query_vector,
            "k": 3,
            "num_candidates": 10,
            "boost": 0.5
        },
        size=3
    )
    
    return response['hits']['hits']

def main():
    print("🔍 Elasticsearch Hybrid Search Demo")
    print("=" * 50)
    
    try:
        # Connect to Elasticsearch
        es = Elasticsearch("http://localhost:9200")
        index_name = "hybrid_blog_docs"
        
        # Load model
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        docs = [
            {"id": 1, "text": "machine learning is powerful"},
            {"id": 2, "text": "deep learning uses neural networks"},
            {"id": 3, "text": "machine learning and AI"}
        ]
        
        # Create Index with vector field
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            
        es.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector", "dims": 384}
                }
            }
        )
        
        # Index documents with vectors
        print(f"📚 Indexing {len(docs)} documents with embeddings...")
        for doc in docs:
            vector = model.encode(doc["text"]).tolist()
            es.index(index=index_name, id=doc["id"], document={"text": doc["text"], "vector": vector})
            
        es.indices.refresh(index=index_name)
        
        # Test queries
        queries = ["AI systems", "neural connections"]
        
        for query in queries:
            print(f"\n🔎 Testing Hybrid query: '{query}'")
            print("-" * 40)
            
            hits = hybrid_search(es, index_name, query, model)
            
            print("Top 3 results:")
            for i, hit in enumerate(hits, 1):
                print(f"  {i}. Score: {hit['_score']:.4f} - {hit['_source']['text']}")
                
        print(f"\n✅ Hybrid search analysis completed!")
    except Exception as e:
        print(f"Ensure Elasticsearch is running and SentenceTransformer is installed. Error: {e}")

if __name__ == "__main__":
    main()
