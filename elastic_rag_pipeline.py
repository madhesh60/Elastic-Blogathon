#!/usr/bin/env python3
"""
RAG Pipeline Implementation using Elasticsearch
Retrieval-Augmented Generation simple demo
Based on blog.md
"""

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

def mock_llm_generate(context, query):
    """A mock LLM for demonstration purposes"""
    return f"Based on the context ('{context}'), the answer to '{query}' is strongly related to machine learning capabilities."

def main():
    print("🧠 RAG Pipeline with Elasticsearch")
    print("=" * 50)
    
    try:
        es = Elasticsearch("http://localhost:9200")
        index_name = "hybrid_blog_docs" # Reusing previous index
        
        print("Loading Embedding Model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        query = "What uses neural networks?"
        print(f"\nUser Query: {query}\n")
        
        print("Step 1 & 2: Embedding Query and Retrieving Context from Elasticsearch...")
        query_vector = model.encode(query).tolist()
        
        response = es.search(
            index=index_name,
            knn={
                "field": "vector",
                "query_vector": query_vector,
                "k": 1, 
                "num_candidates": 5
            }
        )
        
        # Extract Context
        hits = response['hits']['hits']
        if hits:
            top_context = hits[0]['_source']['text']
            print(f"Retrieved Context: '{top_context}' (Score: {hits[0]['_score']:.4f})\n")
            
            print("Step 3 & 4: Generating Response using LLM...")
            answer = mock_llm_generate(top_context, query)
            print(f"🤖 LLM Response:\n{answer}")
        else:
            print("No context found. Did you run elastic_hybrid_search.py first?")
            
    except Exception as e:
        print(f"Ensure Elasticsearch is running. Error: {e}")

if __name__ == "__main__":
    main()
