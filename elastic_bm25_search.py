#!/usr/bin/env python3
"""
Simple BM25 Search Demo using Elasticsearch
Based on blog.md
"""

from elasticsearch import Elasticsearch

print("🔍 Elasticsearch BM25 Search Demo")
print("=" * 50)

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "blog_docs"

# Example Documents based on blog.md
docs = [
    {"id": 1, "text": "machine learning is powerful"},
    {"id": 2, "text": "deep learning uses neural networks"},
    {"id": 3, "text": "machine learning and AI"}
]

print(f"📚 Loaded {len(docs)} documents\n")

try:
    # Create Index
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "text": {"type": "text"}
            }
        }
    )

    # Index the documents
    for doc in docs:
        es.index(index=index_name, id=doc["id"], document={"text": doc["text"]})

    # Refresh index
    es.indices.refresh(index=index_name)

    # Example searches
    queries = ["machine learning", "neural networks", "powerful AI"]

    for query in queries:
        print(f"🔎 Searching for: '{query}'")
        
        # Keyword search (BM25 is the default scoring in Elasticsearch)
        response = es.search(
            index=index_name,
            query={
                "match": {
                    "text": query
                }
            }
        )
        
        # Get top results
        hits = response['hits']['hits']
        
        print("Results:")
        for i, hit in enumerate(hits, 1):
            print(f"  {i}. Score: {hit['_score']:.4f} - {hit['_source']['text']}")
        print()

    print("✅ BM25 search completed!")
except Exception as e:
    print(f"Ensure Elasticsearch is running on localhost:9200. Error: {e}")
