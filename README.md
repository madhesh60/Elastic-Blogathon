# Elastic-Blogathon 📝

This repository contains code examples and practical implementations demonstrating the core concepts of Semantic Search and **RAG (Retrieval-Augmented Generation)** using **Elasticsearch**. 

✨ **Purpose** 
This project was specifically created to accompany my submission for the **Elasticsearch Blogathon** competition. It provides clear, runnable code examples that map directly to the topics discussed in my blog post, helping developers understand how to build reliable AI pipelines.

---

## I) Repository Contents

This repository includes several independent Python scripts, each demonstrating a key component of modern search and retrieval:

### 1. **TF-IDF Search (`tfidf_search.py`)** 
Demonstrates the classic **Term Frequency-Inverse Document Frequency** scoring method. It uses `scikit-learn` to show how documents are ranked based on word importance before vector embeddings became the standard.

### 2. **BM25 Keyword Search (`elastic_bm25_search.py`)**
Shows how to implement a basic keyword search using Elasticsearch. BM25 is Elasticsearch's default ranking algorithm and is excellent for exact-match searches.

### 3. **Hybrid Search (`elastic_hybrid_search.py`)**
Combines the best of both worlds! This script demonstrates how to run a query that simultaneously evaluates:
*   **BM25 (Keyword matching)** via `match` queries.
*   **Semantic/Vector similarity (kNN)** using `SentenceTransformer` embeddings and `dense_vector` fields.

### 4. **RAG Pipeline Mock (`elastic_rag_pipeline.py`)**
A simplified representation of a **Retrieval-Augmented Generation** pipeline. It takes a user query, retrieves the most contextually relevant documents from Elasticsearch (using vectors), and passes that context to a "mock" LLM to generate a grounded, hallucination-free answer.

---

## II) Prerequisites & Setup

To run these examples locally, you will need:

1.  **Python 3.x** installed.
2.  **Elasticsearch** running locally on port `9200` (e.g., via Docker).
3.  The necessary Python packages. You can install them via pip:

```bash
pip install elasticsearch scikit-learn sentence-transformers
```

## III) Running the Examples

Ensure your Elasticsearch instance is up and running. Some scripts (like the RAG pipeline) depend on the indices created by others, so it's recommended to run them in this order to understand the progression:

1.  `python tfidf_search.py`
2.  `python elastic_bm25_search.py`
3.  `python elastic_hybrid_search.py`
4.  `python elastic_rag_pipeline.py`

---
*Created for the Elastic Blogathon*
