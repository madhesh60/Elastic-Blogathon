
# **RAG — Building Reliable AI Pipelines**

## **AUTHOR INTRO**

I am Madhesh, a passionate developer with a strong interest in AI and DevOps. I enjoy learning new technologies, and I have always wanted to start writing blogs to connect with people. I chose to work on RAG because large language models (LLMs) are everywhere, and RAG adds significant power to them by providing proper context for user queries.

---

## **ABSTRACT**

LLMs often hallucinate on domain-specific or recent data because they don’t have the proper context for user queries. Traditional LLM outputs rely solely on trained data, which may not contain up-to-date or domain-specific information. RAG overcomes these problems with strong retrieval pipelines. In this blog, I walk through designing and implementing a complete RAG pipeline using Elastic as the vector database. From ingesting documents to semantic retrieval and LLM augmentation, discover how Elastic’s vector capabilities deliver accurate, hallucination-resistant AI applications.

---

## **NAIVE SEARCH (KEYWORD SEARCH)**

The naive way to search for relevant content in a document or database is by using a basic keyword search.

Example — search in a file:

```bash
grep "keyword" file.txt
```

Example — SQL keyword search in a database:

```sql
SELECT * FROM table_name WHERE column_name LIKE '%keyword%';
```

Keyword search works by finding exact matches. But if the user uses different words with the same meaning, keyword search fails. That is where semantic search and vector embeddings become useful.

---

## **TF-IDF**

TF-IDF is a classic method to score how important a term is in a document relative to a corpus.

* **TF (Term Frequency)** looks at how many times a word appears in a specific document.
* **DF (Document Frequency)** is the number of documents where the word appears.
* **IDF (Inverse Document Frequency)** measures the importance of the word across the entire document set.

Formulas:

```
DF(t) = number of documents containing term t

IDF(t) = log(N / DF(t)),   where N = total number of documents
```

TF-IDF weights terms that are frequent in a document but rare in the corpus, giving more relevant ranking than pure keyword counts.

---

## **BM25**

BM25 is a ranking algorithm used in retrieval systems to determine the relevance of documents to a given user query. It is the default ranking algorithm used in systems like Elasticsearch and Whoosh.

BM25 improves over TF-IDF by:

* Normalizing for document length
* Saturating term frequency (more occurrences do not increase importance linearly)
* Producing better relevance scoring in practice

Here’s how to compute BM25 in Python:

```python
from rank_bm25 import BM25Okapi

docs = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning and AI"
]

tokenized = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized)

query = "machine learning".split()
scores = bm25.get_scores(query)
print(scores)
```

BM25 produces a score for each document based on the query and ranks them by relevance.

---

## **VECTOR EMBEDDINGS**

When a user query uses a different word but similar meaning, keyword methods fail. This is where **vector embeddings** solve the problem.

Embeddings transform text into numerical vectors that capture semantic meaning. Similar texts have vectors close to each other in vector space.

Here’s how to generate embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["machine learning", "deep learning"]
vectors = model.encode(texts)
print(vectors.shape)   # (2, 384)
```

---

## **INTRO TO RAG PIPELINES**

A RAG (Retrieval-Augmented Generation) pipeline has three main stages:

1. **Document Ingestion:** Load or collect raw data (documents, databases, live feeds).
2. **Retrieval:** Find relevant context for a user query.
3. **Generation:** Use retrieved context to generate accurate answers with an LLM.

This process reduces hallucination by grounding the model in real content.

---

## **RELEVANT CONTEXT AND PREPROCESSING**

First, ingest raw data into the RAG system. To make it effective, choose proper preprocessing techniques:

### **Chunking**

Chunking breaks large documents into smaller pieces that are easier to index and retrieve. Good chunking balances context with retrieval efficiency.

---

## **VECTOR DATABASE**

Once text is chunked and embedded into vectors, store it in a vector database (e.g., Elasticsearch). The vector DB stores embeddings and performs similarity search to match user queries with relevant chunks.

---

## **ELASTICSEARCH – SETUP & CODE**

### **1. Create index with vector field**

```bash
curl -X PUT "localhost:9200/docs" -H "Content-Type: application/json" -d '
{
  "mappings": {
    "properties": {
      "text":       { "type": "text" },
      "vector":     { "type": "dense_vector", "dims": 384 }
    }
  }
}'
```

### **2. Insert document with embedding**

```bash
curl -X POST "localhost:9200/docs/_doc" -H "Content-Type: application/json" -d '
{
  "text":   "machine learning is powerful",
  "vector": [0.12, -0.93, ...]   # real embedding vector
}'
```

### **3. Query using BM25 (keyword search)**

```bash
curl -X GET "localhost:9200/docs/_search" -H "Content-Type: application/json" -d '
{
  "query": {
    "match": {
      "text": "machine learning"
    }
  }
}'
```

### **4. Query using Vector Similarity**

```bash
curl -X GET "localhost:9200/docs/_search" -H "Content-Type: application/json" -d '
{
  "knn": {
    "field":        "vector",
    "query_vector": [0.12, -0.93, ...],
    "k":            3,
    "num_candidates": 10
  }
}'
```

### **5. Hybrid Search (BM25 + Vector)**

```bash
curl -X GET "localhost:9200/docs/_search" -H "Content-Type: application/json" -d '
{
  "query": {
    "bool": {
      "should": [
        { "match": { "text": "machine learning" }},
        {
          "knn": {
            "field":        "vector",
            "query_vector": [0.12, -0.93, ...],
            "k": 3,
            "num_candidates": 10
          }
        }
      ]
    }
  }
}'
```

Hybrid search combines keyword ranking (BM25) and semantic ranking (vector similarity).

---

## **RERANKING**

Reranking is a post-processing step that improves result relevance by applying stronger scoring methods. It considers semantic relevance and similarity to reorder results for better quality. Reranking is more computationally expensive and is usually applied only to top results.

---

## **INTEGRATING ELASTIC WITH LLMS**

Elastic can serve as the retrieval backend for a RAG system. When a user query arrives:

1. The query is embedded.
2. Elastic retrieves the most similar chunks (vector search).
3. The retrieved chunks are passed to the LLM.
4. The LLM generates an answer grounded in retrieved context.

This integration reduces hallucination and increases response accuracy.

---

## **PRODUCTION INSIGHTS AROUND ELASTICSEARCH AND RAG**

When deploying semantic search and RAG pipelines to production environments, consider these real-world best practices:

1. **Hardware & Memory Considerations:** Vector search demands heavy memory usage. Elasticsearch relies heavily on off-heap memory (page cache) for mmap directories to efficiently execute HNSW searches. Allocate around 50% of the instance memory to the filesystem cache and prefer instances with fast NVMe SSDs for production-level response times.
2. **Indexing Performance vs. Search Performance:** When utilizing `dense_vector` fields, modifying `m` (number of neighbors) and `ef_construction` parameters in Elasticsearch’s HNSW graph tuning impacts your ingest and recall speeds differently. High `ef_construction` increases build time and index size but significantly boosts search precision.
3. **Filtering Matters:** A pure K-Nearest Neighbor (kNN) search will match universally against all chunks, which isn't always contextually correct within multi-tenant systems. Implement pre-filtering in Elasticsearch (using standard metadata filters attached to KNN requests) to strictly scope out documents based on attributes (e.g., specific departments or user-roles) before executing resource-heavy vector math.
4. **Monitoring & Logging Metrics:** LLMs are often unpredictable, and retrieving the wrong contextual chunk guarantees a hallucinated answer. Establish rigorous evaluation systems for retrieval hit rates before scaling up your LLMs. Track user feedback loops to determine whether BM25, K-NN, or a specific Hybrid alpha weighting setup performs best for your unique domain vocabulary over time.
5. **Fallbacks and Timeouts:** Calling out to an embedding microservice and subsequently to an LLM are latency-heavy activities. Configure Elasticsearch timeout safeguards explicitly to prevent blocking operational resources unnecessarily, and fall back seamlessly to traditional BM25 query modes if the vector generation endpoints ever degrade or experience throttling in production.

---

## **CONCLUSION**

Engineers can over-engineer things. The true value of RAG lies in strengthening LLM responses with real context from scalable systems like Elasticsearch. RAG makes LLMs less prone to hallucination and vastly improves relevance and accuracy.

If neither step 1 (retrieval) nor step 2 (generation) gives high-quality results, then consider improving both parts of a RAG pipeline and the retrieval components.

---

If you want, I can further **add an architecture diagram and SEO keywords** for this blog.
