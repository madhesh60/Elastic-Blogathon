#!/usr/bin/env python3
"""
TF-IDF Text Search Implementation
Demonstrates how TF-IDF works for text retrieval.
Based on blog.md
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(query, docs):
    """Calculate TF-IDF scores for a query against a set of documents"""
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the documents into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # Transform the query using the fitted vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between the query and documents
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    return scores

def main():
    print("🔍 TF-IDF Search Demo")
    print("=" * 50)
    
    # Example Documents based on blog.md
    docs = [
        "machine learning is powerful",
        "deep learning uses neural networks",
        "machine learning and AI are changing the world",
        "traditional software engineering focuses on rules",
        "data science relies heavily on statistical features"
    ]
    
    print(f"📚 Loaded {len(docs)} documents\n")
    
    # Test queries
    queries = ["machine learning", "neural networks algorithms", "rules in software"]
    
    for query in queries:
        print(f"🔎 Testing query: '{query}'")
        
        scores = tfidf_search(query, docs)
        
        # Get indices sorted by score in descending order
        top_indices = scores.argsort()[::-1]
        
        print("Results:")
        for i, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                print(f"  {i}. Score: {scores[idx]:.4f} - {docs[idx]}")
        
        print("-" * 50)
        
    print("✅ TF-IDF search analysis completed!")

if __name__ == "__main__":
    main()
