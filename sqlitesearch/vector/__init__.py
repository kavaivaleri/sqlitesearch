"""
Vector search module using LSH (Locality-Sensitive Hashing).

This module provides persistent vector search with LSH-based approximate
nearest neighbor search, followed by exact cosine similarity reranking.
"""

from sqlitesearch.vector.lsh import VectorSearchIndex

__all__ = ["VectorSearchIndex"]
