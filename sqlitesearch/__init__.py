"""
sqlitesearch - A tiny, SQLite-backed search library for small, local projects.

sqlitesearch provides persistent text search using SQLite FTS5 and persistent
vector search using LSH (random projections) with exact reranking.
"""

from sqlitesearch.text import TextSearchIndex
from sqlitesearch.vector import VectorSearchIndex

__version__ = "0.0.1"

__all__ = [
    "TextSearchIndex",
    "VectorSearchIndex",
]
