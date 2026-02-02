"""
Text search module using SQLite FTS5 (Full-Text Search).

This module provides persistent text search with BM25 ranking.
"""

from sqlitesearch.text.fts import TextSearchIndex

__all__ = ["TextSearchIndex"]
