"""
TextSearchIndex - Persistent full-text search using SQLite FTS5.

This module provides a text search index backed by SQLite's FTS5 (Full-Text Search)
extension, enabling efficient BM25 ranking and boolean queries.
"""

import json
import re
import sqlite3
import threading
from typing import Any, Optional


class TextSearchIndex:
    """
    A persistent text search index using SQLite FTS5.

    This index stores documents in a SQLite database and uses FTS5 for efficient
    full-text search with BM25 ranking.

    API matches minsearch.Index for easy migration:
    - __init__(text_fields, keyword_fields=None, id_field=None)
    - fit(docs) - Index documents (only if index is empty)
    - add(doc) - Add a single document to existing index
    - search(query, filter_dict=None, boost_dict=None, num_results=10, output_ids=False)

    Example:
        >>> index = TextSearchIndex(
        ...     text_fields=["title", "description"],
        ...     keyword_fields=["category"],
        ...     id_field="id",
        ...     db_path="search.db"
        ... )
        >>> index.fit([{"id": 1, "title": "Hello", "description": "World"}])
        >>> results = index.search("hello world")
    """

    def __init__(
        self,
        text_fields: list[str],
        keyword_fields: Optional[list[str]] = None,
        id_field: Optional[str] = None,
        db_path: str = "sqlitesearch.db",
        stemming: bool = False,
    ):
        """
        Initialize the TextSearchIndex.

        Args:
            text_fields: List of field names to index with FTS5.
            keyword_fields: List of field names for exact filtering (not full-text searched).
            id_field: Field name to use as document ID. If None, auto-generates IDs.
            db_path: Path to the SQLite database file.
            stemming: If True, use Porter stemmer for better matching (e.g., "running" matches "run").
        """
        self.text_fields = text_fields
        self.keyword_fields = list(keyword_fields) if keyword_fields is not None else []
        self.id_field = id_field
        self.db_path = db_path
        self.stemming = stemming
        self._local = threading.local()

        # Add id_field to keyword_fields if provided and not already there
        if self.id_field and self.id_field not in self.keyword_fields:
            self.keyword_fields.append(self.id_field)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build keyword column definitions
        keyword_cols = []
        for field in self.keyword_fields:
            keyword_cols.append(f', "{field}" TEXT')
        keyword_sql = "\n".join(keyword_cols)

        # Create main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_json TEXT NOT NULL{keyword_sql}
            )
        """)

        # Create FTS5 virtual table
        # Note: tokenizer applies to both indexing AND querying
        fts_columns = ["docid"] + [f'"{col}"' for col in self.text_fields]
        fts_col_list = ", ".join(fts_columns)

        if self.stemming:
            tokenizer = "tokenize='porter unicode61'"
        else:
            tokenizer = "tokenize='unicode61'"

        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                {fts_col_list},
                {tokenizer}
            )
        """)

        # Create indexes on keyword fields for faster filtering
        for field in self.keyword_fields:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{field} ON docs ("{field}")')

        conn.commit()

    def _is_empty(self) -> bool:
        """Check if the index is empty."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM docs")
        row = cursor.fetchone()
        return row["count"] == 0

    def fit(self, docs: list[dict[str, Any]]) -> "TextSearchIndex":
        """
        Index the provided documents.

        Only works if the index is empty. Use add() to append documents.

        Args:
            docs: List of documents to index. Each document is a dictionary.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If the index already contains documents.
        """
        if not self._is_empty():
            raise ValueError(
                "Index already contains documents. "
                "Use clear() to reset the index or add() to append documents."
            )

        return self._add_docs(docs)

    def add(self, doc: dict[str, Any]) -> "TextSearchIndex":
        """
        Add a single document to the index.

        Args:
            doc: Document to add. Must be a dictionary.

        Returns:
            self for method chaining.
        """
        return self._add_docs([doc])

    def _add_docs(self, docs: list[dict[str, Any]]) -> "TextSearchIndex":
        """Internal method to add documents to the index."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build column lists
        all_cols = ["doc_json"] + [f'"{field}"' for field in self.keyword_fields]
        col_names = ", ".join(all_cols)
        placeholders = ", ".join(["?"] * len(all_cols))

        for doc in docs:
            doc_json = json.dumps(doc)
            keyword_vals = [doc.get(field) for field in self.keyword_fields]

            # Insert into main table
            cursor.execute(
                f"INSERT INTO docs ({col_names}) VALUES ({placeholders})",
                [doc_json] + keyword_vals
            )
            doc_id = cursor.lastrowid

            # Insert into FTS5 table
            fts_cols = [doc_id] + [str(doc.get(field, "")) for field in self.text_fields]
            fts_placeholders = ", ".join(["?"] * len(fts_cols))
            fts_col_names = ", ".join(["docid"] + [f'"{field}"' for field in self.text_fields])
            cursor.execute(
                f"INSERT INTO docs_fts ({fts_col_names}) VALUES ({fts_placeholders})",
                fts_cols
            )

        conn.commit()
        return self

    def clear(self) -> "TextSearchIndex":
        """
        Clear all documents from the index.

        Returns:
            self for method chaining.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM docs")
        cursor.execute("DELETE FROM docs_fts")

        conn.commit()
        return self

    def search(
        self,
        query: str,
        filter_dict: Optional[dict[str, Any]] = None,
        boost_dict: Optional[dict[str, float]] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search the index with the given query.

        Args:
            query: The search query string. Supports FTS5 query syntax.
            filter_dict: Dictionary of keyword fields to filter by.
            boost_dict: Dictionary of boost scores for text fields.
            num_results: Maximum number of results to return.
            output_ids: If True, adds an 'id' field with the document ID.

        Returns:
            List of documents matching the search criteria, ranked by relevance.
        """
        if filter_dict is None:
            filter_dict = {}
        if boost_dict is None:
            boost_dict = {}

        # Handle empty query - return empty results
        if not query or not query.strip():
            return []

        conn = self._get_conn()
        cursor = conn.cursor()

        # Build FTS5 query with boosts
        fts_query = self._build_fts_query(query, boost_dict)

        # Build WHERE clause for keyword filters
        where_clauses = []
        where_params = []

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                if value is None:
                    where_clauses.append(f'd."{field}" IS NULL')
                else:
                    where_clauses.append(f'd."{field}" = ?')
                    where_params.append(value)

        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Execute search query - simpler without content table
        search_query = f"""
            SELECT
                f.docid,
                d.doc_json,
                bm25(docs_fts) AS score
            FROM docs_fts f
            JOIN docs d ON f.docid = d.id
            WHERE docs_fts MATCH ?{where_sql}
            ORDER BY score
            LIMIT ?
        """

        cursor.execute(search_query, [fts_query] + where_params + [num_results])
        rows = cursor.fetchall()

        results = []
        for row in rows:
            doc = json.loads(row["doc_json"])
            if output_ids:
                # Use id_field value if available, otherwise use database id
                if self.id_field:
                    doc_id = doc.get(self.id_field)
                    # Try to convert to int if possible
                    if doc_id is not None and str(doc_id).isdigit():
                        doc_id = int(doc_id)
                else:
                    doc_id = row["docid"]
                doc["id"] = doc_id
            results.append(doc)

        return results

    def _build_fts_query(self, query: str, boost_dict: dict[str, float]) -> str:
        """
        Build an FTS5 query with boost weights.

        Args:
            query: The raw query string.
            boost_dict: Field -> boost weight mapping.

        Returns:
            An FTS5 query string.
        """
        query_terms = self._extract_query_terms(query)

        # Note: empty queries are handled in search() method
        if not boost_dict:
            # OR query - any term matches (better recall)
            return " OR ".join(query_terms)

        # Build boosted query for each field
        parts = []

        for field in self.text_fields:
            boost = boost_dict.get(field, 1.0)
            if boost == 0:
                continue

            # Use OR within field for better recall
            field_query = " OR ".join(query_terms)
            parts.append(f'"{field}":({field_query})')

        return " OR ".join(parts) if parts else " OR ".join(query_terms)

    def _extract_query_terms(self, query: str) -> list[str]:
        """
        Extract search terms from a query string.

        Simple tokenizer that splits on whitespace and special characters.
        """
        terms = re.findall(r'\w+', query.lower())
        return terms if terms else [query]

    def _escape_fts_query(self, query: str) -> str:
        """
        Escape special FTS5 characters in a query.

        FTS5 special characters: " ( ) [ ] * & | - +
        """
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def __enter__(self) -> "TextSearchIndex":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
