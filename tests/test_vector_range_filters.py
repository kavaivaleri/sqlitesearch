"""
Tests for numeric and date range filters in VectorSearchIndex.
"""

import os
import tempfile
from datetime import date, datetime

import numpy as np
import pytest

from sqlitesearch import VectorSearchIndex


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing."""
    np.random.seed(42)
    return np.random.rand(10, 16).astype(np.float32)


@pytest.fixture
def sample_payload_with_numeric():
    """Sample payload with numeric fields."""
    return [
        {"doc_id": i, "category": "test", "price": 100 + i * 10, "rating": 4.0 + i * 0.1}
        for i in range(10)
    ]


@pytest.fixture
def sample_payload_with_dates():
    """Sample payload with date fields."""
    return [
        {
            "doc_id": i,
            "category": "test",
            "created_at": date(2024, 1, 1 + i),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_payload_with_datetime():
    """Sample payload with datetime fields."""
    return [
        {
            "doc_id": i,
            "category": "test",
            "created_at": datetime(2024, 1, 1, 10, 0, i),
        }
        for i in range(10)
    ]


class TestVectorNumericFilters:
    """Tests for numeric field range filters in VectorSearchIndex."""

    def test_numeric_field_initialization(self, temp_db):
        """Test that index initializes correctly with numeric fields."""
        index = VectorSearchIndex(
            keyword_fields=[],
            numeric_fields=["price", "rating"],
            db_path=temp_db
        )
        assert index.numeric_fields == ["price", "rating"]

    def test_numeric_filter_greater_than_or_equal(
        self, sample_vectors, sample_payload_with_numeric, temp_db
    ):
        """Test numeric filter with >= operator."""
        index = VectorSearchIndex(
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_numeric)

        query = sample_vectors[0]
        # Query vector 0 has price=100, so include it in the filter
        results = index.search(query, filter_dict={"price": [('>=', 100)]})
        assert len(results) > 0
        assert all(doc["price"] >= 100 for doc in results)

    def test_numeric_filter_less_than(
        self, sample_vectors, sample_payload_with_numeric, temp_db
    ):
        """Test numeric filter with < operator."""
        index = VectorSearchIndex(
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_numeric)

        query = sample_vectors[0]
        results = index.search(query, filter_dict={"price": [('<', 115)]})
        assert len(results) > 0
        assert all(doc["price"] < 115 for doc in results)

    def test_numeric_filter_range(
        self, sample_vectors, sample_payload_with_numeric, temp_db
    ):
        """Test numeric filter with range."""
        index = VectorSearchIndex(
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_numeric)

        query = sample_vectors[0]
        # Query vector 0 has price=100, so include that in the range
        results = index.search(query, filter_dict={"price": [('>=', 100), ('<', 130)]})
        assert len(results) > 0
        assert all(100 <= doc["price"] < 130 for doc in results)

    def test_numeric_filter_exact_match(
        self, sample_vectors, sample_payload_with_numeric, temp_db
    ):
        """Test numeric filter with exact match."""
        index = VectorSearchIndex(
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_numeric)

        query = sample_vectors[0]
        # Query vector 0 has price=100
        results = index.search(query, filter_dict={"price": 100})
        assert len(results) >= 1
        assert results[0]["price"] == 100

    def test_numeric_filter_with_keyword(
        self, sample_vectors, sample_payload_with_numeric, temp_db
    ):
        """Test combining numeric and keyword filters."""
        index = VectorSearchIndex(
            keyword_fields=["category"],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_numeric)

        query = sample_vectors[0]
        results = index.search(
            query,
            filter_dict={"category": "test", "price": [('>=', 100)]}
        )
        assert len(results) > 0
        assert all(doc["category"] == "test" and doc["price"] >= 100 for doc in results)


class TestVectorDateFilters:
    """Tests for date field range filters in VectorSearchIndex."""

    def test_date_field_initialization(self, temp_db):
        """Test that index initializes correctly with date fields."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        assert index.date_fields == ["created_at"]

    def test_date_filter_after(
        self, sample_vectors, sample_payload_with_dates, temp_db
    ):
        """Test date filter with >= operator."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_dates)

        query = sample_vectors[0]
        # Query vector 0 has date 2024-01-01, so use a range that includes it
        results = index.search(
            query,
            filter_dict={"created_at": [('>=', date(2024, 1, 1))]}
        )
        assert len(results) > 0
        assert all(doc["created_at"] >= date(2024, 1, 1) for doc in results)

    def test_date_filter_before(
        self, sample_vectors, sample_payload_with_dates, temp_db
    ):
        """Test date filter with < operator."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_dates)

        query = sample_vectors[0]
        # Query vector 0 has date 2024-01-01, so include it in the range
        results = index.search(
            query,
            filter_dict={"created_at": [('<', date(2024, 1, 2))]}
        )
        assert len(results) > 0
        assert all(doc["created_at"] < date(2024, 1, 2) for doc in results)
        results = index.search(
            query,
            filter_dict={"created_at": [('<', date(2024, 1, 5))]}
        )
        assert len(results) > 0
        assert all(doc["created_at"] < date(2024, 1, 5) for doc in results)

    def test_date_filter_range(
        self, sample_vectors, sample_payload_with_dates, temp_db
    ):
        """Test date filter with range."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_dates)

        query = sample_vectors[0]
        # Query vector 0 has date 2024-01-01, so include it in the range
        results = index.search(
            query,
            filter_dict={
                "created_at": [('>=', date(2024, 1, 1)), ('<', date(2024, 1, 5))]
            }
        )
        assert len(results) > 0
        assert all(
            date(2024, 1, 1) <= doc["created_at"] < date(2024, 1, 5)
            for doc in results
        )

    def test_date_filter_exact_match(
        self, sample_vectors, sample_payload_with_dates, temp_db
    ):
        """Test date filter with exact match."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_dates)

        query = sample_vectors[0]
        # Query vector 0 has date 2024-01-01, so filter for that date
        results = index.search(query, filter_dict={"created_at": date(2024, 1, 1)})
        assert len(results) >= 1
        assert results[0]["created_at"] == date(2024, 1, 1)

    def test_date_filter_with_datetime(
        self, sample_vectors, sample_payload_with_datetime, temp_db
    ):
        """Test date filter with datetime objects."""
        index = VectorSearchIndex(
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_with_datetime)

        query = sample_vectors[0]
        # Query vector 0 has datetime 2024-01-01 10:00:00, so include it in the range
        results = index.search(
            query,
            filter_dict={"created_at": [('>=', datetime(2024, 1, 1, 10, 0, 0))]}
        )
        assert len(results) > 0


class TestVectorCombinedFilters:
    """Tests for combining multiple filter types in VectorSearchIndex."""

    @pytest.fixture
    def sample_payload_combined(self):
        """Sample payload with all field types."""
        return [
            {
                "doc_id": i,
                "category": "test" if i % 2 == 0 else "prod",
                "price": 100 + i * 10,
                "created_at": date(2024, 1, 1 + i),
            }
            for i in range(10)
        ]

    def test_all_filter_types(
        self, sample_vectors, sample_payload_combined, temp_db
    ):
        """Test combining keyword, numeric, and date filters."""
        index = VectorSearchIndex(
            keyword_fields=["category"],
            numeric_fields=["price"],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_vectors, sample_payload_combined)

        query = sample_vectors[0]
        # Query vector 0 has category="test", price=100, created_at=2024-01-01
        # Use filters that include this document
        results = index.search(
            query,
            filter_dict={
                "category": "test",
                "price": [('>=', 100)],
                "created_at": [('>=', date(2024, 1, 1))],
            }
        )
        assert len(results) > 0
        assert all(
            doc["category"] == "test"
            and doc["price"] >= 100
            and doc["created_at"] >= date(2024, 1, 1)
            for doc in results
        )
