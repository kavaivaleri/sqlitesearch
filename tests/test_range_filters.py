"""
Tests for numeric and date range filters in TextSearchIndex.
"""

import os
import tempfile
from datetime import date, datetime

import pytest

from sqlitesearch import TextSearchIndex


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
def sample_docs_with_numeric():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language. It's easy to learn.",
            "price": 100,
            "rating": 4.5,
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms.",
            "price": 200,
            "rating": 3.8,
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality. Use pytest for Python.",
            "price": 150,
            "rating": 4.2,
        },
        {
            "question": "What is data science?",
            "text": "Data science involves statistics and programming.",
            "price": 50,
            "rating": 4.8,
        },
    ]


@pytest.fixture
def sample_docs_with_dates():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "created_at": date(2024, 1, 15),
        },
        {
            "question": "What is machine learning in Python?",
            "text": "Machine learning is a subset of AI.",
            "created_at": date(2024, 2, 20),
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "created_at": date(2024, 3, 10),
        },
        {
            "question": "What is data science?",
            "text": "Data science involves statistics.",
            "created_at": date(2024, 1, 5),
        },
    ]


@pytest.fixture
def sample_docs_with_datetime():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "created_at": datetime(2024, 1, 15, 10, 30),
        },
        {
            "question": "What is machine learning in Python?",
            "text": "Machine learning is a subset of AI.",
            "created_at": datetime(2024, 2, 20, 14, 45),
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "created_at": datetime(2024, 3, 10, 8, 0),
        },
    ]


class TestNumericFilters:
    """Tests for numeric field range filters."""

    def test_numeric_field_initialization(self, temp_db):
        """Test that index initializes correctly with numeric fields."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price", "rating"],
            db_path=temp_db
        )
        assert index.text_fields == ["question", "text"]
        assert index.numeric_fields == ["price", "rating"]

    def test_numeric_field_fit(self, sample_docs_with_numeric, temp_db):
        """Test that fit works correctly with numeric fields."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price", "rating"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)
        # Verify by searching
        results = index.search("python")
        assert len(results) > 0

    def test_numeric_filter_greater_than_or_equal(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with >= operator."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # price >= 100
        results = index.search("python", filter_dict={"price": [('>=', 100)]})
        assert len(results) == 2
        assert all(doc["price"] >= 100 for doc in results)

    def test_numeric_filter_less_than(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with < operator."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # price < 150
        results = index.search("python", filter_dict={"price": [('<', 150)]})
        assert len(results) == 1
        assert results[0]["price"] == 100

    def test_numeric_filter_range(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with range (AND of two conditions)."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # 100 < price < 200
        results = index.search("python", filter_dict={"price": [('>', 100), ('<', 200)]})
        assert len(results) == 1
        assert results[0]["price"] == 150

    def test_numeric_filter_inclusive_range(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with inclusive range."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # Use a query that matches multiple docs
        # "How" appears in 3 docs
        results = index.search("How", filter_dict={"price": [('>=', 100), ('<=', 200)]})
        assert len(results) == 2
        assert all(100 <= doc["price"] <= 200 for doc in results)

    def test_numeric_filter_exact_match(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with exact match."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        results = index.search("python", filter_dict={"price": 100})
        assert len(results) == 1
        assert results[0]["price"] == 100

    def test_numeric_filter_none(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with None value."""
        # Add a doc with None price
        docs = sample_docs_with_numeric + [
            {"question": "Free course", "text": "Free learning", "price": None}
        ]
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(docs)

        results = index.search("learning", filter_dict={"price": None})
        assert len(results) == 1
        assert results[0]["price"] is None

    def test_numeric_filter_multiple_operators(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with multiple operators."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["rating"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # rating >= 4.0 AND rating < 4.5
        results = index.search("python", filter_dict={"rating": [('>=', 4.0), ('<', 4.5)]})
        assert len(results) == 1
        assert results[0]["rating"] == 4.2

    def test_numeric_filter_not_equal(self, sample_docs_with_numeric, temp_db):
        """Test numeric filter with != operator."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_numeric)

        # "python" matches docs with prices 100 and 150. Filter out 100, get 1 result.
        results = index.search("python", filter_dict={"price": [('!=', 100)]})
        assert len(results) == 1
        assert all(doc["price"] != 100 for doc in results)


class TestDateFilters:
    """Tests for date field range filters."""

    def test_date_field_initialization(self, temp_db):
        """Test that index initializes correctly with date fields."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        assert index.text_fields == ["question", "text"]
        assert index.date_fields == ["created_at"]

    def test_date_field_fit(self, sample_docs_with_dates, temp_db):
        """Test that fit works correctly with date fields."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_dates)
        results = index.search("python")
        assert len(results) > 0

    def test_date_filter_after(self, sample_docs_with_dates, temp_db):
        """Test date filter with >= operator."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_dates)

        # created_at >= date(2024, 2, 1)
        results = index.search("python", filter_dict={"created_at": [('>=', date(2024, 2, 1))]})
        assert len(results) == 2
        assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)

    def test_date_filter_before(self, sample_docs_with_dates, temp_db):
        """Test date filter with < operator."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_dates)

        # created_at < date(2024, 2, 1)
        # Only the Jan 15 doc has "python" AND is before Feb 1
        results = index.search("python", filter_dict={"created_at": [('<', date(2024, 2, 1))]})
        assert len(results) == 1
        assert all(doc["created_at"] < date(2024, 2, 1) for doc in results)

    def test_date_filter_range(self, sample_docs_with_dates, temp_db):
        """Test date filter with range."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_dates)

        # date(2024, 1, 10) <= created_at < date(2024, 3, 1)
        results = index.search(
            "python",
            filter_dict={"created_at": [('>=', date(2024, 1, 10)), ('<', date(2024, 3, 1))]}
        )
        assert len(results) == 2
        assert all(date(2024, 1, 10) <= doc["created_at"] < date(2024, 3, 1) for doc in results)

    def test_date_filter_exact_match(self, sample_docs_with_dates, temp_db):
        """Test date filter with exact match."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_dates)

        results = index.search("python", filter_dict={"created_at": date(2024, 1, 15)})
        assert len(results) == 1
        assert results[0]["created_at"] == date(2024, 1, 15)

    def test_date_filter_with_datetime(self, sample_docs_with_datetime, temp_db):
        """Test date filter with datetime objects."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_with_datetime)

        # created_at >= datetime(2024, 2, 1, 0, 0)
        results = index.search(
            "python",
            filter_dict={"created_at": [('>=', datetime(2024, 2, 1, 0, 0))]}
        )
        assert len(results) == 2

    def test_date_filter_none(self, temp_db):
        """Test date filter with None value."""
        docs = [
            {"question": "Python", "text": "Python code", "created_at": date(2024, 1, 15)},
            {"question": "No date", "text": "No date set", "created_at": None},
        ]
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=[],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(docs)

        # Use a query that matches the doc with None date
        results = index.search("date", filter_dict={"created_at": None})
        assert len(results) == 1
        assert results[0]["created_at"] is None


class TestCombinedFilters:
    """Tests for combining multiple filter types."""

    @pytest.fixture
    def sample_docs_combined(self):
        return [
            {
                "question": "Python course",
                "text": "Learn Python programming",
                "category": "programming",
                "price": 100,
                "created_at": date(2024, 1, 15),
            },
            {
                "question": "ML course",
                "text": "Learn machine learning",
                "category": "ai",
                "price": 200,
                "created_at": date(2024, 2, 20),
            },
            {
                "question": "Testing course",
                "text": "Learn testing with pytest",
                "category": "programming",
                "price": 150,
                "created_at": date(2024, 3, 10),
            },
            {
                "question": "Data science course",
                "text": "Learn data science",
                "category": "data",
                "price": 50,
                "created_at": date(2024, 1, 5),
            },
        ]

    def test_keyword_and_numeric_filter(self, sample_docs_combined, temp_db):
        """Test combining keyword and numeric filters."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["category"],
            numeric_fields=["price"],
            db_path=temp_db
        )
        index.fit(sample_docs_combined)

        # category = "programming" AND price >= 100
        results = index.search(
            "course",
            filter_dict={"category": "programming", "price": [('>=', 100)]}
        )
        assert len(results) == 2
        assert all(doc["category"] == "programming" for doc in results)
        assert all(doc["price"] >= 100 for doc in results)

    def test_all_filter_types(self, sample_docs_combined, temp_db):
        """Test combining keyword, numeric, and date filters."""
        index = TextSearchIndex(
            text_fields=["question", "text"],
            keyword_fields=["category"],
            numeric_fields=["price"],
            date_fields=["created_at"],
            db_path=temp_db
        )
        index.fit(sample_docs_combined)

        # category = "programming" AND price >= 100 AND created_at >= date(2024, 2, 1)
        results = index.search(
            "course",
            filter_dict={
                "category": "programming",
                "price": [('>=', 100)],
                "created_at": [('>=', date(2024, 2, 1))]
            }
        )
        assert len(results) == 1
        assert results[0]["category"] == "programming"
        assert results[0]["price"] >= 100
        assert results[0]["created_at"] >= date(2024, 2, 1)
