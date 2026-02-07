"""
Operators for range filtering in sqlitesearch.

This module provides operator functions used for numeric and date range filtering.
"""

# Operator mapping for range filters
# Each operator is a lambda that takes two values (a, b) and returns a boolean
OPERATORS = {
    '>=': lambda a, b: a >= b,
    '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b,
    '<': lambda a, b: a < b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
}


def is_range_filter(value: object) -> bool:
    """
    Check if a value is a range filter (list of tuples).

    Args:
        value: The value to check.

    Returns:
        True if value is a list of (operator, value) tuples.
    """
    return (
        isinstance(value, list)
        and all(isinstance(v, tuple) and len(v) == 2 for v in value)
    )
