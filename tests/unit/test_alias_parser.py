"""
Unit tests for alias_parser utilities.
"""
import numpy as np
import pytest

from src.utils.alias_parser import parse_boss_aliases


class TestParseBossAliases:
    """Tests for parse_boss_aliases()."""

    # ----- None and empty tests -----

    def test_returns_empty_list_for_none(self):
        """Should return empty list for None input."""
        result = parse_boss_aliases(None)
        assert result == []

    def test_returns_empty_list_for_empty_list(self):
        """Should return empty list for empty list input."""
        result = parse_boss_aliases([])
        assert result == []

    def test_returns_empty_list_for_empty_string(self):
        """Should return empty list for empty string input."""
        result = parse_boss_aliases("")
        assert result == []

    def test_returns_empty_list_for_whitespace_only(self):
        """Should return empty list for whitespace-only string."""
        result = parse_boss_aliases("   ")
        assert result == []

    def test_returns_empty_list_for_nan(self):
        """Should return empty list for NaN input."""
        result = parse_boss_aliases(float('nan'))
        assert result == []

    def test_returns_empty_list_for_np_nan(self):
        """Should return empty list for numpy NaN input."""
        result = parse_boss_aliases(np.nan)
        assert result == []

    # ----- Standard Python list tests -----

    def test_returns_cleaned_list_from_python_list(self):
        """Should return cleaned list from standard Python list."""
        result = parse_boss_aliases(["Alice", "Bob", "Charlie"])
        assert result == ["Alice", "Bob", "Charlie"]

    def test_strips_whitespace_from_list_items(self):
        """Should strip whitespace from list items."""
        result = parse_boss_aliases(["  Alice  ", "  Bob", "Charlie  "])
        assert result == ["Alice", "Bob", "Charlie"]

    def test_filters_empty_strings_from_list(self):
        """Should filter out empty strings from list."""
        result = parse_boss_aliases(["Alice", "", "  ", "Bob"])
        assert result == ["Alice", "Bob"]

    def test_converts_non_string_items_to_strings_in_list(self):
        """Should convert non-string items to strings in list."""
        result = parse_boss_aliases(["Alice", 123, "Bob"])
        assert result == ["Alice", "123", "Bob"]

    # ----- NumPy array tests -----

    def test_returns_list_from_numpy_array(self):
        """Should return list from numpy array."""
        result = parse_boss_aliases(np.array(["Alice", "Bob"]))
        assert result == ["Alice", "Bob"]

    def test_strips_whitespace_from_numpy_array_items(self):
        """Should strip whitespace from numpy array items."""
        result = parse_boss_aliases(np.array(["  Alice  ", "  Bob"]))
        assert result == ["Alice", "Bob"]

    def test_filters_empty_strings_from_numpy_array(self):
        """Should filter empty strings from numpy array."""
        result = parse_boss_aliases(np.array(["Alice", "", "  "]))
        assert result == ["Alice"]

    def test_converts_numeric_numpy_array_to_strings(self):
        """Should convert numeric numpy array to strings."""
        result = parse_boss_aliases(np.array([1, 2, 3]))
        assert result == ["1", "2", "3"]

    # ----- PostgreSQL array string tests -----

    def test_parses_postgresql_array_string(self):
        """Should parse PostgreSQL array format '{"item1","item2"}'."""
        result = parse_boss_aliases('{"Alice","Bob"}')
        assert result == ["Alice", "Bob"]

    def test_parses_postgresql_array_string_with_spaces(self):
        """Should parse PostgreSQL array - items have whitespace stripped."""
        result = parse_boss_aliases('{" Alice "," Bob "}')
        # Whitespace should be stripped from each item
        assert result == ["Alice", "Bob"]

    def test_parses_single_item_postgresql_array(self):
        """Should parse single item PostgreSQL array."""
        result = parse_boss_aliases('{"Alice"}')
        assert result == ["Alice"]

    def test_parses_postgresql_array_with_empty_items(self):
        """PostgreSQL array items with content are parsed."""
        result = parse_boss_aliases('{"Alice","Bob"}')
        assert result == ["Alice", "Bob"]

    # ----- JSON array string tests -----

    def test_parses_json_array_string(self):
        """Should parse JSON array format '["item1", "item2"]'."""
        result = parse_boss_aliases('["Alice", "Bob"]')
        assert result == ["Alice", "Bob"]

    def test_parses_json_array_string_with_spaces(self):
        """Should parse JSON array with extra spaces."""
        result = parse_boss_aliases('[  "Alice"  ,  "Bob"  ]')
        assert result == ["Alice", "Bob"]

    def test_filters_empty_strings_from_json_array(self):
        """Should filter empty strings from JSON array."""
        result = parse_boss_aliases('["Alice", "", "  "]')
        assert result == ["Alice"]

    def test_handles_malformed_json_as_plain_string(self):
        """Should fall back to plain string if JSON is malformed."""
        result = parse_boss_aliases('["Alice", "Bob]')  # Missing closing quote
        assert result == ['["Alice", "Bob]']

    # ----- Plain string tests -----

    def test_returns_single_item_list_for_plain_string(self):
        """Should return single-item list for plain string."""
        result = parse_boss_aliases("Alice")
        assert result == ["Alice"]

    def test_strips_whitespace_from_plain_string(self):
        """Should strip whitespace from plain string."""
        result = parse_boss_aliases("  Alice  ")
        assert result == ["Alice"]

    # ----- Other iterable types -----

    def test_handles_tuple_input(self):
        """Should handle tuple input."""
        result = parse_boss_aliases(("Alice", "Bob"))
        assert result == ["Alice", "Bob"]

    def test_handles_set_input(self):
        """Should handle set input (order may vary)."""
        result = parse_boss_aliases({"Alice", "Bob"})
        assert set(result) == {"Alice", "Bob"}