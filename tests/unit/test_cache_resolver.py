"""
Unit tests for cache_resolver utilities.
"""
import math
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.utils.cache_resolver import (
    get_bid_predictor_cache_files,
    get_bid_predictor_queries,
    merge_bid_windows_with_new_csv,
    safe_int,
    safe_float,
    merge_bid_window_csv_into_cache,
)


class TestGetBidPredictorCacheFiles:
    """Tests for get_bid_predictor_cache_files()."""

    def test_returns_correct_dict_structure(self):
        """Should return dict with correct keys and Path values."""
        cache_dir = "/some/cache/dir"
        result = get_bid_predictor_cache_files(cache_dir)

        expected_keys = ['courses', 'classes', 'acad_terms', 'professors', 'bid_windows', 'bid_prediction']
        assert list(result.keys()) == expected_keys

        assert result['courses'] == Path("/some/cache/dir/courses_cache.pkl")
        assert result['classes'] == Path("/some/cache/dir/classes_cache.pkl")
        assert result['acad_terms'] == Path("/some/cache/dir/acad_term_cache.pkl")
        assert result['professors'] == Path("/some/cache/dir/professors_cache.pkl")
        assert result['bid_windows'] == Path("/some/cache/dir/bid_window_cache.pkl")
        assert result['bid_prediction'] == Path("/some/cache/dir/bid_prediction_cache.pkl")

    def test_returns_path_objects(self):
        """All values should be Path objects."""
        result = get_bid_predictor_cache_files("/cache")
        for val in result.values():
            assert isinstance(val, Path)


class TestGetBidPredictorQueries:
    """Tests for get_bid_predictor_queries()."""

    def test_returns_six_query_strings(self):
        """Should return exactly 6 query strings."""
        result = get_bid_predictor_queries()
        assert len(result) == 6

    def test_has_correct_keys(self):
        """Should have correct keys for all query types."""
        result = get_bid_predictor_queries()
        expected_keys = ['courses', 'classes', 'acad_terms', 'professors', 'bid_windows', 'bid_prediction']
        assert list(result.keys()) == expected_keys

    def test_all_queries_select_all(self):
        """All queries should be SELECT * FROM queries."""
        result = get_bid_predictor_queries()
        for query in result.values():
            assert query.startswith("SELECT * FROM ")


class TestSafeInt:
    """Tests for safe_int()."""

    def test_converts_valid_int(self):
        """Should convert valid integer."""
        assert safe_int(42) == 42

    def test_converts_valid_float_to_int(self):
        """Should convert float to int."""
        assert safe_int(42.9) == 42

    def test_converts_numeric_string(self):
        """Should convert numeric string to int."""
        assert safe_int("123") == 123

    def test_returns_none_for_nan(self):
        """Should return None for NaN."""
        assert safe_int(float('nan')) is None

    def test_returns_none_for_np_nan(self):
        """Should return None for numpy NaN."""
        assert safe_int(np.nan) is None

    def test_returns_none_for_none(self):
        """Should return None for None."""
        assert safe_int(None) is None

    def test_returns_none_for_pd_na(self):
        """Should return None for pandas NA."""
        assert safe_int(pd.NA) is None


class TestSafeFloat:
    """Tests for safe_float()."""

    def test_converts_valid_float(self):
        """Should convert valid float."""
        assert safe_float(42.5) == 42.5

    def test_converts_valid_int_to_float(self):
        """Should convert integer to float."""
        assert safe_float(42) == 42.0

    def test_converts_numeric_string(self):
        """Should convert numeric string to float."""
        assert safe_float("123.45") == 123.45

    def test_returns_none_for_nan(self):
        """Should return None for NaN."""
        assert safe_float(float('nan')) is None

    def test_returns_none_for_np_nan(self):
        """Should return None for numpy NaN."""
        assert safe_float(np.nan) is None

    def test_returns_none_for_none(self):
        """Should return None for None."""
        assert safe_float(None) is None

    def test_returns_none_for_pd_na(self):
        """Should return None for pandas NA."""
        assert safe_float(pd.NA) is None


class TestMergeBidWindowsWithNewCsv:
    """Tests for merge_bid_windows_with_new_csv()."""

    def setup_method(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_returns_empty_df_when_file_missing(self):
        """Should return empty DataFrame when CSV file does not exist."""
        result = merge_bid_windows_with_new_csv(
            existing_bid_windows_df=None,
            new_bid_window_path=os.path.join(self.temp_dir, "nonexistent.csv")
        )
        assert result.empty

    def test_returns_existing_df_when_file_missing(self):
        """Should return existing DataFrame unchanged when file missing."""
        existing_df = pd.DataFrame({'a': [1, 2]})
        result = merge_bid_windows_with_new_csv(
            existing_bid_windows_df=existing_df,
            new_bid_window_path=os.path.join(self.temp_dir, "nonexistent.csv")
        )
        assert result.equals(existing_df)

    def test_merges_new_csv_into_empty_existing(self):
        """Should return new CSV data when existing is None/empty."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        new_df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1'],
            'round': ['1'],
            'window': [1],
            'id': [100]
        })
        new_df.to_csv(csv_path, index=False)

        result = merge_bid_windows_with_new_csv(
            existing_bid_windows_df=None,
            new_bid_window_path=csv_path
        )
        assert len(result) == 1
        assert result.iloc[0]['acad_term_id'] == '2025-26_T1'

    def test_concatenates_with_existing_df(self):
        """Should concatenate new data with existing DataFrame."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        new_df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1'],
            'round': ['1'],
            'window': [1],
            'id': [100]
        })
        new_df.to_csv(csv_path, index=False)

        existing_df = pd.DataFrame({
            'acad_term_id': ['2024-25_T1'],
            'round': ['1'],
            'window': [1],
            'id': [99]
        })

        result = merge_bid_windows_with_new_csv(
            existing_bid_windows_df=existing_df,
            new_bid_window_path=csv_path
        )
        assert len(result) == 2

    def test_removes_duplicates_by_term_round_window(self):
        """Should remove duplicates keeping last."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        new_df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1', '2025-26_T1'],
            'round': ['1', '1'],
            'window': [1, 1],
            'id': [100, 101]
        })
        new_df.to_csv(csv_path, index=False)

        result = merge_bid_windows_with_new_csv(
            existing_bid_windows_df=None,
            new_bid_window_path=csv_path
        )
        assert len(result) == 1
        assert result.iloc[0]['id'] == 101  # Keep last


class TestMergeBidWindowCsvIntoCache:
    """Tests for merge_bid_window_csv_into_cache()."""

    def setup_method(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def test_returns_zero_when_file_missing(self):
        """Should return 0 when CSV file does not exist."""
        cache = {}
        result = merge_bid_window_csv_into_cache(
            bid_window_cache=cache,
            new_bid_window_path=os.path.join(self.temp_dir, "nonexistent.csv")
        )
        assert result == 0
        assert len(cache) == 0

    def test_adds_new_entries_to_empty_cache(self):
        """Should add new entries to empty cache."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1'],
            'round': ['1'],
            'window': [1],
            'id': [100]
        })
        df.to_csv(csv_path, index=False)

        cache = {}
        result = merge_bid_window_csv_into_cache(
            bid_window_cache=cache,
            new_bid_window_path=csv_path
        )
        assert result == 1
        assert (('2025-26_T1', '1', 1) in cache)
        assert cache[('2025-26_T1', '1', 1)] == 100

    def test_skips_existing_keys(self):
        """Should not overwrite existing cache entries."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1'],
            'round': ['1'],
            'window': [1],
            'id': [200]
        })
        df.to_csv(csv_path, index=False)

        cache = {('2025-26_T1', '1', 1): 100}  # Already exists
        result = merge_bid_window_csv_into_cache(
            bid_window_cache=cache,
            new_bid_window_path=csv_path
        )
        assert result == 0  # No new entries added
        assert cache[('2025-26_T1', '1', 1)] == 100  # Original preserved

    def test_adds_multiple_new_entries(self):
        """Should add multiple new entries."""
        csv_path = os.path.join(self.temp_dir, "new_bid_window.csv")
        df = pd.DataFrame({
            'acad_term_id': ['2025-26_T1', '2025-26_T1'],
            'round': ['1', '2'],
            'window': [1, 1],
            'id': [100, 101]
        })
        df.to_csv(csv_path, index=False)

        cache = {}
        result = merge_bid_window_csv_into_cache(
            bid_window_cache=cache,
            new_bid_window_path=csv_path
        )
        assert result == 2
        assert len(cache) == 2