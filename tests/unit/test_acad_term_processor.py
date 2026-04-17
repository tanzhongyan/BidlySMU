"""
Tests for AcadTermProcessor - handles academic term CREATE logic.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processor_context import ProcessorContext


def create_mock_context(standalone_data_df, acad_term_cache=None):
    """Create a mock ProcessorContext with standalone data."""
    mock_logger = MagicMock()
    context = ProcessorContext(
        logger=mock_logger,
        standalone_data=standalone_data_df,
        acad_term_cache=acad_term_cache or {},
        new_acad_terms=[],
        expected_acad_term_id="AY20242026T1"
    )
    context.boss_stats = {}
    return context


class TestAcadTermProcessor:
    """Tests for AcadTermProcessor._do_process()."""

    def test_do_process_groups_by_year_and_term(self):
        """Test _do_process() groups by (year_start, year_end, term)."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T2',
                'period_text': 'Semester 2',
                'start_dt': '2025-01-01',
                'end_dt': '2025-05-31',
                'acad_term_boss_id': 2
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        # Should create 2 unique acad_terms (T1 and T2)
        assert len(context.new_acad_terms) == 2

        # Verify the acad_term_ids
        acad_term_ids = [t['id'] for t in context.new_acad_terms]
        # Verify the acad_term_ids (format: AY{year_start}{year_end}{term})
        acad_term_ids = [t['id'] for t in context.new_acad_terms]
        assert 'AY20242026T1' in acad_term_ids
        assert 'AY20242026T2' in acad_term_ids

    def test_process_term_group_creates_correct_acad_term_id(self):
        """Test _process_term_group() creates correct acad_term_id format like 'AY20242026T1'."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 5
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        assert len(context.new_acad_terms) == 1
        term = context.new_acad_terms[0]

        # Verify ID format - AY{year_start}{year_end}{term}
        assert term['id'] == 'AY20242026T1'
        assert term['acad_year_start'] == 2024
        assert term['acad_year_end'] == 2026
        assert term['term'] == '1'  # T prefix removed

    def test_process_term_group_handles_t_prefix_removal(self):
        """Test that _process_term_group() removes T prefix when storing term."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2025,
                'acad_year_end': 2026,
                'term': 'T3',
                'period_text': 'Semester 3',
                'start_dt': '2025-06-01',
                'end_dt': '2025-07-31',
                'acad_term_boss_id': 3
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        term = context.new_acad_terms[0]
        # Format is AY{year_start}{year_end}{term}
        assert term['id'] == 'AY20252026T3'
        assert term['term'] == '3'  # T removed

    def test_extract_acad_term_from_path_parses_filename(self):
        """Test _extract_acad_term_from_path() parses filename."""
        processor = AcadTermProcessor(create_mock_context(pd.DataFrame()))

        # Test various path formats - note: path must have AY followed by 4 digits, 2 digits, T
        # For AY202426T1: year_start=2024, year_end=26, term=T1
        result1 = processor._extract_acad_term_from_path('/path/to/AY202426T1.csv')
        assert result1 == 'AY202426T1'

        result2 = processor._extract_acad_term_from_path('data/AY202526T2.xlsx')
        assert result2 == 'AY202526T2'

        result3 = processor._extract_acad_term_from_path('AY202325T1')
        assert result3 == 'AY202325T1'

    def test_extract_acad_term_from_path_returns_none_for_invalid(self):
        """Test _extract_acad_term_from_path() returns None for invalid paths."""
        processor = AcadTermProcessor(create_mock_context(pd.DataFrame()))

        result = processor._extract_acad_term_from_path('/path/to/invalid_name.csv')
        assert result is None

        result2 = processor._extract_acad_term_from_path('')
        assert result2 is None

    def test_do_process_skips_duplicates_in_cache(self):
        """Test that _do_process() skips terms already in cache."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            }
        ])

        existing_cache = {'AY20242026T1': {'id': 'AY20242026T1'}}
        context = create_mock_context(df, acad_term_cache=existing_cache)
        processor = AcadTermProcessor(context)
        processor._do_process()

        # Should not create since already in cache
        assert len(context.new_acad_terms) == 0

    def test_do_process_falls_back_to_source_file_when_data_missing(self):
        """Test fallback extraction from source_file path when row data is missing."""
        df = pd.DataFrame([
            {
                'acad_year_start': pd.NA,
                'acad_year_end': pd.NA,
                'term': pd.NA,
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1,
                'source_file': '/data/AY202426T1.csv'
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        # Should still create the term using fallback extraction from source_file
        # The source_file path has year_end=26 (2 digits), which is used as-is
        assert len(context.new_acad_terms) == 1
        assert context.new_acad_terms[0]['id'] == 'AY202426T1'

    def test_do_process_uses_most_common_period_text(self):
        """Test that _process_term_group() uses most common period_text for dates."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',  # Same period - more common
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Term 1',  # Less common
                'start_dt': '2025-01-01',  # Different dates
                'end_dt': '2025-05-31',
                'acad_term_boss_id': 1
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        term = context.new_acad_terms[0]
        # Should use dates from most common period_text ("Semester 1")
        assert term['start_dt'] == '2024-08-01'
        assert term['end_dt'] == '2024-12-31'


class TestAcadTermProcessorIntegration:
    """Integration tests for AcadTermProcessor with multiple term groups."""

    def test_do_process_handles_multiple_term_groups(self):
        """Test _do_process() correctly handles multiple term groups."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T2',
                'period_text': 'Semester 2',
                'start_dt': '2025-01-01',
                'end_dt': '2025-05-31',
                'acad_term_boss_id': 2
            },
            {
                'acad_year_start': 2025,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',
                'start_dt': '2025-08-01',
                'end_dt': '2025-12-31',
                'acad_term_boss_id': 3
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        processor._do_process()

        # Should create 3 unique terms
        assert len(context.new_acad_terms) == 3

        acad_term_ids = sorted([t['id'] for t in context.new_acad_terms])
        # year_start=2024,year_end=2026 -> AY20242026T1/T2; year_start=2025,year_end=2026 -> AY20252026T1
        assert acad_term_ids == ['AY20242026T1', 'AY20242026T2', 'AY20252026T1']
