"""
Tests for AcadTermProcessor - handles academic term CREATE logic.
Refactored to check returned List[AcadTermDTO] instead of context mutation.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processor_context import ProcessorContext
from src.pipeline.dtos.acad_term_dto import AcadTermDTO


def create_mock_context(standalone_data_df, acad_term_cache=None):
    """Create a mock ProcessorContext with standalone data."""
    mock_logger = MagicMock()
    context = ProcessorContext(
        logger=mock_logger,
        standalone_data=standalone_data_df,
        acad_term_cache=acad_term_cache or {},
        new_acad_terms=[],
        expected_acad_term_id="AY202426T1"
    )
    context.boss_stats = {}
    return context


class TestAcadTermProcessor:
    """Tests for AcadTermProcessor._do_process()."""

    def test_do_process_returns_list_of_acad_term_dto(self):
        """Test _do_process() returns List[AcadTermDTO] for rows with acad_term_id."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_term_id': 'AY202426T2',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T2',
                'start_dt': '2025-01-01',
                'end_dt': '2025-05-31',
                'acad_term_boss_id': 2
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        # Should return 2 DTOs
        assert len(results) == 2
        assert all(isinstance(r, AcadTermDTO) for r in results)

        # Verify IDs
        acad_term_ids = sorted([r.id for r in results])
        assert acad_term_ids == ['AY202426T1', 'AY202426T2']

    def test_do_process_creates_correct_acad_term_id_format(self):
        """Test _do_process() creates correct acad_term_id from row data."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 5
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        assert len(results) == 1
        term = results[0]

        # Verify DTO fields
        assert term.id == 'AY202426T1'
        assert term.acad_year_start == 2024
        assert term.acad_year_end == 2026
        assert term.term == '1'  # T prefix removed

    def test_do_process_handles_t_prefix_removal(self):
        """Test that _do_process() removes T prefix when storing term."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202526T3',
                'acad_year_start': 2025,
                'acad_year_end': 2026,
                'term': 'T3',
                'start_dt': '2025-06-01',
                'end_dt': '2025-07-31',
                'acad_term_boss_id': 3
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        term = results[0]
        assert term.id == 'AY202526T3'
        assert term.term == '3'  # T removed

    def test_do_process_skips_rows_without_acad_term_id(self):
        """Test that _do_process() skips rows without acad_term_id."""
        df = pd.DataFrame([
            {
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
                # acad_term_id is MISSING
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        # Should return empty - no acad_term_id
        assert len(results) == 0

    def test_do_process_skips_duplicates_in_cache(self):
        """Test that _do_process() skips terms already in cache."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            }
        ])

        existing_cache = {'AY202426T1': {'id': 'AY202426T1'}}
        context = create_mock_context(df, acad_term_cache=existing_cache)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        # Should not return since already in cache
        assert len(results) == 0

    def test_do_process_uses_row_start_dt_end_dt_directly(self):
        """Test that _do_process() uses start_dt/end_dt directly from row."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'period_text': 'Semester 1',  # Not used
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        term = results[0]
        # Uses row's start_dt/end_dt directly (no period_text counting)
        assert term.start_dt == '2024-08-01'
        assert term.end_dt == '2024-12-31'

    def test_do_process_handles_multiple_term_groups(self):
        """Test _do_process() correctly handles multiple term groups."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_term_id': 'AY202426T2',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T2',
                'start_dt': '2025-01-01',
                'end_dt': '2025-05-31',
                'acad_term_boss_id': 2
            },
            {
                'acad_term_id': 'AY202526T1',
                'acad_year_start': 2025,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2025-08-01',
                'end_dt': '2025-12-31',
                'acad_term_boss_id': 3
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        # Should create 3 unique terms
        assert len(results) == 3

        acad_term_ids = sorted([r.id for r in results])
        assert acad_term_ids == ['AY202426T1', 'AY202426T2', 'AY202526T1']

    def test_do_process_handles_duplicate_rows_same_term(self):
        """Test that _do_process() deduplicates duplicate rows for same term."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            },
            {
                'acad_term_id': 'AY202426T1',  # Same term
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': 1
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        # Should create only 1 term (duplicates are deduplicated by iterating)
        assert len(results) == 1
        assert results[0].id == 'AY202426T1'

    def test_do_process_handles_null_boss_id(self):
        """Test that _do_process() handles null boss_id."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202426T1',
                'acad_year_start': 2024,
                'acad_year_end': 2026,
                'term': 'T1',
                'start_dt': '2024-08-01',
                'end_dt': '2024-12-31',
                'acad_term_boss_id': None  # Null boss_id
            }
        ])

        context = create_mock_context(df)
        processor = AcadTermProcessor(context)
        results = processor._do_process()

        assert len(results) == 1
        assert results[0].boss_id is None
