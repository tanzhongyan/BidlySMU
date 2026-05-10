"""
Tests for AcadTermProcessor - handles academic term CREATE logic.
Class-based processor with direct parameters.
"""
import pytest
import pandas as pd

from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.dtos.acad_term_dto import AcadTermDTO


class TestAcadTermProcessor:
    """Tests for AcadTermProcessor.process()."""

    def test_returns_tuple_of_new_and_updated(self):
        """Test process() returns tuple of (new_terms, updated_terms)."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 2
        assert len(updated_terms) == 0
        assert all(isinstance(r, AcadTermDTO) for r in new_terms)
        acad_term_ids = sorted([r.id for r in new_terms])
        assert acad_term_ids == ['AY202426T1', 'AY202426T2']

    def test_creates_correct_acad_term_id_format(self):
        """Test creates correct acad_term_id from row data."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 1
        term = new_terms[0]
        assert term.id == 'AY202426T1'
        assert term.acad_year_start == 2024
        assert term.acad_year_end == 2026
        assert term.term == '1'  # T prefix removed

    def test_handles_t_prefix_removal(self):
        """Test that T prefix is removed when storing term."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        term = new_terms[0]
        assert term.id == 'AY202526T3'
        assert term.term == '3'  # T removed

    def test_skips_rows_without_acad_term_id(self):
        """Test that rows without acad_term_id are skipped."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 0

    def test_skips_duplicates_in_cache(self):
        """Test that terms already in cache are skipped."""
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
        processor = AcadTermProcessor(df, existing_cache)
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 0

    def test_uses_row_start_dt_end_dt_directly(self):
        """Test that start_dt/end_dt are used directly from row."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        term = new_terms[0]
        assert term.start_dt == '2024-08-01'
        assert term.end_dt == '2024-12-31'

    def test_handles_multiple_term_groups(self):
        """Test correctly handles multiple term groups."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 3
        acad_term_ids = sorted([r.id for r in new_terms])
        assert acad_term_ids == ['AY202426T1', 'AY202426T2', 'AY202526T1']

    def test_handles_duplicate_rows_same_term(self):
        """Test that duplicate rows for same term are deduplicated."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 1
        assert new_terms[0].id == 'AY202426T1'

    def test_handles_null_boss_id(self):
        """Test that null boss_id is handled."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 1
        assert new_terms[0].boss_id is None

    def test_updated_terms_always_empty(self):
        """Test that updated_terms is always empty since acad_term only creates."""
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

        processor = AcadTermProcessor(df, {})
        new_terms, updated_terms = processor.process()

        assert len(new_terms) == 1
        assert len(updated_terms) == 0