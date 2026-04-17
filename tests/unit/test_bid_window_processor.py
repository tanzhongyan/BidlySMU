"""
Tests for BidWindowProcessor - handles bid window CREATE logic.
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.pipeline.processors.bid_window_processor import BidWindowProcessor
from src.pipeline.processor_context import ProcessorContext


def create_mock_context(boss_data_df, bid_window_cache=None, bid_window_id_counter=1):
    """Create a mock ProcessorContext with boss data."""
    mock_logger = MagicMock()
    context = ProcessorContext(
        logger=mock_logger,
        boss_data=boss_data_df,
        bid_window_cache=bid_window_cache or {},
        new_bid_windows=[],
        bid_window_id_counter=bid_window_id_counter,
        expected_acad_term_id="AY20242026T1"
    )
    context.boss_stats = {}
    context.stats = {}
    return context


class TestBidWindowProcessor:
    """Tests for BidWindowProcessor._do_process()."""

    def test_do_process_discovers_windows_from_boss_data(self):
        """Test _do_process() discovers windows from boss_data."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 2',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS102',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should have created 2 unique windows (Round 1 Window 1 and Round 1 Window 2)
        assert len(context.new_bid_windows) == 2

    def test_do_process_parses_window_names_correctly(self):
        """Test that parse_window_name is called correctly for each row."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1A Window 2',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        assert len(context.new_bid_windows) == 1
        window = context.new_bid_windows[0]
        assert window['round'] == '1A'
        assert window['window'] == 2
        assert window['acad_term_id'] == 'AY20242026T1'

    def test_correct_new_bid_windows_structure(self):
        """Test correct new_bid_windows structure with all required fields."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        assert len(context.new_bid_windows) == 1
        window = context.new_bid_windows[0]

        # Verify structure
        assert 'id' in window
        assert 'acad_term_id' in window
        assert 'round' in window
        assert 'window' in window

        # Verify values
        assert window['acad_term_id'] == 'AY20242026T1'
        assert window['round'] == '1'
        assert window['window'] == 1

    def test_do_process_skips_existing_windows_in_cache(self):
        """Test _do_process() skips windows already in cache."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        # Pre-populate cache with this window
        existing_cache = {('AY20242026T1', '1', 1): 100}
        context = create_mock_context(df, bid_window_cache=existing_cache)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should not create since already in cache
        assert len(context.new_bid_windows) == 0

    def test_do_process_handles_multiple_terms(self):
        """Test _do_process() handles windows from multiple academic terms."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T2',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS201',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should have 2 windows from different terms
        assert len(context.new_bid_windows) == 2

        acad_term_ids = sorted([w['acad_term_id'] for w in context.new_bid_windows])
        assert acad_term_ids == ['AY20242026T1', 'AY20242026T2']

    def test_do_process_handles_empty_boss_data(self):
        """Test _do_process() handles empty boss_data gracefully."""
        df = pd.DataFrame(columns=['acad_term_id', 'bidding_window'])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should handle gracefully without errors
        assert len(context.new_bid_windows) == 0

    def test_do_process_handles_na_values(self):
        """Test _do_process() skips rows with NA values."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': pd.NA,
                'bidding_window': 'Round 1 Window 2',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': pd.NA,
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should only process valid row
        assert len(context.new_bid_windows) == 1

    def test_do_process_increments_counter(self):
        """Test _do_process() increments bid_window_id_counter correctly."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 2',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df, bid_window_id_counter=5)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Counter should be incremented by number of windows created
        assert context.bid_window_id_counter == 7  # Started at 5, created 2 windows

    def test_do_process_updates_bid_window_cache(self):
        """Test _do_process() updates bid_window_cache correctly."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Cache should be updated
        expected_key = ('AY20242026T1', '1', 1)
        assert expected_key in context.bid_window_cache
        assert context.bid_window_cache[expected_key] == 1


class TestBidWindowProcessorSorting:
    """Tests for window sorting within BidWindowProcessor."""

    def test_do_process_sorts_windows_by_round_order(self):
        """Test that windows are sorted by round order (1, 1A, 1B, 1C, 1F, 2, 2A)."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 2 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1A Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A'
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1B Window 1',
                'course_code': 'CS101',
                'section': 'A'
            }
        ])

        context = create_mock_context(df)
        processor = BidWindowProcessor(context)
        processor._do_process()

        # Should have 4 windows created
        assert len(context.new_bid_windows) == 4

        # Verify order: 1 -> 1A -> 1B -> 2
        rounds = [w['round'] for w in context.new_bid_windows]
        assert rounds == ['1', '1A', '1B', '2']
