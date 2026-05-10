"""
Unit tests for BidWindowProcessor (refactored DTO pattern).
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.pipeline.processors.bid_window_processor import BidWindowProcessor
from src.pipeline.dtos.bid_window_dto import BidWindowDTO


class TestBidWindowDTO:
    """Tests for BidWindowDTO."""

    def test_from_dict(self):
        """from_dict should create BidWindowDTO from a dictionary."""
        item = {'id': 5, 'acad_term_id': 'AY202526T3A', 'round': '2', 'window': 1}
        dto = BidWindowDTO.from_dict(item)
        assert dto.id == 5
        assert dto.acad_term_id == 'AY202526T3A'
        assert dto.round == '2'
        assert dto.window == 1

    def test_from_dict_coerces_types(self):
        """from_dict should coerce types to handle inconsistent cache data."""
        item = {'id': '5', 'acad_term_id': 'AY202526T3A', 'round': 2, 'window': '1'}
        dto = BidWindowDTO.from_dict(item)
        assert isinstance(dto.id, int)
        assert dto.id == 5
        assert isinstance(dto.round, str)
        assert dto.round == '2'
        assert isinstance(dto.window, int)
        assert dto.window == 1

    def test_to_csv_row(self):
        dto = BidWindowDTO(
            id=1,
            acad_term_id='AY202526T1',
            round='1',
            window=1
        )

        row = dto.to_csv_row()
        assert row['id'] == 1
        assert row['acad_term_id'] == 'AY202526T1'
        assert row['round'] == '1'
        assert row['window'] == 1

    def test_to_db_row(self):
        dto = BidWindowDTO(
            id=100,
            acad_term_id='AY202526T3A',
            round='1A',
            window=2
        )

        row = dto.to_db_row()
        assert row['id'] == 100
        assert row['acad_term_id'] == 'AY202526T3A'
        assert row['round'] == '1A'
        assert row['window'] == 2

    def test_round_order_constant(self):
        """Test ROUND_ORDER class constant has correct values."""
        expected = {'1': 1, '1A': 2, '1B': 3, '1C': 4, '1F': 5, '2': 6, '2A': 7}
        assert BidWindowDTO.ROUND_ORDER == expected


class TestBidWindowProcessor:
    """Tests for BidWindowProcessor.process()."""

    def test_creates_new_bid_windows_for_unseen_combinations(self):
        """Test creates new windows for combinations not in cache."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202526T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'G1'
            },
            {
                'acad_term_id': 'AY202526T1',
                'bidding_window': 'Round 1 Window 2',
                'course_code': 'CS101',
                'section': 'G1'
            },
            {
                'acad_term_id': 'AY202526T1',
                'bidding_window': 'Round 1 Window 1',  # Duplicate
                'course_code': 'CS102',
                'section': 'G1'
            }
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 2  # Deduplicated
        assert len(updated_windows) == 0

        # Check DTOs have correct fields
        rounds = sorted([w.round for w in new_windows])
        assert rounds == ['1', '1']

    def test_skips_existing_windows_in_cache(self):
        """Test skips windows already in cache."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY202526T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'G1'
            }
        ])

        # Pre-populate cache with this window
        existing_cache = {('AY202526T1', '1', 1): 100}
        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache=existing_cache,
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 0
        assert len(updated_windows) == 0

    def test_uses_round_order_sorting(self):
        """Test windows are sorted by round order (1, 1A, 1B, 1C, 1F, 2, 2A)."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 2 Window 1', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1A Window 1', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1B Window 1', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 4

        # Verify order: 1 -> 1A -> 1B -> 2
        rounds = [w.round for w in new_windows]
        assert rounds == ['1', '1A', '1B', '2']

    def test_handles_multiple_terms(self):
        """Test handles windows from multiple academic terms."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': 'AY202526T2', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS201', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 2

        acad_term_ids = sorted([w.acad_term_id for w in new_windows])
        assert acad_term_ids == ['AY202526T1', 'AY202526T2']

    def test_returns_tuple_of_new_and_updated(self):
        """Test process() returns tuple of (new, updated)."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        result = processor.process()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # new
        assert isinstance(result[1], list)  # updated

    def test_handles_empty_dataframe(self):
        """Test handles empty DataFrame gracefully."""
        df = pd.DataFrame(columns=['acad_term_id', 'bidding_window'])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 0
        assert len(updated_windows) == 0

    def test_handles_na_values(self):
        """Test skips rows with NA values."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': pd.NA, 'bidding_window': 'Round 1 Window 2', 'course_code': 'CS101', 'section': 'G1'},
            {'acad_term_id': 'AY202526T1', 'bidding_window': pd.NA, 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        # Should only process valid row
        assert len(new_windows) == 1
        assert new_windows[0].acad_term_id == 'AY202526T1'

    def test_increments_id_from_cache_max(self):
        """Test ID counter starts from max ID in cache + 1."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 3', 'course_code': 'CS101', 'section': 'G1'}
        ])

        # Pre-populate cache with different windows (not the one we're creating)
        # This simulates existing windows with max ID = 100
        existing_cache = {
            ('AY202526T1', '1', 1): 50,
            ('AY202526T1', '1', 2): 100  # Max ID is 100
        }
        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache=existing_cache,
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        # Should create new window with ID = max(50, 100) + 1 = 101
        assert len(new_windows) == 1
        assert new_windows[0].id == 101

    def test_dto_to_csv_row_with_all_fields(self):
        """Test BidWindowDTO.to_csv_row() includes all fields."""
        dto = BidWindowDTO(
            id=42,
            acad_term_id='AY202526T3A',
            round='1C',
            window=3
        )

        row = dto.to_csv_row()

        assert row['id'] == 42
        assert row['acad_term_id'] == 'AY202526T3A'
        assert row['round'] == '1C'
        assert row['window'] == 3

    def test_parses_window_names_correctly(self):
        """Test parse_bidding_window is called correctly for each row."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1A Window 2', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        assert len(new_windows) == 1
        window = new_windows[0]
        assert window.round == '1A'
        assert window.window == 2
        assert window.acad_term_id == 'AY202526T1'

    def test_cache_updated_after_processing(self):
        """Test that bid_window_cache is updated with new entries."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Round 1 Window 1', 'course_code': 'CS101', 'section': 'G1'}
        ])

        cache = {}
        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache=cache,
            logger=MagicMock()
        )

        new_windows, updated_windows = processor.process()

        # Cache should be updated with dict entry (not just an integer ID)
        expected_key = ('AY202526T1', '1', 1)
        assert expected_key in cache
        assert isinstance(cache[expected_key], dict)
        assert cache[expected_key]['id'] == 1
        assert cache[expected_key]['acad_term_id'] == 'AY202526T1'
        assert cache[expected_key]['round'] == '1'
        assert cache[expected_key]['window'] == 1


class TestBidWindowParseBiddingWindowIntegration:
    """Tests verifying parse_bidding_window is called correctly for all window formats."""

    def test_incoming_freshmen_window(self):
        """'Incoming Freshmen Rnd 1 Win 4' should parse to round='1F', window=4."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Incoming Freshmen Rnd 1 Win 4', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, _ = processor.process()

        assert len(new_windows) == 1
        assert new_windows[0].round == '1F'
        assert new_windows[0].window == 4

    def test_incoming_exchange_window(self):
        """'Incoming Exchange Rnd 1C Win 1' should parse correctly."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Incoming Exchange Rnd 1C Win 1', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, _ = processor.process()

        assert len(new_windows) == 1
        assert new_windows[0].round == '1C'
        assert new_windows[0].window == 1

    def test_abbreviated_window_format(self):
        """'Rnd 1A Win 2' should parse to round='1A', window=2."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'Rnd 1A Win 2', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, _ = processor.process()

        assert len(new_windows) == 1
        assert new_windows[0].round == '1A'
        assert new_windows[0].window == 2

    def test_unparseable_window_skipped(self):
        """Unparseable bidding_window text should result in no new window."""
        df = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'bidding_window': 'xyz', 'course_code': 'CS101', 'section': 'G1'}
        ])

        processor = BidWindowProcessor(
            raw_data=df,
            bid_window_cache={},
            logger=MagicMock()
        )

        new_windows, _ = processor.process()

        assert len(new_windows) == 0