"""
Tests for ClassAvailabilityProcessor - handles class availability CREATE logic.
"""
import pytest
import pandas as pd
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.pipeline.processors.class_availability_processor import ClassAvailabilityProcessor
from src.pipeline.processor_context import ProcessorContext


def create_mock_context(standalone_data_df, config=None, new_classes=None, existing_classes_cache=None):
    """Create a mock ProcessorContext with standalone data."""
    mock_logger = MagicMock()

    if config is None:
        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = {}
        config.start_ay_term = '2024-25_T1'

    context = ProcessorContext(
        logger=mock_logger,
        standalone_data=standalone_data_df,
        config=config,
        new_classes=new_classes or [],
        existing_classes_cache=existing_classes_cache or [],
        new_class_availability=[],
        bid_window_cache={},
        expected_acad_term_id="AY20242026T1"
    )
    context.boss_stats = {}
    context.stats = {}
    context.failed_mappings = []
    return context


class TestClassAvailabilityProcessor:
    """Tests for ClassAvailabilityProcessor._do_process()."""

    def test_do_process_creates_new_class_availability_records(self):
        """Test _do_process() creates new_class_availability records."""
        # Create bidding schedule: now is current, results in 7 days
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1'),
                (future_date + timedelta(days=7), 'Round 1A Window 1', 'r1a')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        assert len(context.new_class_availability) == 1
        record = context.new_class_availability[0]
        assert record['class_id'] == 500
        assert record['bid_window_id'] == 100
        assert record['total'] == 30
        assert record['current_enrolled'] == 25
        assert record['reserved'] == 5
        assert record['available'] == 10

    def test_do_process_filters_by_current_bidding_window(self):
        """Test _do_process() filters by current bidding window."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1'),
                (future_date + timedelta(days=7), 'Round 1A Window 1', 'r1a')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1A Window 1',  # Not current window
                'course_code': 'CS101',
                'section': 'B',
                'class_boss_id': 2,
                'total': 25,
                'current_enrolled': 20,
                'reserved': 3,
                'available': 8
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100,
            ('AY20242026T1', '1A', 1): 101
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1},
            {'id': 501, 'acad_term_id': 'AY20242026T1', 'boss_id': 2}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # Should only process Round 1 Window 1 (current window)
        assert len(context.new_class_availability) == 1
        assert context.new_class_availability[0]['class_id'] == 500

    def test_do_process_uses_get_current_live_window_name(self):
        """Test _do_process() uses get_current_live_window_name() from schedule_resolver."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)

        with patch('src.pipeline.processors.class_availability_processor.get_current_live_window_name') as mock_get_window:
            mock_get_window.return_value = 'Round 1 Window 1'
            processor._do_process()

            # Verify get_current_live_window_name was called
            mock_get_window.assert_called_once()

    def test_do_process_filters_by_acad_term_id(self):
        """Test _do_process() filters by acad_term_id to prevent cross-term contamination."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            },
            {
                'acad_term_id': 'AY20252026T1',  # Different term - should be filtered
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS201',
                'section': 'A',
                'class_boss_id': 2,
                'total': 25,
                'current_enrolled': 20,
                'reserved': 3,
                'available': 8
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100,
            ('AY20252026T1', '1', 1): 101
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1},
            {'id': 501, 'acad_term_id': 'AY20252026T1', 'boss_id': 2}
        ]
        context.expected_acad_term_id = 'AY20242026T1'

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # Should only process AY20242026T1 record
        assert len(context.new_class_availability) == 1
        assert context.new_class_availability[0]['class_id'] == 500

    def test_do_process_skips_rows_with_missing_class(self):
        """Test _do_process() skips rows where class is not found via find_all_class_ids."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 999,  # Non-existent class
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = []  # No matching class
        context.existing_classes_cache = []

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # Should skip and add to failed_mappings
        assert len(context.new_class_availability) == 0
        assert len(context.failed_mappings) == 1
        assert context.failed_mappings[0]['reason'] == 'class_not_found'

    def test_do_process_skips_existing_availability_in_cache(self):
        """Test _do_process() skips records already in cache (prevents duplicates)."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        # Mock existing cache file with existing record
        with patch.object(os.path, 'exists', return_value=False):
            processor = ClassAvailabilityProcessor(context)
            processor._do_process()

        # First run - should create
        assert len(context.new_class_availability) == 1

        # Second run - same class_availability should be skipped (in current_run_keys)
        context2 = create_mock_context(df, config=config)
        context2.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context2.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]
        # Pre-populate new_class_availability with existing record
        context2.new_class_availability = [
            {'class_id': 500, 'bid_window_id': 100}
        ]

        with patch.object(os.path, 'exists', return_value=False):
            processor2 = ClassAvailabilityProcessor(context2)
            processor2._do_process()

        # Should not add duplicate
        assert len(context2.new_class_availability) == 1

    def test_do_process_handles_na_values(self):
        """Test _do_process() skips rows with NA values."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': pd.NA,
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': pd.NA,  # Missing window
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # All rows have NA issues, should create nothing
        assert len(context.new_class_availability) == 0

    def test_do_process_handles_missing_bid_window_id(self):
        """Test _do_process() skips rows when bid_window_id not found in cache."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {}  # Empty cache - no bid_window_id found
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # Should skip due to no bid_window_id
        assert len(context.new_class_availability) == 0

    def test_do_process_default_values_for_availability_fields(self):
        """Test _do_process() uses 0 for missing availability numeric fields."""
        now = datetime.now()
        future_date = now + timedelta(days=7)

        bidding_schedules = {
            '2024-25_T1': [
                (future_date, 'Round 1 Window 1', 'r1')
            ]
        }

        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': pd.NA,  # Missing values should become 0
                'current_enrolled': pd.NA,
                'reserved': pd.NA,
                'available': pd.NA
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        assert len(context.new_class_availability) == 1
        record = context.new_class_availability[0]
        assert record['total'] == 0
        assert record['current_enrolled'] == 0
        assert record['reserved'] == 0
        assert record['available'] == 0


class TestClassAvailabilityProcessorNoWindow:
    """Tests for ClassAvailabilityProcessor when current window cannot be determined."""

    def test_do_process_handles_no_bidding_schedule(self):
        """Test _do_process() handles empty bidding schedule gracefully."""
        df = pd.DataFrame([
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS101',
                'section': 'A',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 25,
                'reserved': 5,
                'available': 10
            }
        ])

        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = {}  # Empty schedule
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }
        context.new_classes = [
            {'id': 500, 'acad_term_id': 'AY20242026T1', 'boss_id': 1}
        ]

        processor = ClassAvailabilityProcessor(context)
        processor._do_process()

        # Should process all data when window cannot be determined
        assert len(context.new_class_availability) == 1
