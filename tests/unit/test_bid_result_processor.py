"""
Tests for BidResultProcessor - handles bid result CREATE and UPDATE logic.
"""
import pytest
import pandas as pd
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.pipeline.processors.bid_result_processor import BidResultProcessor
from src.pipeline.processor_context import ProcessorContext
from src.utils.cache_resolver import safe_int, safe_float


def create_mock_context(standalone_data_df, config=None, new_classes=None, existing_classes_cache=None):
    """Create a mock ProcessorContext with standalone data."""
    mock_logger = MagicMock()

    if config is None:
        config = MagicMock()
        config.cache_dir = '/tmp/test_cache'
        config.bidding_schedules = {}
        config.start_ay_term = '2024-25_T1'

    # Auto-generate mock classes if not provided, so find_all_class_ids can find them
    if new_classes is None:
        new_classes = []
        if not standalone_data_df.empty:
            if 'acad_term_id' in standalone_data_df.columns and 'class_boss_id' in standalone_data_df.columns:
                unique_pairs = standalone_data_df[['acad_term_id', 'class_boss_id']].drop_duplicates().dropna()
                for idx, row in unique_pairs.iterrows():
                    acad_term_id = row['acad_term_id']
                    boss_id = row['class_boss_id']
                    if pd.notna(acad_term_id) and pd.notna(boss_id):
                        new_classes.append({
                            'id': len(new_classes) + 1,  # synthetic class ID
                            'acad_term_id': acad_term_id,
                            'boss_id': int(boss_id)
                        })

    context = ProcessorContext(
        logger=mock_logger,
        standalone_data=standalone_data_df,
        config=config,
        new_classes=new_classes,
        existing_classes_cache=existing_classes_cache or [],
        new_bid_result=[],
        update_bid_result=[],
        bid_window_cache={},
        expected_acad_term_id="AY20242026T1"
    )
    context.boss_stats = {}
    return context


class MockConfig:
    """Real config object to avoid MagicMock.get() issues."""
    def __init__(self):
        self.cache_dir = '/tmp/test_cache'
        self.bidding_schedules = {}
        self.start_ay_term = '2024-25_T1'

    def get(self, key, default=None):
        """Delegate to bidding_schedules dict's get method."""
        return self.bidding_schedules.get(key, default)


class TestBidResultProcessor:
    """Tests for BidResultProcessor._do_process()."""

    def test_do_process_filters_by_current_bidding_window(self):
        """Test _do_process() filters by current bidding window."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'median': 1500,
                'min': 500
            },
            {
                'acad_term_id': 'AY20242026T1',
                'bidding_window': 'Round 1A Window 1',  # Not current window
                'course_code': 'CS101',
                'section': 'B',
                'class_boss_id': 1,
                'total': 30,
                'current_enrolled': 20,
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 10,
                'median': 1200,
                'min': 400
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        # Should only process the Round 1 Window 1 record (current window)
        assert len(context.new_bid_result) == 1
        assert context.new_bid_result[0]['bid_window_id'] == 100

    def test_do_process_creates_new_records(self):
        """Test _do_process() creates new bid result records."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'median': 1500,
                'min': 500
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        assert len(context.new_bid_result) == 1
        result = context.new_bid_result[0]
        assert result['bid_window_id'] == 100
        assert result['vacancy'] == 30
        assert result['enrolled_students'] == 25
        assert result['median'] == 1500.0
        assert result['min'] == 500.0

    def test_do_process_determines_update_vs_create(self):
        """Test _do_process() CREATE vs UPDATE determination."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'median': 1500,
                'min': 500
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        # First run - should create
        assert len(context.new_bid_result) == 1
        assert len(context.update_bid_result) == 0

    def test_do_process_uses_safe_int_safe_float(self):
        """Test _do_process() uses safe_int() and safe_float()."""
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
                'total': '30',  # String that should be converted
                'current_enrolled': '25',
                'd_i_c_e': '1',
                'opening_vacancy': '30',
                'after_process_vacancy': '5',
                'median': '1500.50',
                'min': '500.25'
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        result = context.new_bid_result[0]
        # Verify safe_int and safe_float were used
        assert result['vacancy'] == 30
        assert result['enrolled_students'] == 25
        assert result['median'] == 1500.50
        assert result['min'] == 500.25

    def test_do_process_handles_na_values(self):
        """Test _do_process() handles NA values correctly."""
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
                'class_boss_id': pd.NA,  # Missing required field
                'total': 30,
                'current_enrolled': 25,
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        # Row with NA acad_term_id or class_boss_id should be skipped
        assert len(context.new_bid_result) == 0

    def test_do_process_filters_by_acad_term_id(self):
        """Test _do_process() also filters by acad_term_id to prevent cross-term contamination."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'median': 1500,
                'min': 500
            },
            {
                'acad_term_id': 'AY20252026T1',  # Different term - should be filtered out
                'bidding_window': 'Round 1 Window 1',
                'course_code': 'CS201',
                'section': 'A',
                'class_boss_id': 2,
                'total': 30,
                'current_enrolled': 20,
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 10,
                'median': 1200,
                'min': 400
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100,
            ('AY20252026T1', '1', 1): 101
        }
        context.expected_acad_term_id = 'AY20242026T1'

        processor = BidResultProcessor(context)
        processor._do_process()

        # Should only process AY20242026T1 records
        assert len(context.new_bid_result) == 1
        assert context.new_bid_result[0]['bid_window_id'] == 100


class TestBidResultProcessorMedianMinColumns:
    """Tests for median/min column name variations."""

    def test_do_process_handles_multiple_median_column_names(self):
        """Test _do_process() tries multiple column names for median."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'Median Bid': 1500,  # Alternative column name
                'min': 500
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        assert len(context.new_bid_result) == 1
        assert context.new_bid_result[0]['median'] == 1500.0

    def test_do_process_handles_multiple_min_column_names(self):
        """Test _do_process() tries multiple column names for min."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5,
                'median': 1500,
                'Min Bid': 500  # Alternative column name
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        assert len(context.new_bid_result) == 1
        assert context.new_bid_result[0]['min'] == 500.0


class TestBidResultProcessorBeforeProcessVacancy:
    """Tests for before_process_vacancy calculation."""

    def test_do_process_calculates_before_process_vacancy(self):
        """Test _do_process() calculates before_process_vacancy as total - enrolled."""
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
                'd_i_c_e': 1,
                'opening_vacancy': 30,
                'after_process_vacancy': 5
            }
        ])

        config = MockConfig()
        config.bidding_schedules = bidding_schedules
        config.start_ay_term = '2024-25_T1'

        context = create_mock_context(df, config=config)
        context.bid_window_cache = {
            ('AY20242026T1', '1', 1): 100
        }

        processor = BidResultProcessor(context)
        processor._do_process()

        result = context.new_bid_result[0]
        # before_process_vacancy = total - enrolled = 30 - 25 = 5
        assert result['before_process_vacancy'] == 5
