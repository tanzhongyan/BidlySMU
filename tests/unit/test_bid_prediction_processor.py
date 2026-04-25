"""
Unit tests for BidPredictionProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from src.pipeline.processors.bid_prediction_processor import BidPredictionProcessor
from src.pipeline.dtos.bid_prediction_dto import BidPredictionDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO


class TestBidPredictionProcessor:
    """Tests for BidPredictionProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup={}
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_lookups(self):
        """Processor should initialize with empty lookups when not provided."""
        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame({'course_code': ['MGMT715']}),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup={}
        )
        assert processor._class_lookup == {}
        assert processor._bid_window_lookup == {}

    def test_process_returns_list(self):
        """process() should return a list of BidPredictionDTOs."""
        with patch('src.config.CURRENT_WINDOW_NAME', None):
            processor = BidPredictionProcessor(
                raw_data=pd.DataFrame(),
                class_lookup={},
                bid_window_lookup={},
                multiple_lookup={},
                bidding_schedule=[],
                logger=Mock()
            )
            result = processor.process()
            assert isinstance(result, list)

    @patch('src.config.CURRENT_WINDOW_NAME', None)
    def test_process_returns_empty_when_no_window(self):
        """process() should return empty list when no current window."""
        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup={},
            bidding_schedule=[],
            logger=Mock()
        )
        result = processor.process()
        assert result == []


class TestGetInstructorMap:
    """Tests for _get_instructor_map method."""

    def test_builds_instructor_map(self):
        """_get_instructor_map should build record_key -> [professor_names] mapping."""
        raw_data = pd.DataFrame({'record_key': ['key1', 'key2']})
        multiple_lookup = {
            'key1': [{'professor_name': 'John Smith'}, {'professor_name': 'Jane Doe'}],
            'key2': [{'professor_name': 'Bob Wilson'}]
        }

        processor = BidPredictionProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup=multiple_lookup
        )

        instructor_map = processor._get_instructor_map()

        assert 'key1' in instructor_map
        assert 'John Smith' in instructor_map['key1']
        assert 'Jane Doe' in instructor_map['key1']
        assert 'key2' in instructor_map
        assert 'Bob Wilson' in instructor_map['key2']

    def test_skips_missing_professor_names(self):
        """_get_instructor_map should skip rows without professor_name."""
        multiple_lookup = {
            'key1': [{'professor_name': 'John Smith'}, {'other_field': 'value'}]
        }

        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup=multiple_lookup
        )

        instructor_map = processor._get_instructor_map()

        assert len(instructor_map['key1']) == 1
        assert instructor_map['key1'][0] == 'John Smith'


class TestGetDayTimeMaps:
    """Tests for _get_day_time_maps method."""

    def test_extracts_days_and_times(self):
        """_get_day_time_maps should extract day_of_week and start_time maps."""
        multiple_lookup = {
            'key1': [
                {'type': 'CLASS', 'day_of_week': 'Monday', 'start_time': '09:00'},
                {'type': 'CLASS', 'day_of_week': 'Wednesday', 'start_time': '09:00'}
            ],
            'key2': [
                {'type': 'CLASS', 'day_of_week': 'Friday', 'start_time': '14:00'}
            ]
        }

        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup=multiple_lookup
        )

        day_map, time_map = processor._get_day_time_maps()

        assert 'key1' in day_map
        assert 'Monday' in day_map['key1'] and 'Wednesday' in day_map['key1']
        assert 'key1' in time_map
        assert time_map['key1'] == '09:00'

    def test_only_processes_class_type(self):
        """_get_day_time_maps should only process rows with type == 'CLASS'."""
        multiple_lookup = {
            'key1': [
                {'type': 'EXAM', 'day_of_week': 'Saturday', 'start_time': '10:00'},
                {'type': 'CLASS', 'day_of_week': 'Monday', 'start_time': '09:00'}
            ]
        }

        processor = BidPredictionProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup=multiple_lookup
        )

        day_map, time_map = processor._get_day_time_maps()

        assert 'key1' in day_map
        assert 'Monday' in day_map['key1']
        assert 'Saturday' not in day_map['key1']


class TestFilterToCurrentWindow:
    """Tests for _filter_to_current_window method."""

    def test_filters_by_window_name(self):
        """_filter_to_current_window should filter data by bidding_window."""
        raw_data = pd.DataFrame({
            'bidding_window': ['Round 1 Window 1', 'Round 1 Window 2', 'Round 1 Window 1'],
            'course_code': ['MGMT715', 'MGMT715', 'MGMT715']
        })

        processor = BidPredictionProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup={}
        )

        result = processor._filter_to_current_window('Round 1 Window 1')

        assert len(result) == 2


class TestEnrichBiddingData:
    """Tests for _enrich_bidding_data method."""

    def test_adds_before_process_vacancy(self):
        """_enrich_bidding_data should add before_process_vacancy column."""
        raw_data = pd.DataFrame({
            'bidding_window': ['Round 1 Window 1'],
            'record_key': ['key1'],
            'total': [40],
            'current_enrolled': [25]
        })

        processor = BidPredictionProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup={},
            logger=Mock()
        )

        result = processor._enrich_bidding_data(raw_data)

        assert 'before_process_vacancy' in result.columns
        assert result['before_process_vacancy'].iloc[0] == 15

    def test_adds_instructor_from_multiple_lookup(self):
        """_enrich_bidding_data should add instructor column from multiple_lookup."""
        raw_data = pd.DataFrame({
            'bidding_window': ['Round 1 Window 1'],
            'record_key': ['key1'],
            'total': [40],
            'current_enrolled': [25]
        })

        multiple_lookup = {
            'key1': [{'professor_name': 'John Smith'}]
        }

        processor = BidPredictionProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={},
            multiple_lookup=multiple_lookup,
            logger=Mock()
        )

        result = processor._enrich_bidding_data(raw_data)

        assert 'instructor' in result.columns
        assert 'John Smith' in result['instructor'].iloc[0]


