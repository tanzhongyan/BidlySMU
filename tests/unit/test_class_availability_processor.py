"""
Unit tests for ClassAvailabilityProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from src.pipeline.processors.class_availability_processor import ClassAvailabilityProcessor
from src.pipeline.dtos.class_availability_dto import ClassAvailabilityDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO


class TestClassAvailabilityProcessor:
    """Tests for ClassAvailabilityProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={}
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_class_lookup(self):
        """Processor should initialize with empty class_lookup when not provided."""
        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={}
        )
        assert processor._class_lookup == {}

    def test_process_returns_list(self):
        """process() should return a list of ClassAvailabilityDTOs."""
        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, list)

    @patch('src.config.CURRENT_WINDOW_NAME', None)
    def test_process_returns_empty_when_no_window(self):
        """process() should return empty list when no current window."""
        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert result == []


class TestFilterToCurrentWindow:
    """Tests for _filter_to_current_window method."""

    def test_filters_by_window_name(self):
        """_filter_to_current_window should filter data by bidding_window."""
        raw_data = pd.DataFrame({
            'bidding_window': ['Round 1 Window 1', 'Round 1 Window 2', 'Round 1 Window 1'],
            'course_code': ['MGMT715', 'MGMT715', 'MGMT715']
        })

        processor = ClassAvailabilityProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={}
        )

        result = processor._filter_to_current_window('Round 1 Window 1')

        assert len(result) == 2

    def test_filters_by_term_when_specified(self):
        """_filter_to_current_window should also filter by acad_term_id if expected_acad_term_id is set."""
        raw_data = pd.DataFrame({
            'bidding_window': ['Round 1 Window 1', 'Round 1 Window 1'],
            'acad_term_id': ['AY202526T1', 'AY202526T2'],
            'course_code': ['MGMT715', 'MGMT715']
        })

        processor = ClassAvailabilityProcessor(
            raw_data=raw_data,
            class_lookup={},
            bid_window_lookup={},
            expected_acad_term_id='AY202526T1',
            logger=Mock()
        )

        result = processor._filter_to_current_window('Round 1 Window 1')

        assert len(result) == 1
        assert result.iloc[0]['acad_term_id'] == 'AY202526T1'


class TestFindAllClassIds:
    """Tests for _find_all_class_ids method."""

    def test_finds_class_ids_by_term_and_boss(self):
        """_find_all_class_ids should find classes by acad_term_id and boss_id."""
        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-uuid-2'),
            ('AY202526T1', 1002, 'prof1'): MagicMock(id='class-uuid-3'),
        }

        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup=class_lookup,
            bid_window_lookup={}
        )

        result = processor._find_all_class_ids('AY202526T1', 1001)

        assert len(result) == 2
        assert 'class-uuid-1' in result
        assert 'class-uuid-2' in result

    def test_returns_empty_when_no_match(self):
        """_find_all_class_ids should return empty list when no classes match."""
        class_lookup = {
            ('AY202526T1', 1001, None): MagicMock(id='class-uuid-1'),
        }

        processor = ClassAvailabilityProcessor(
            raw_data=pd.DataFrame(),
            class_lookup=class_lookup,
            bid_window_lookup={}
        )

        result = processor._find_all_class_ids('AY202526T2', 9999)

        assert result == []


class TestClassAvailabilityDTO:
    """Tests for ClassAvailabilityDTO."""

    def test_create_availability_dto(self):
        """ClassAvailabilityDTO should be creatable with all fields."""
        dto = ClassAvailabilityDTO(
            class_id='class-456',
            bid_window_id='bw-123',
            total=40,
            current_enrolled=25,
            reserved=5,
            available=10
        )

        assert dto.class_id == 'class-456'
        assert dto.bid_window_id == 'bw-123'
        assert dto.total == 40
        assert dto.current_enrolled == 25
        assert dto.reserved == 5
        assert dto.available == 10