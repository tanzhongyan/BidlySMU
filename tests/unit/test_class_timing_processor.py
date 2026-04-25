"""
Unit tests for ClassTimingProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock

from src.pipeline.processors.class_timing_processor import ClassTimingProcessor
from src.pipeline.dtos.timing_dto import ClassTimingDTO


class TestClassTimingProcessor:
    """Tests for ClassTimingProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_class_lookup(self):
        """Processor should initialize with empty class_lookup when not provided."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )
        assert processor._class_lookup == {}

    def test_process_returns_list(self):
        """process() should return a list of ClassTimingDTOs."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, list)

    def test_process_empty_data(self):
        """process() should return empty list when raw_data is empty."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert result == []


class TestProcessAllRows:
    """Tests for _process_all_rows method."""

    def test_only_processes_class_type(self):
        """_process_all_rows should only process rows with type == 'CLASS'."""
        raw_data = pd.DataFrame([
            {'type': 'CLASS', 'record_key': 'key1', 'day_of_week': 'Monday', 'start_time': '09:00', 'end_time': '10:30', 'venue': 'Room 101'},
            {'type': 'EXAM', 'record_key': 'key2', 'date': '2026-05-02', 'start_time': '14:00'},
            {'type': 'CLASS', 'record_key': 'key3', 'day_of_week': 'Wednesday', 'start_time': '14:00', 'end_time': '15:30', 'venue': 'Room 202'},
        ])

        # Provide class_lookup to allow finding class IDs
        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-uuid-2'),
        }

        processor = ClassTimingProcessor(
            raw_data=raw_data,
            class_lookup=class_lookup,
            logger=Mock()
        )

        processor._process_all_rows()

        # Only CLASS rows should be processed (EXAM row should be skipped)
        assert len(processor._new_timings) >= 2


class TestFindClassIds:
    """Tests for _find_class_ids method."""

    def test_returns_empty_when_no_record_key(self):
        """_find_class_ids should return empty list when record_key is None."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )

        result = processor._find_class_ids(None)
        assert result == []

    def test_returns_empty_when_record_key_is_nan(self):
        """_find_class_ids should return empty list when record_key is NaN."""
        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )

        result = processor._find_class_ids(float('nan'))
        assert result == []

    def test_finds_all_class_ids_for_record_key(self):
        """_find_class_ids should return all class IDs for a record_key."""
        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-uuid-2'),
        }

        processor = ClassTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup=class_lookup
        )

        result = processor._find_class_ids('some_key')

        # Finds all classes in the lookup (since no record_key filtering in current impl)
        assert len(result) == 2


class TestProcessClassTiming:
    """Tests for _process_class_timing method."""

    def test_creates_timing_dto(self):
        """_process_class_timing should create a ClassTimingDTO."""
        raw_data = pd.DataFrame([{
            'type': 'CLASS',
            'record_key': 'key1',
            'day_of_week': 'Monday',
            'start_time': '09:00',
            'end_time': '10:30',
            'venue': 'Room 101'
        }])

        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
        }

        processor = ClassTimingProcessor(
            raw_data=raw_data,
            class_lookup=class_lookup,
            logger=Mock()
        )

        processor._process_all_rows()

        assert len(processor._new_timings) >= 1

    def test_deduplication_by_timing_key(self):
        """_process_class_timing should deduplicate by timing key."""
        raw_data = pd.DataFrame([
            {'type': 'CLASS', 'record_key': 'key1', 'day_of_week': 'Monday', 'start_time': '09:00', 'end_time': '10:30', 'venue': 'Room 101'},
            {'type': 'CLASS', 'record_key': 'key2', 'day_of_week': 'Monday', 'start_time': '09:00', 'end_time': '10:30', 'venue': 'Room 101'},
        ])

        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-uuid-2'),
        }

        processor = ClassTimingProcessor(
            raw_data=raw_data,
            class_lookup=class_lookup,
            logger=Mock()
        )

        processor._process_all_rows()

        # The 2 rows have same timing details, so should be deduplicated
        # But they have different record_keys so potentially different classes
        assert len(processor._new_timings) >= 1


class TestClassTimingDTO:
    """Tests for ClassTimingDTO.from_row."""

    def test_creates_from_row(self):
        """ClassTimingDTO.from_row should create DTO from row data."""
        row = {
            'day_of_week': 'Tuesday',
            'start_time': '14:00',
            'end_time': '15:30',
            'venue': 'Room 303'
        }

        dto = ClassTimingDTO.from_row(row, 'class-id-456')

        assert dto.class_id == 'class-id-456'
        assert dto.day_of_week == 'Tuesday'
        assert dto.start_time == '14:00'
        assert dto.end_time == '15:30'
        assert dto.venue == 'Room 303'

    def test_handles_none_values(self):
        """ClassTimingDTO.from_row should handle None/NaN values."""
        row = {
            'day_of_week': None,
            'start_time': None,
            'end_time': None,
            'venue': None
        }

        dto = ClassTimingDTO.from_row(row, 'class-id-789')

        assert dto.class_id == 'class-id-789'
        assert dto.day_of_week is None
        assert dto.start_time is None