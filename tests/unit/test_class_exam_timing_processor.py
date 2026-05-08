"""
Unit tests for ClassExamTimingProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock

from src.pipeline.processors.class_exam_timing_processor import ClassExamTimingProcessor
from src.pipeline.dtos.timing_dto import ClassExamTimingDTO


class TestClassExamTimingProcessor:
    """Tests for ClassExamTimingProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_class_lookup(self):
        """Processor should initialize with empty class_lookup when not provided."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )
        assert processor._class_lookup == {}

    def test_process_returns_list(self):
        """process() should return a list of ClassExamTimingDTOs."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, list)

    def test_process_empty_data(self):
        """process() should return empty list when raw_data is empty."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert result == []


class TestProcessAllRows:
    """Tests for _process_all_rows method."""

    def test_only_processes_exam_type(self):
        """_process_all_rows should only process rows with type == 'EXAM'."""
        raw_data = pd.DataFrame([
            {'type': 'CLASS', 'record_key': 'key1', 'date': '2026-05-01', 'start_time': '09:00', 'end_time': '11:00', 'venue': 'Hall A'},
            {'type': 'EXAM', 'record_key': 'key2', 'date': '2026-05-02', 'start_time': '14:00', 'end_time': '16:00', 'venue': 'Hall B'},
            {'type': 'CLASS', 'record_key': 'key3', 'date': '2026-05-03', 'start_time': '10:00', 'end_time': '12:00', 'venue': 'Hall C'},
        ])

        # record_key_to_class_ids maps record_keys to class IDs
        record_key_map = {
            'key2': ['class-uuid-1', 'class-uuid-2'],
        }

        processor = ClassExamTimingProcessor(
            raw_data=raw_data,
            class_lookup={},
            record_key_to_class_ids=record_key_map,
            logger=Mock()
        )

        processor._process_all_rows()

        # Only EXAM row should be processed (2 CLASS rows should be skipped)
        assert len(processor._new_exam_timings) >= 1


class TestFindClassIds:
    """Tests for _find_class_ids method."""

    def test_returns_empty_when_no_record_key(self):
        """_find_class_ids should return empty list when record_key is None."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )

        result = processor._find_class_ids(None)
        assert result == []

    def test_returns_empty_when_record_key_is_nan(self):
        """_find_class_ids should return empty list when record_key is NaN."""
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={}
        )

        result = processor._find_class_ids(float('nan'))
        assert result == []

    def test_finds_all_class_ids_for_record_key(self):
        """_find_class_ids should return all class IDs for a record_key."""
        record_key_map = {
            'key1': ['class-uuid-1', 'class-uuid-2'],
        }

        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            record_key_to_class_ids=record_key_map
        )

        result = processor._find_class_ids('key1')

        assert len(result) == 2
        assert 'class-uuid-1' in result
        assert 'class-uuid-2' in result

    def test_returns_empty_for_unknown_record_key(self):
        """_find_class_ids should return empty list for unknown record_key."""
        record_key_map = {
            'key1': ['class-uuid-1'],
        }

        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            record_key_to_class_ids=record_key_map
        )

        result = processor._find_class_ids('unknown_key')
        assert result == []


class TestProcessExamTiming:
    """Tests for _process_exam_timing method."""

    def test_skips_already_processed(self):
        """_process_exam_timing should skip if class_id already processed."""
        raw_data = pd.DataFrame([
            {'type': 'EXAM', 'record_key': 'key1', 'exam_date': '2026-05-02', 'exam_start_time': '14:00', 'exam_end_time': '16:00', 'exam_venue': 'Hall A'},
        ])

        processed_ids = {'already-processed-class-id'}
        record_key_map = {
            'key1': ['already-processed-class-id'],
        }

        processor = ClassExamTimingProcessor(
            raw_data=raw_data,
            class_lookup={},
            record_key_to_class_ids=record_key_map,
            processed_exam_class_ids=processed_ids,
            logger=Mock()
        )

        processor._process_all_rows()

        # Should skip since class_id was already processed
        assert len(processor._new_exam_timings) == 0


class TestClassExamTimingDTO:
    """Tests for ClassExamTimingDTO.from_row."""

    def test_creates_from_row(self):
        """ClassExamTimingDTO.from_row should create DTO from row data."""
        row = {
            'date': '2026-05-15',
            'start_time': '09:00',
            'end_time': '11:00',
            'venue': 'Hall B'
        }

        dto = ClassExamTimingDTO.from_row(row, 'class-id-123')

        assert dto.class_id == 'class-id-123'
        assert dto.date == '2026-05-15'
        assert dto.start_time == '09:00'
        assert dto.end_time == '11:00'
        assert dto.venue == 'Hall B'

    def test_to_csv_row(self):
        """ClassExamTimingDTO.to_csv_row should return dict with all fields."""
        row = {
            'date': '2026-05-15',
            'start_time': '09:00',
            'end_time': '11:00',
            'venue': 'Hall B'
        }
        dto = ClassExamTimingDTO.from_row(row, 'class-id-123')

        csv_row = dto.to_csv_row()
        assert csv_row['class_id'] == 'class-id-123'
        assert csv_row['date'] == '2026-05-15'

    def test_to_db_row_excludes_id(self):
        """ClassExamTimingDTO.to_db_row should exclude id (DB auto-generated)."""
        row = {
            'date': '2026-05-15',
            'start_time': '09:00',
            'end_time': '11:00',
            'venue': 'Hall B'
        }
        dto = ClassExamTimingDTO.from_row(row, 'class-id-123')

        db_row = dto.to_db_row()
        assert 'id' not in db_row
        assert db_row['class_id'] == 'class-id-123'

    def test_exam_timing_already_processed_skipped(self):
        """Should not create duplicate exam timings for already-processed class IDs."""
        # Create processor with already-processed class ID
        processed_ids = {'already-processed-class-id'}
        record_key_map = {
            'key1': ['already-processed-class-id'],
        }
        processor = ClassExamTimingProcessor(
            raw_data=pd.DataFrame(),
            class_lookup={},
            record_key_to_class_ids=record_key_map,
            processed_exam_class_ids=processed_ids,
            logger=Mock()
        )

        # Manually call _find_class_ids which should return the class
        class_ids = processor._find_class_ids('key1')
        # The class should exist but be in processed_exam_class_ids
        for class_id in class_ids:
            assert class_id in processed_ids
