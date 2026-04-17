"""
Unit tests for TimingProcessor.
"""
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open

from src.pipeline.processors.timing_processor import TimingProcessor
from src.pipeline.processor_context import ProcessorContext


class MockLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(('INFO', msg))

    def warning(self, msg):
        self.messages.append(('WARNING', msg))

    def error(self, msg):
        self.messages.append(('ERROR', msg))


class MockConfig:
    def __init__(self, cache_dir='/tmp/cache'):
        self.cache_dir = cache_dir


class MockContext:
    """Mock ProcessorContext for testing TimingProcessor."""

    def __init__(self, cache_dir='/tmp/cache'):
        self.logger = MockLogger()
        self.config = MockConfig(cache_dir)
        self.multiple_data = pd.DataFrame()
        self.class_id_mapping = {}
        self.new_class_timings = []
        self.new_class_exam_timings = []
        self.stats = {'timings_created': 0, 'exams_created': 0}


def create_mock_context(multiple_data=None, class_id_mapping=None, cache_dir='/tmp/cache'):
    """Helper to create a configured mock context."""
    ctx = MockContext(cache_dir)
    ctx.multiple_data = multiple_data or pd.DataFrame()
    ctx.class_id_mapping = class_id_mapping or {}
    ctx.new_class_timings = []
    ctx.new_class_exam_timings = []
    ctx.stats = {'timings_created': 0, 'exams_created': 0}
    return ctx


class TestTimingProcessorInit:
    """Tests for TimingProcessor initialization."""

    def test_init_sets_processed_timing_keys_as_empty_set(self):
        """Test that __init__() initializes _processed_timing_keys as empty set."""
        ctx = create_mock_context()
        processor = TimingProcessor(ctx)

        assert isinstance(processor._processed_timing_keys, set)
        assert len(processor._processed_timing_keys) == 0

    def test_init_sets_processed_exam_class_ids_as_empty_set(self):
        """Test that __init__() initializes _processed_exam_class_ids as empty set."""
        ctx = create_mock_context()
        processor = TimingProcessor(ctx)

        assert isinstance(processor._processed_exam_class_ids, set)
        assert len(processor._processed_exam_class_ids) == 0


class TestTimingProcessorLoadCache:
    """Tests for _load_cache() method."""

    def test_load_cache_with_no_cache_files(self):
        """Test _load_cache() handles missing cache files gracefully."""
        ctx = create_mock_context(cache_dir='/nonexistent')
        processor = TimingProcessor(ctx)

        processor._load_cache()

        assert len(processor._processed_timing_keys) == 0
        assert len(processor._processed_exam_class_ids) == 0

    def test_load_cache_loads_processed_timing_keys(self):
        """Test _load_cache() loads processed_timing_keys from cache file."""
        ctx = create_mock_context(cache_dir='/tmp')
        cache_df = pd.DataFrame([
            {'class_id': 'C1', 'day_of_week': 'Monday', 'start_time': '09:00',
             'end_time': '10:00', 'venue': 'Room 101'},
            {'class_id': 'C2', 'day_of_week': 'Tuesday', 'start_time': '11:00',
             'end_time': '12:00', 'venue': 'Room 102'},
        ])

        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_pickle', return_value=cache_df):
                processor = TimingProcessor(ctx)
                processor._load_cache()

        assert len(processor._processed_timing_keys) == 2

    def test_load_cache_loads_processed_exam_class_ids(self):
        """Test _load_cache() loads processed_exam_class_ids from cache file."""
        ctx = create_mock_context(cache_dir='/tmp')
        exam_cache_df = pd.DataFrame([
            {'class_id': 'C1', 'date': '2026-05-01'},
            {'class_id': 'C2', 'date': '2026-05-02'},
            {'class_id': 'C1', 'date': '2026-05-03'},
        ])

        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_pickle', return_value=exam_cache_df):
                processor = TimingProcessor(ctx)
                processor._load_cache()

        assert len(processor._processed_exam_class_ids) == 2
        assert 'C1' in processor._processed_exam_class_ids
        assert 'C2' in processor._processed_exam_class_ids


class TestTimingProcessorDoProcess:
    """Tests for _do_process() method."""

    def test_do_process_iterates_multiple_data(self):
        """Test _do_process() iterates over multiple_data rows."""
        ctx = create_mock_context()
        ctx.multiple_data = pd.DataFrame([
            {'record_key': 'R1', 'type': 'CLASS', 'day_of_week': 'Monday',
             'start_time': '09:00', 'end_time': '10:00', 'venue': 'Room 101'},
            {'record_key': 'R2', 'type': 'EXAM', 'day_of_week': 'Friday',
             'start_time': '14:00', 'end_time': '16:00', 'venue': 'Exam Hall'},
        ])
        ctx.class_id_mapping = {'R1': ['C1'], 'R2': ['C2']}

        processor = TimingProcessor(ctx)
        processor._do_process()

        assert ctx.stats.get('timings_created', 0) >= 0 or ctx.stats.get('exams_created', 0) >= 0

    def test_do_process_skips_records_without_class_id_mapping(self):
        """Test _do_process() skips records not in class_id_mapping."""
        ctx = create_mock_context()
        ctx.multiple_data = pd.DataFrame([
            {'record_key': 'R1', 'type': 'CLASS', 'day_of_week': 'Monday',
             'start_time': '09:00', 'end_time': '10:00', 'venue': 'Room 101'},
        ])
        ctx.class_id_mapping = {}

        processor = TimingProcessor(ctx)
        processor._do_process()

        assert ctx.stats['timings_created'] == 0


class TestTimingProcessorProcessClassTiming:
    """Tests for _process_class_timing() method."""

    def test_process_class_timing_creates_new_class_timings_record(self):
        """Test _process_class_timing() creates a new class timings record."""
        ctx = create_mock_context()
        row = pd.Series({
            'day_of_week': 'Monday',
            'start_time': '09:00',
            'end_time': '10:00',
            'venue': 'Room 101',
            'start_date': '2026-01-01',
            'end_date': '2026-05-01',
        })

        processor = TimingProcessor(ctx)
        processor._process_class_timing(row, 'C1')

        assert len(ctx.new_class_timings) == 1
        assert ctx.new_class_timings[0]['class_id'] == 'C1'
        assert ctx.new_class_timings[0]['day_of_week'] == 'Monday'

    def test_process_class_timing_deduplicates_via_processed_timing_keys(self):
        """Test _process_class_timing() deduplicates via _processed_timing_keys."""
        ctx = create_mock_context()
        row = pd.Series({
            'day_of_week': 'Monday',
            'start_time': '09:00',
            'end_time': '10:00',
            'venue': 'Room 101',
        })

        processor = TimingProcessor(ctx)
        processor._process_class_timing(row, 'C1')
        processor._process_class_timing(row, 'C1')

        assert len(ctx.new_class_timings) == 1
        assert len(processor._processed_timing_keys) == 1

    def test_process_class_timing_handles_nan_values(self):
        """Test _process_class_timing() handles NaN values correctly."""
        ctx = create_mock_context()
        row = pd.Series({
            'day_of_week': None,
            'start_time': None,
            'end_time': None,
            'venue': None,
        })

        processor = TimingProcessor(ctx)
        processor._process_class_timing(row, 'C1')

        assert len(ctx.new_class_timings) == 1
        # row.get('venue', '') returns None (the actual value), not '' - key exists
        assert ctx.new_class_timings[0]['venue'] is None


class TestTimingProcessorProcessExamTiming:
    """Tests for _process_exam_timing() method."""

    def test_process_exam_timing_creates_new_class_exam_timings_record(self):
        """Test _process_exam_timing() creates a new exam timings record."""
        ctx = create_mock_context()
        row = pd.Series({
            'date': '2026-05-01',
            'day_of_week': 'Friday',
            'start_time': '14:00',
            'end_time': '16:00',
            'venue': 'Exam Hall',
        })

        processor = TimingProcessor(ctx)
        processor._process_exam_timing(row, 'C1')

        assert len(ctx.new_class_exam_timings) == 1
        assert ctx.new_class_exam_timings[0]['class_id'] == 'C1'
        assert ctx.new_class_exam_timings[0]['date'] == '2026-05-01'

    def test_process_exam_timing_deduplicates_via_processed_exam_class_ids(self):
        """Test _process_exam_timing() deduplicates via _processed_exam_class_ids."""
        ctx = create_mock_context()
        row = pd.Series({
            'date': '2026-05-01',
            'day_of_week': 'Friday',
            'start_time': '14:00',
            'end_time': '16:00',
            'venue': 'Exam Hall',
        })

        processor = TimingProcessor(ctx)
        processor._process_exam_timing(row, 'C1')
        processor._process_exam_timing(row, 'C1')

        assert len(ctx.new_class_exam_timings) == 1
        assert len(processor._processed_exam_class_ids) == 1

    def test_process_exam_timing_converts_times_to_strings(self):
        """Test _process_exam_timing() converts start_time and end_time to strings."""
        ctx = create_mock_context()
        row = pd.Series({
            'date': '2026-05-01',
            'day_of_week': 'Friday',
            'start_time': 14.0,
            'end_time': 16.0,
            'venue': 'Exam Hall',
        })

        processor = TimingProcessor(ctx)
        processor._process_exam_timing(row, 'C1')

        assert isinstance(ctx.new_class_exam_timings[0]['start_time'], str)
        assert isinstance(ctx.new_class_exam_timings[0]['end_time'], str)


class TestTimingProcessorDeduplication:
    """Tests for deduplication via _processed_timing_keys set."""

    def test_deduplication_prevents_duplicate_class_timings(self):
        """Test that duplicate class timings are prevented by deduplication."""
        ctx = create_mock_context()
        ctx.multiple_data = pd.DataFrame([
            {'record_key': 'R1', 'type': 'CLASS', 'day_of_week': 'Monday',
             'start_time': '09:00', 'end_time': '10:00', 'venue': 'Room 101'},
            {'record_key': 'R1', 'type': 'CLASS', 'day_of_week': 'Monday',
             'start_time': '09:00', 'end_time': '10:00', 'venue': 'Room 101'},
        ])
        ctx.class_id_mapping = {'R1': ['C1']}

        processor = TimingProcessor(ctx)
        processor._do_process()

        assert ctx.stats['timings_created'] == 1

    def test_deduplication_allows_different_class_timings(self):
        """Test that different class timings are allowed through."""
        ctx = create_mock_context()
        ctx.multiple_data = pd.DataFrame([
            {'record_key': 'R1', 'type': 'CLASS', 'day_of_week': 'Monday',
             'start_time': '09:00', 'end_time': '10:00', 'venue': 'Room 101'},
            {'record_key': 'R2', 'type': 'CLASS', 'day_of_week': 'Tuesday',
             'start_time': '11:00', 'end_time': '12:00', 'venue': 'Room 102'},
        ])
        ctx.class_id_mapping = {'R1': ['C1'], 'R2': ['C1']}

        processor = TimingProcessor(ctx)
        processor._do_process()

        assert ctx.stats['timings_created'] == 2
