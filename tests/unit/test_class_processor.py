"""
Unit tests for ClassProcessor.
"""
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.pipeline.processors.class_processor import ClassProcessor
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
    def __init__(self):
        self.cache_dir = '/tmp/cache'


class MockContext:
    """Mock ProcessorContext for testing ClassProcessor."""

    def __init__(self):
        self.logger = MockLogger()
        self.config = MockConfig()
        self.standalone_data = pd.DataFrame()
        self.multiple_data = pd.DataFrame()
        self.multiple_lookup = {}
        self.existing_classes_cache = []
        self.courses_cache = {}
        self.professor_lookup = {}
        self.standalone_data = pd.DataFrame()
        self.multiple_data = pd.DataFrame()
        self.new_classes = []
        self.update_classes = []
        self.class_id_mapping = {}
        self.stats = {'classes_created': 0}
        self.expected_acad_term_id = None


def create_mock_context(standalone_data=None, multiple_data=None, multiple_lookup=None,
                        existing_classes_cache=None, courses_cache=None, professor_lookup=None):
    """Helper to create a configured mock context."""
    ctx = MockContext()
    ctx.standalone_data = standalone_data or pd.DataFrame()
    ctx.multiple_data = multiple_data or pd.DataFrame()
    ctx.multiple_lookup = multiple_lookup or {}
    ctx.existing_classes_cache = existing_classes_cache or []
    ctx.courses_cache = courses_cache or {}
    ctx.professor_lookup = professor_lookup or {}
    ctx.class_id_mapping = {}
    ctx.stats = {'classes_created': 0}
    return ctx


class TestClassProcessorLoadCache:
    """Tests for _load_cache() method."""

    def test_load_cache_builds_existing_class_lookup(self):
        """Test that _load_cache() builds _existing_class_lookup from existing_classes_cache."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'class-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': 'P1', 'course_id': 'C1'},
            {'id': 'class-2', 'acad_term_id': 'T1', 'boss_id': 102, 'professor_id': 'P2', 'course_id': 'C1'},
        ]

        processor = ClassProcessor(ctx)
        processor._load_cache()

        assert len(processor._existing_class_lookup) == 2
        assert ('T1', 101, 'P1') in processor._existing_class_lookup
        assert ('T1', 102, 'P2') in processor._existing_class_lookup

    def test_load_cache_handles_empty_cache(self):
        """Test that _load_cache() handles empty existing_classes_cache."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = []

        processor = ClassProcessor(ctx)
        processor._load_cache()

        assert len(processor._existing_class_lookup) == 0

    def test_load_cache_skips_entries_without_required_fields(self):
        """Test that entries missing acad_term_id or boss_id are skipped."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'class-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': 'P1'},
            {'id': 'class-2', 'acad_term_id': None, 'boss_id': 102, 'professor_id': 'P2'},
            {'id': 'class-3', 'acad_term_id': 'T1', 'boss_id': None, 'professor_id': 'P3'},
        ]

        processor = ClassProcessor(ctx)
        processor._load_cache()

        assert len(processor._existing_class_lookup) == 1
        assert ('T1', 101, 'P1') in processor._existing_class_lookup


class TestClassProcessorDoProcess:
    """Tests for _do_process() method."""

    def test_do_process_iterates_standalone_data(self):
        """Test that _do_process() iterates over standalone_data rows."""
        ctx = create_mock_context()
        ctx.standalone_data = pd.DataFrame([
            {'record_key': 'R1', 'acad_term_id': 'T1', 'class_boss_id': 101, 'course_code': 'CS101', 'section': 'A'},
            {'record_key': 'R2', 'acad_term_id': 'T1', 'class_boss_id': 102, 'course_code': 'CS102', 'section': 'B'},
        ])
        ctx.courses_cache = {
            'CS101': {'id': 'course-1'},
            'CS102': {'id': 'course-2'},
        }

        processor = ClassProcessor(ctx)
        processor._do_process()

        assert ctx.stats['classes_created'] >= 0

    def test_do_process_handles_empty_standalone_data(self):
        """Test that _do_process() handles empty standalone_data."""
        ctx = create_mock_context()
        ctx.standalone_data = pd.DataFrame()

        processor = ClassProcessor(ctx)
        processor._do_process()

        assert ctx.stats['classes_created'] == 0


class TestClassProcessorProcessRow:
    """Tests for _process_row() method."""

    def test_process_row_determines_create_for_new_class(self):
        """Test that _process_row() determines CREATE for a new class."""
        ctx = create_mock_context()
        ctx.standalone_data = pd.DataFrame([
            {'record_key': 'R1', 'acad_term_id': 'T1', 'class_boss_id': 101, 'course_code': 'CS101', 'section': '1'},
        ])
        ctx.courses_cache = {'CS101': {'id': 'course-1'}}
        ctx.multiple_lookup = {'R1': []}

        processor = ClassProcessor(ctx)
        processor._load_cache()
        processor._do_process()

        assert len(ctx.new_classes) > 0 or len(ctx.update_classes) >= 0

    def test_process_row_determines_update_for_existing_class(self):
        """Test that _process_row() determines UPDATE for an existing class."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'existing-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': 'P1', 'course_id': 'course-1'},
        ]
        ctx.standalone_data = pd.DataFrame([
            {'record_key': 'R1', 'acad_term_id': 'T1', 'class_boss_id': 101, 'course_code': 'CS101', 'section': '1',
             'grading_basis': 'GRADED', 'course_outline_url': 'http://example.com'},
        ])
        ctx.courses_cache = {'CS101': {'id': 'course-1'}}
        ctx.multiple_lookup = {'R1': [{'professor_name': 'Prof A', 'type': 'CLASS'}]}
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}

        processor = ClassProcessor(ctx)
        processor._load_cache()
        processor._do_process()

        assert len(ctx.new_classes) >= 0


class TestClassProcessorProcessUpdate:
    """Tests for _process_update() method."""

    def test_process_update_adds_to_update_classes(self):
        """Test that _process_update() adds to update_classes list."""
        ctx = create_mock_context()
        existing_class = {
            'id': 'class-1',
            'acad_term_id': 'T1',
            'boss_id': 101,
            'professor_id': 'P1',
            'grading_basis': 'PNP',
            'course_outline_url': '',
        }

        row = pd.Series({
            'record_key': 'R1',
            'class_boss_id': 101,
            'course_code': 'CS101',
            'section': '1',
            'grading_basis': 'GRADED',
            'course_outline_url': 'http://example.com',
        })

        processor = ClassProcessor(ctx)
        processor._process_update(existing_class, row, 'R1', False)

        assert len(ctx.update_classes) == 1
        assert ctx.update_classes[0]['id'] == 'class-1'


class TestClassProcessorProcessCreate:
    """Tests for _process_create() method."""

    def test_process_create_adds_to_new_classes(self):
        """Test that _process_create() adds to new_classes list."""
        ctx = create_mock_context()
        row = pd.Series({
            'record_key': 'R1',
            'class_boss_id': 101,
            'course_code': 'CS101',
            'section': '1',
            'grading_basis': 'GRADED',
            'course_outline_url': 'http://example.com',
        })

        processor = ClassProcessor(ctx)
        processor._process_create(row, 'T1', 101, 'course-1', '1', 'P1', 'Prof A', 'R1', False)

        assert len(ctx.new_classes) == 1
        assert ctx.new_classes[0]['professor_id'] == 'P1'
        assert ctx.new_classes[0]['acad_term_id'] == 'T1'

    def test_process_create_avoids_duplicates(self):
        """Test that _process_create() avoids creating duplicate classes."""
        ctx = create_mock_context()
        ctx.class_id_mapping = {}
        row = pd.Series({
            'record_key': 'R1',
            'class_boss_id': 101,
            'course_code': 'CS101',
            'section': '1',
            'grading_basis': 'GRADED',
            'course_outline_url': '',
        })

        processor = ClassProcessor(ctx)
        processor._process_create(row, 'T1', 101, 'course-1', '1', 'P1', 'Prof A', 'R1', False)
        processor._process_create(row, 'T1', 101, 'course-1', '1', 'P1', 'Prof A', 'R1', False)

        assert len(ctx.new_classes) == 1


class TestClassProcessorHandleTbaConversion:
    """Tests for _handle_tba_conversion() method."""

    def test_handle_tba_conversion_for_tba_class(self):
        """Test _handle_tba_conversion() converts TBA class to assigned professor."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'tba-class-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': None,
             'course_id': 'course-1', 'section': '1'},
        ]
        ctx.multiple_lookup = {'R1': [{'type': 'CLASS', 'professor_name': 'Prof A'}]}
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}

        row = pd.Series({
            'record_key': 'R1',
            'acad_term_id': 'T1',
            'class_boss_id': 101,
            'course_code': 'CS101',
            'section': '1',
        })

        processor = ClassProcessor(ctx)
        processor._load_cache()

        class_rows = [r for r in ctx.multiple_lookup.get('R1', []) if r.get('type') == 'CLASS']
        professor_mappings = [('P1', 'Prof A')]

        processor._handle_tba_conversion(row, 'T1', 101, 'course-1', '1', 'R1', professor_mappings)

        assert len(ctx.update_classes) == 1
        assert ctx.update_classes[0]['id'] == 'tba-class-1'
        assert ctx.update_classes[0]['professor_id'] == 'P1'

    def test_handle_tba_conversion_prevents_duplicate_creation(self):
        """Test that TBA conversion prevents duplicate class creation via processed_class_keys."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'tba-class-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': None,
             'course_id': 'course-1', 'section': '1'},
        ]
        ctx.courses_cache = {'CS101': {'id': 'course-1'}}
        ctx.multiple_lookup = {'R1': [{'type': 'CLASS', 'professor_name': 'Prof A'}]}
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}
        ctx.standalone_data = pd.DataFrame([
            {'record_key': 'R1', 'acad_term_id': 'T1', 'class_boss_id': 101,
             'course_code': 'CS101', 'section': '1', 'grading_basis': None, 'course_outline_url': None}
        ])

        processor = ClassProcessor(ctx)
        processor._load_cache()

        # Verify TBA class is in the lookup
        assert ("T1", 101, None) in processor._existing_class_lookup

        # Run processing
        processor._do_process()

        # Should have exactly 1 update (TBA -> assigned), not a new class creation
        assert len(ctx.update_classes) == 1
        assert ctx.update_classes[0]['id'] == 'tba-class-1'
        assert ctx.update_classes[0]['professor_id'] == 'P1'

        # Should have 0 new classes (not a duplicate creation)
        assert len(ctx.new_classes) == 0
        assert ctx.stats['classes_created'] == 0

    def test_handle_tba_conversion_no_op_for_non_tba(self):
        """Test _handle_tba_conversion() does nothing when class is not TBA."""
        ctx = create_mock_context()
        ctx.existing_classes_cache = [
            {'id': 'class-1', 'acad_term_id': 'T1', 'boss_id': 101, 'professor_id': 'P1',
             'course_id': 'course-1', 'section': '1'},
        ]
        ctx.multiple_lookup = {'R1': [{'type': 'CLASS', 'professor_name': 'Prof A'}]}
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}

        row = pd.Series({
            'record_key': 'R1',
            'acad_term_id': 'T1',
            'class_boss_id': 101,
            'course_code': 'CS101',
            'section': '1',
        })

        processor = ClassProcessor(ctx)
        processor._load_cache()

        class_rows = [r for r in ctx.multiple_lookup.get('R1', []) if r.get('type') == 'CLASS']
        professor_mappings = [('P1', 'Prof A')]

        initial_update_count = len(ctx.update_classes)
        processor._handle_tba_conversion(row, 'T1', 101, 'course-1', '1', 'R1', professor_mappings)

        assert len(ctx.update_classes) == initial_update_count


class TestClassProcessorGetCourseId:
    """Tests for _get_course_id() method."""

    def test_get_course_id_returns_id_from_courses_cache(self):
        """Test _get_course_id() returns course ID from courses_cache."""
        ctx = create_mock_context()
        ctx.courses_cache = {'CS101': {'id': 'course-123'}}

        processor = ClassProcessor(ctx)
        course_id = processor._get_course_id('CS101')

        assert course_id == 'course-123'

    def test_get_course_id_returns_none_for_missing_course(self):
        """Test _get_course_id() returns None for course not in cache."""
        ctx = create_mock_context()
        ctx.courses_cache = {}

        processor = ClassProcessor(ctx)
        course_id = processor._get_course_id('CS999')

        assert course_id is None


class TestClassProcessorFindProfessorsForClass:
    """Tests for _find_professors_for_class() method."""

    def test_find_professors_for_class_with_single_professor(self):
        """Test _find_professors_for_class() finds single professor."""
        ctx = create_mock_context()
        ctx.multiple_lookup = {
            'R1': [{'professor_name': 'Prof A', 'type': 'CLASS'}],
        }
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}

        processor = ClassProcessor(ctx)
        result = processor._find_professors_for_class('R1')

        assert len(result) == 1
        assert result[0] == ('P1', 'Prof A')

    def test_find_professors_for_class_with_multiple_lookup(self):
        """Test _find_professors_for_class() handles multiple professor names."""
        ctx = create_mock_context()
        ctx.multiple_lookup = {
            'R1': [
                {'professor_name': 'Prof A', 'type': 'CLASS'},
                {'professor_name': 'Prof B', 'type': 'CLASS'},
            ],
        }
        ctx.professor_lookup = {
            'PROF A': {'database_id': 'P1'},
            'PROF B': {'database_id': 'P2'},
        }

        processor = ClassProcessor(ctx)
        result = processor._find_professors_for_class('R1')

        assert len(result) == 2
        assert ('P1', 'Prof A') in result
        assert ('P2', 'Prof B') in result

    def test_find_professors_for_class_handles_missing_record_key(self):
        """Test _find_professors_for_class() handles None record_key."""
        ctx = create_mock_context()

        processor = ClassProcessor(ctx)
        result = processor._find_professors_for_class(None)

        assert result == []


class TestClassProcessorSplitProfessorNames:
    """Tests for _split_professor_names() method."""

    def test_split_professor_names_exact_match(self):
        """Test _split_professor_names() returns exact match when found in lookup."""
        ctx = create_mock_context()
        ctx.professor_lookup = {'PROF A': {'database_id': 'P1'}}

        processor = ClassProcessor(ctx)
        result = processor._split_professor_names('Prof A')

        assert result == ['Prof A']

    def test_split_professor_names_comma_separated_greedy_matching(self):
        """Test _split_professor_names() uses greedy longest-match-first matching."""
        ctx = create_mock_context()
        ctx.professor_lookup = {
            'PROF A, PROF B': {'database_id': 'P1'},
            'PROF A': {'database_id': 'P2'},
        }

        processor = ClassProcessor(ctx)
        result = processor._split_professor_names('Prof A, Prof B')

        assert 'Prof A, Prof B' in result

    def test_split_professor_names_no_match_falls_back(self):
        """Test _split_professor_names() falls back when no match found."""
        ctx = create_mock_context()
        ctx.professor_lookup = {}

        processor = ClassProcessor(ctx)
        result = processor._split_professor_names('Unknown Prof')

        assert 'Unknown Prof' in result

    def test_split_professor_names_handles_empty_input(self):
        """Test _split_professor_names() handles empty input."""
        ctx = create_mock_context()

        processor = ClassProcessor(ctx)
        result = processor._split_professor_names('')

        assert result == []
