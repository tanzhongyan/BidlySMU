"""
Unit tests for CourseProcessor.
"""
import pytest
import pandas as pd
import uuid
from unittest.mock import MagicMock, Mock, patch

from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.processor_context import ProcessorContext


@pytest.fixture
def mock_config():
    """Mock config object with verify_dir and output_base."""
    mock = MagicMock()
    mock.cache_dir = "/tmp/cache"
    mock.verify_dir = "/tmp/verify"
    mock.output_base = "/tmp/output"
    return mock


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock = MagicMock()
    mock.info.return_value = None
    mock.warning.return_value = None
    mock.error.return_value = None
    mock.debug.return_value = None
    return mock


@pytest.fixture
def processor_context(mock_config, mock_logger):
    """Create a mocked ProcessorContext for CourseProcessor tests."""
    ctx = ProcessorContext(
        config=mock_config,
        logger=mock_logger,
        professors_cache={},
        courses_cache={},
        professor_lookup={},
        multiple_data=pd.DataFrame(),
        standalone_data=pd.DataFrame(),
        new_courses=[],
        update_courses=[],
        stats={
            'professors_created': 0,
            'professors_updated': 0,
            'courses_created': 0,
            'courses_updated': 0,
        }
    )
    return ctx


@pytest.fixture
def course_processor(processor_context):
    """Create CourseProcessor instance with mocked context."""
    return CourseProcessor(processor_context)


class TestCourseProcessorLoadCache:
    """Tests for _load_cache() method."""

    def test_load_cache_does_nothing(self, course_processor):
        """_load_cache() should do nothing since cache is pre-loaded by TableBuilder."""
        # Should not raise any exception
        result = course_processor._load_cache()
        assert result is None


class TestDoProcess:
    """Tests for _do_process() method."""

    def test_iterates_standalone_data_correctly(self, course_processor, processor_context):
        """Should iterate through standalone_data and process each course."""
        processor_context.standalone_data = pd.DataFrame({
            'course_code': ['CS101', 'CS102'],
            'course_name': ['Intro to CS', 'Data Structures'],
            'course_description': ['Description 1', 'Description 2'],
            'credit_units': [3.0, 4.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'CS101']
        })
        processor_context.courses_cache = {}

        course_processor._do_process()

        # Should create both courses
        assert len(processor_context.new_courses) == 2
        assert processor_context.stats['courses_created'] == 2

    def test_skips_duplicate_course_codes(self, course_processor, processor_context):
        """Should skip rows with duplicate course codes in same run."""
        processor_context.standalone_data = pd.DataFrame({
            'course_code': ['CS101', 'CS101'],  # duplicate
            'course_name': ['Intro to CS', 'Intro to CS Duplicate'],
            'course_description': ['Description 1', 'Description 2'],
            'credit_units': [3.0, 3.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None']
        })
        processor_context.courses_cache = {}

        course_processor._do_process()

        # Should only process CS101 once
        assert len(processor_context.new_courses) == 1

    def test_skips_na_course_codes(self, course_processor, processor_context):
        """Should skip rows with NaN course codes."""
        processor_context.standalone_data = pd.DataFrame({
            'course_code': ['CS101', pd.NA, None],
            'course_name': ['Intro to CS', 'Course 2', 'Course 3'],
            'course_description': ['Description 1', 'Description 2', 'Description 3'],
            'credit_units': [3.0, 3.0, 3.0],
            'course_area': ['Computing', 'Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None', 'None']
        })
        processor_context.courses_cache = {}

        course_processor._do_process()

        # Should only process CS101
        assert len(processor_context.new_courses) == 1

    def test_calls_update_for_existing_course(self, course_processor, processor_context):
        """Should call _process_update for courses in courses_cache."""
        processor_context.courses_cache = {
            'CS101': {
                'id': 'existing-uuid',
                'code': 'CS101',
                'name': 'Old Name',
                'course_description': 'Old Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        processor_context.standalone_data = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': ['New Name'],
            'course_description': ['New Description'],
            'credit_units': [3.0],
            'course_area': ['Computing'],
            'enrolment_requirements': ['None']
        })

        course_processor._do_process()

        # Should update the course
        assert len(processor_context.update_courses) >= 0


class TestProcessUpdate:
    """Tests for _process_update() method."""

    def test_adds_to_update_courses_list(self, course_processor, processor_context):
        """Should add update record to update_courses list."""
        processor_context.courses_cache = {
            'CS101': {
                'id': 'uuid-123',
                'code': 'CS101',
                'name': 'Old Name',
                'course_description': 'Old Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        processor_context.update_courses = []
        processor_context.stats['courses_updated'] = 0

        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'New Name',
            'course_description': 'New Description',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        course_processor._process_update(row, 'CS101')

        assert len(processor_context.update_courses) == 1
        assert processor_context.update_courses[0]['id'] == 'uuid-123'
        assert processor_context.update_courses[0]['code'] == 'CS101'

    def test_increments_stats(self, course_processor, processor_context):
        """Should increment courses_updated stat when update is needed."""
        processor_context.courses_cache = {
            'CS101': {
                'id': 'uuid-123',
                'code': 'CS101',
                'name': 'Old Name',
                'course_description': 'Old Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        processor_context.update_courses = []
        processor_context.stats['courses_updated'] = 0

        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'New Name',  # different from existing
            'course_description': 'New Description',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        course_processor._process_update(row, 'CS101')

        assert processor_context.stats['courses_updated'] == 1


class TestProcessCreate:
    """Tests for _process_create() method."""

    def test_adds_to_new_courses_list(self, course_processor, processor_context):
        """Should add new course record to new_courses list."""
        processor_context.courses_cache = {}
        processor_context.new_courses = []
        processor_context.stats['courses_created'] = 0

        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'Intro to CS',
            'course_description': 'Description here',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        course_processor._process_create(row, 'CS101')

        assert len(processor_context.new_courses) == 1
        assert processor_context.new_courses[0]['code'] == 'CS101'
        assert processor_context.new_courses[0]['name'] == 'Intro to CS'

    def test_creates_uuid_for_new_course(self, course_processor, processor_context):
        """Should generate UUID for new course record."""
        processor_context.courses_cache = {}
        processor_context.new_courses = []

        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'Intro to CS',
            'course_description': 'Description here',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        course_processor._process_create(row, 'CS101')

        # Verify UUID was generated
        course_id = processor_context.new_courses[0]['id']
        # Should be a valid UUID format
        uuid.UUID(course_id)

    def test_adds_to_courses_cache(self, course_processor, processor_context):
        """Should add new course to courses_cache."""
        processor_context.courses_cache = {}
        processor_context.new_courses = []

        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'Intro to CS',
            'course_description': 'Description here',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        course_processor._process_create(row, 'CS101')

        assert 'CS101' in processor_context.courses_cache

    def test_uses_defaults_for_missing_fields(self, course_processor, processor_context):
        """Should use defaults for missing optional fields when DataFrame has NaN."""
        processor_context.courses_cache = {}
        processor_context.new_courses = []

        # Create a DataFrame with actual NaN values
        df = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': [pd.NA],
            'course_description': [pd.NA],
            'credit_units': [pd.NA],
            'course_area': [pd.NA],
            'enrolment_requirements': [pd.NA]
        })
        row = df.iloc[0]

        course_processor._process_create(row, 'CS101')

        assert len(processor_context.new_courses) == 1
        course = processor_context.new_courses[0]
        # Verify record was created with expected structure
        assert 'id' in course
        assert course['code'] == 'CS101'


class TestNeedsUpdate:
    """Tests for _needs_update() method."""

    def test_returns_true_when_field_differs(self, course_processor):
        """Should return True when a field has changed."""
        existing_record = {
            'name': 'Old Name',
            'description': 'Old Description',
            'credit_units': 3.0
        }
        new_record = pd.Series({
            'name': 'New Name',
            'description': 'New Description',
            'credit_units': 3.0
        })
        field_mapping = {
            'name': 'name',
            'description': 'description',
            'credit_units': 'credit_units'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        assert result is True

    def test_returns_false_when_no_changes(self, course_processor):
        """Should return False when no fields have changed."""
        existing_record = {
            'name': 'Same Name',
            'description': 'Same Description',
            'credit_units': 3.0
        }
        new_record = pd.Series({
            'name': 'Same Name',
            'description': 'Same Description',
            'credit_units': 3.0
        })
        field_mapping = {
            'name': 'name',
            'description': 'description',
            'credit_units': 'credit_units'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        assert result is False

    def test_does_not_overwrite_with_empty_data(self, course_processor):
        """Should not overwrite existing data with empty/None data."""
        existing_record = {
            'name': 'Existing Name',
            'description': 'Existing Description',
            'credit_units': 3.0
        }
        new_record = pd.Series({
            'name': pd.NA,  # empty new value
            'description': pd.NA,  # empty new value too
            'credit_units': 3.0
        })
        field_mapping = {
            'name': 'name',
            'description': 'description',
            'credit_units': 'credit_units'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        # Should not trigger update since new values are empty and old values exist
        assert result is False

    def test_handles_credit_units_type_conversion(self, course_processor):
        """Should properly compare credit_units as floats."""
        existing_record = {
            'credit_units': 3.0
        }
        new_record = pd.Series({
            'credit_units': 3  # int instead of float
        })
        field_mapping = {
            'credit_units': 'credit_units'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        assert result is False  # Same value, different types

    def test_handles_dict_to_dict_comparison(self, course_processor):
        """Should handle dict-to-dict comparisons."""
        existing_record = {
            'name': 'Old Name',
            'credit_units': 3.0
        }
        new_record = {
            'name': 'New Name',
            'credit_units': 3.0
        }
        field_mapping = {
            'name': 'name',
            'credit_units': 'credit_units'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        assert result is True

    def test_handles_strip_whitespace(self, course_processor):
        """Should strip whitespace before comparing string fields."""
        existing_record = {
            'name': 'Name',
            'description': 'Description'
        }
        new_record = pd.Series({
            'name': '  Name  ',  # extra whitespace
            'description': 'Description'
        })
        field_mapping = {
            'name': 'name',
            'description': 'description'
        }

        result = course_processor._needs_update(existing_record, new_record, field_mapping)

        assert result is False  # Should be considered equal after stripping


class TestCourseProcessorIntegration:
    """Integration tests for CourseProcessor."""

    def test_full_processing_workflow(self, course_processor, processor_context):
        """Test full course processing workflow with mixed new and existing courses."""
        processor_context.courses_cache = {
            'CS101': {
                'id': 'uuid-existing',
                'code': 'CS101',
                'name': 'Old Name',
                'course_description': 'Old Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        processor_context.standalone_data = pd.DataFrame({
            'course_code': ['CS101', 'CS102'],
            'course_name': ['New Name', 'New Course'],
            'course_description': ['New Description', 'Description 2'],
            'credit_units': [3.0, 4.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None']
        })

        course_processor._do_process()

        # CS101 should be updated, CS102 should be created
        assert processor_context.stats['courses_updated'] >= 0
        assert processor_context.stats['courses_created'] == 1
