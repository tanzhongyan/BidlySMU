"""
Unit tests for CourseProcessor - class-based processor.
"""
import pytest
import pandas as pd

from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.dtos.course_dto import CourseDTO


class TestBuildFacultyResolutionIndex:
    """Tests for _build_faculty_resolution_index()."""

    def test_extracts_prefix_from_course_code(self):
        """Should extract prefix and map to faculty."""
        courses_cache = {
            'MGMT715': {'code': 'MGMT715', 'belong_to_faculty': 1},
            'MGMT701': {'code': 'MGMT701', 'belong_to_faculty': 1},
            'LAW4087': {'code': 'LAW4087', 'belong_to_faculty': 2}
        }
        faculties_cache = {1: {'id': 1}, 2: {'id': 2}}

        processor = CourseProcessor(pd.DataFrame(), courses_cache, faculties_cache)
        processor._build_faculty_resolution_index()

        assert processor._prefix_faculty_index['MGMT'] == 1
        assert processor._prefix_faculty_index['LAW'] == 2

    def test_picks_most_common_faculty_for_prefix(self):
        """Should pick most common faculty when multiple exist for prefix."""
        courses_cache = {
            'MGMT715': {'code': 'MGMT715', 'belong_to_faculty': 1},
            'MGMT701': {'code': 'MGMT701', 'belong_to_faculty': 1},
            'MGMT600': {'code': 'MGMT600', 'belong_to_faculty': 3}
        }
        faculties_cache = {1: {'id': 1}, 3: {'id': 3}}

        processor = CourseProcessor(pd.DataFrame(), courses_cache, faculties_cache)
        processor._build_faculty_resolution_index()

        assert processor._prefix_faculty_index['MGMT'] == 1

    def test_handles_hyphenated_prefix(self):
        """Should handle hyphenated prefixes like COR-COMM."""
        courses_cache = {
            'COR-COMM175': {'code': 'COR-COMM175', 'belong_to_faculty': 4}
        }
        faculties_cache = {4: {'id': 4}}

        processor = CourseProcessor(pd.DataFrame(), courses_cache, faculties_cache)
        processor._build_faculty_resolution_index()

        assert processor._prefix_faculty_index['COR-COMM'] == 4


class TestDetermineFacultyForCourse:
    """Tests for _determine_faculty_for_course()."""

    def test_returns_fallback_for_unknown_prefix(self):
        """Should return fallback faculty for unknown prefix."""
        processor = CourseProcessor(pd.DataFrame(), {}, {1: {'id': 1}})
        processor._prefix_faculty_index = {'MGMT': 1, 'LAW': 2}
        processor._fallback_faculty_id = 1

        result = processor._determine_faculty_for_course('XYZ999')

        assert result == 1

    def test_returns_mapped_faculty_for_known_prefix(self):
        """Should return mapped faculty for known prefix."""
        processor = CourseProcessor(pd.DataFrame(), {}, {1: {'id': 1}})
        processor._prefix_faculty_index = {'MGMT': 1, 'LAW': 2}
        processor._fallback_faculty_id = 1

        result = processor._determine_faculty_for_course('MGMT715')

        assert result == 1

    def test_handles_numeric_after_prefix(self):
        """Should handle course codes with numbers after prefix."""
        processor = CourseProcessor(pd.DataFrame(), {}, {1: {'id': 1}})
        processor._prefix_faculty_index = {'ESM': 3}
        processor._fallback_faculty_id = 1

        result = processor._determine_faculty_for_course('ESM105')

        assert result == 3


class TestNeedsUpdate:
    """Tests for _needs_update()."""

    def test_returns_false_when_no_changes(self):
        """Should return False when no fields have changed."""
        existing = {
            'name': 'Same Name',
            'description': 'Same Description',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        }
        new_row = pd.Series({
            'course_name': 'Same Name',
            'course_description': 'Same Description',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })
        processor = CourseProcessor(pd.DataFrame(), {}, {})
        result = processor._needs_update(existing, new_row)
        assert result is False

    def test_returns_true_when_name_changed(self):
        """Should return True when name has changed."""
        existing = {'name': 'Old Name', 'description': 'Desc', 'credit_units': 3.0, 'course_area': None, 'enrolment_requirements': None}
        new_row = pd.Series({'course_name': 'New Name', 'course_description': 'Desc', 'credit_units': 3.0, 'course_area': None, 'enrolment_requirements': None})
        processor = CourseProcessor(pd.DataFrame(), {}, {})
        result = processor._needs_update(existing, new_row)
        assert result is True

    def test_returns_true_when_description_changed(self):
        """Should return True when description has changed."""
        existing = {'name': 'Name', 'description': 'Old Desc', 'credit_units': 3.0, 'course_area': None, 'enrolment_requirements': None}
        new_row = pd.Series({'course_name': 'Name', 'course_description': 'New Desc', 'credit_units': 3.0, 'course_area': None, 'enrolment_requirements': None})
        processor = CourseProcessor(pd.DataFrame(), {}, {})
        result = processor._needs_update(existing, new_row)
        assert result is True

    def test_returns_true_when_na_in_new(self):
        """Should return True when new value is NA but old value exists."""
        existing = {'name': 'Name', 'description': 'Desc', 'credit_units': 3.0, 'course_area': 'Area', 'enrolment_requirements': None}
        new_row = pd.Series({'course_name': 'Name', 'course_description': 'Desc', 'credit_units': 3.0, 'course_area': pd.NA, 'enrolment_requirements': None})
        processor = CourseProcessor(pd.DataFrame(), {}, {})
        result = processor._needs_update(existing, new_row)
        assert result is True


class TestProcessCourses:
    """Tests for process() method."""

    def test_creates_new_courses(self):
        """Should create new courses when not in cache."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101', 'CS102'],
            'course_name': ['Intro to CS', 'Data Structures'],
            'course_description': ['Description 1', 'Description 2'],
            'credit_units': [3.0, 4.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'CS101']
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 2
        assert len(updated_courses) == 0
        assert all(isinstance(c, CourseDTO) for c in new_courses)
        assert new_courses[0].code == 'CS101'
        assert new_courses[0].updated_at is None  # CREATE should not set updated_at
        assert new_courses[1].code == 'CS102'

    def test_skips_duplicate_course_codes(self):
        """Should skip rows with duplicate course codes."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101', 'CS101'],
            'course_name': ['Intro to CS', 'Intro to CS Duplicate'],
            'course_description': ['Description 1', 'Description 2'],
            'credit_units': [3.0, 3.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None']
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 1

    def test_skips_na_course_codes(self):
        """Should skip rows with NaN course codes."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101', pd.NA, None],
            'course_name': ['Intro to CS', 'Course 2', 'Course 3'],
            'course_description': ['Description 1', 'Description 2', 'Description 3'],
            'credit_units': [3.0, 3.0, 3.0],
            'course_area': ['Computing', 'Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None', 'None']
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 1

    def test_updates_existing_course(self):
        """Should update course when it exists in cache and has changes."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': ['New Name'],
            'course_description': ['New Description'],
            'credit_units': [3.0],
            'course_area': ['Computing'],
            'enrolment_requirements': ['None']
        })
        courses_cache = {
            'CS101': {
                'id': 'existing-uuid',
                'code': 'CS101',
                'name': 'Old Name',
                'description': 'Old Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 0
        assert len(updated_courses) == 1
        assert updated_courses[0].id == 'existing-uuid'
        assert updated_courses[0].name == 'New Name'
        assert updated_courses[0].updated_at is not None  # updated_at should be set on UPDATE

    def test_no_update_when_no_changes(self):
        """Should not update when existing course has same values."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': ['Same Name'],
            'course_description': ['Same Description'],
            'credit_units': [3.0],
            'course_area': ['Computing'],
            'enrolment_requirements': ['None']
        })
        courses_cache = {
            'CS101': {
                'id': 'existing-uuid',
                'code': 'CS101',
                'name': 'Same Name',
                'description': 'Same Description',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 0
        assert len(updated_courses) == 0

    def test_uses_fallback_faculty_for_unknown_prefix(self):
        """Should use fallback faculty when prefix not in index."""
        raw_data = pd.DataFrame({
            'course_code': ['XYZ999'],
            'course_name': ['Unknown Course'],
            'course_description': ['Description'],
            'credit_units': [3.0],
            'course_area': [None],
            'enrolment_requirements': [None]
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, _ = processor.process()

        assert len(new_courses) == 1
        assert new_courses[0].belong_to_faculty == 1

    def test_generates_valid_uuid_for_new_courses(self):
        """Should generate valid UUID for new course."""
        import uuid
        raw_data = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': ['Intro to CS'],
            'course_description': ['Description'],
            'credit_units': [3.0],
            'course_area': ['Computing'],
            'enrolment_requirements': ['None']
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, _ = processor.process()

        uuid.UUID(new_courses[0].id)

    def test_handles_defaults_for_missing_optional_fields(self):
        """Should use defaults when optional fields are missing."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101'],
            'course_name': [pd.NA],
            'course_description': [pd.NA],
            'credit_units': [pd.NA],
            'course_area': [pd.NA],
            'enrolment_requirements': [pd.NA]
        })
        courses_cache = {}
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, _ = processor.process()

        assert len(new_courses) == 1
        assert new_courses[0].name == 'Unknown Course'
        assert new_courses[0].description == 'No description available'
        assert new_courses[0].credit_units == 1.0

    def test_mixed_new_and_updated_courses(self):
        """Should handle mix of new and updated courses."""
        raw_data = pd.DataFrame({
            'course_code': ['CS101', 'CS102'],
            'course_name': ['New Name for CS101', 'Intro to CS102'],
            'course_description': ['New Desc', 'Description 2'],
            'credit_units': [3.0, 4.0],
            'course_area': ['Computing', 'Computing'],
            'enrolment_requirements': ['None', 'None']
        })
        courses_cache = {
            'CS101': {
                'id': 'uuid-1',
                'code': 'CS101',
                'name': 'Old Name',
                'description': 'Old Desc',
                'credit_units': 3.0,
                'course_area': 'Computing',
                'enrolment_requirements': 'None'
            }
        }
        faculties_cache = {1: {'id': 1, 'name': 'Computing'}}

        processor = CourseProcessor(raw_data, courses_cache, faculties_cache)
        new_courses, updated_courses = processor.process()

        assert len(new_courses) == 1
        assert new_courses[0].code == 'CS102'
        assert len(updated_courses) == 1
        assert updated_courses[0].code == 'CS101'
        assert updated_courses[0].name == 'New Name for CS101'


class TestCourseDTO:
    """Tests for CourseDTO."""

    def test_to_csv_row(self):
        """Should return correct CSV row."""
        dto = CourseDTO(
            id='test-uuid',
            code='CS101',
            name='Intro to CS',
            description='Description',
            credit_units=3.0,
            belong_to_university=1,
            belong_to_faculty=2,
            course_area='Computing',
            enrolment_requirements=None
        )
        row = dto.to_csv_row()
        assert row['id'] == 'test-uuid'
        assert row['code'] == 'CS101'
        assert row['name'] == 'Intro to CS'
        assert row['credit_units'] == 3.0

    def test_to_db_row(self):
        """Should return correct DB row."""
        dto = CourseDTO(
            id='test-uuid',
            code='CS101',
            name='Intro to CS',
            description='Description',
            credit_units=3.0,
            belong_to_university=1,
            belong_to_faculty=2,
            course_area='Computing',
            enrolment_requirements=None
        )
        row = dto.to_db_row()
        assert row['id'] == 'test-uuid'
        assert row['code'] == 'CS101'

    def test_from_row(self):
        """Should create DTO from DataFrame row."""
        row = pd.Series({
            'course_code': 'CS101',
            'course_name': 'Intro to CS',
            'course_description': 'Description',
            'credit_units': 3.0,
            'course_area': 'Computing',
            'enrolment_requirements': 'None'
        })

        dto = CourseDTO.from_row(row, faculty_id=2)

        assert dto.code == 'CS101'
        assert dto.name == 'Intro to CS'
        assert dto.belong_to_faculty == 2
        assert dto.belong_to_university == 1