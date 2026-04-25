"""
Unit tests for ClassProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.pipeline.processors.class_processor import ClassProcessor
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.course_dto import CourseDTO


class TestClassProcessor:
    """Tests for ClassProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_lookups(self):
        """Processor should initialize with empty lookups when not provided."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )
        assert processor._multiple_lookup == {}
        assert processor._course_lookup == {}

    def test_process_returns_tuple(self):
        """process() should return (new_classes, updated_classes) tuple."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[],
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestBuildExistingLookup:
    """Tests for _build_existing_lookup method."""

    def test_builds_lookup_from_cache(self):
        """_build_existing_lookup should build lookup from existing_classes_cache."""
        existing_cache = [
            {'acad_term_id': 'AY202526T1', 'boss_id': 1001, 'professor_id': 'prof1', 'id': 'class-1'},
            {'acad_term_id': 'AY202526T1', 'boss_id': 1001, 'professor_id': 'prof2', 'id': 'class-2'},
        ]

        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=existing_cache
        )

        processor._build_existing_lookup()

        assert ('AY202526T1', 1001, 'prof1') in processor._existing_class_lookup
        assert ('AY202526T1', 1001, 'prof2') in processor._existing_class_lookup
        assert processor._existing_class_lookup[('AY202526T1', 1001, 'prof1')]['id'] == 'class-1'

    def test_handles_missing_boss_id(self):
        """_build_existing_lookup should skip entries without boss_id."""
        existing_cache = [
            {'acad_term_id': 'AY202526T1', 'boss_id': None, 'professor_id': 'prof1', 'id': 'class-1'},
        ]

        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=existing_cache
        )

        processor._build_existing_lookup()

        assert len(processor._existing_class_lookup) == 0


class TestFindProfessorsForClass:
    """Tests for _find_professors_for_class method."""

    def test_returns_empty_when_no_record_key(self):
        """_find_professors_for_class should return empty list when record_key is None."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        result = processor._find_professors_for_class(None)
        assert result == []

    def test_returns_empty_when_record_key_not_in_lookup(self):
        """_find_professors_for_class should return empty list when key not in multiple_lookup."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        result = processor._find_professors_for_class('nonexistent_key')
        assert result == []

    def test_finds_professors_normalized(self):
        """_find_professors_for_class should find professors using normalized names."""
        professor_lookup = {
            'JOHN SMITH': 'prof-uuid-1',
            'JANE DOE': 'prof-uuid-2'
        }

        multiple_lookup = {
            'key1': [
                {'professor_name': 'John Smith'},
                {'professor_name': 'Jane Doe'}
            ]
        }

        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup=multiple_lookup,
            course_lookup={},
            professor_lookup=professor_lookup,
            existing_classes_cache=[]
        )

        result = processor._find_professors_for_class('key1')

        assert len(result) == 2
        professor_ids = [p[0] for p in result]
        assert 'prof-uuid-1' in professor_ids
        assert 'prof-uuid-2' in professor_ids

    def test_skips_missing_professor_names(self):
        """_find_professors_for_class should skip rows without professor_name."""
        professor_lookup = {
            'JOHN SMITH': 'prof-uuid-1',
        }

        multiple_lookup = {
            'key1': [
                {'professor_name': 'John Smith'},
                {'other_field': 'value'}
            ]
        }

        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup=multiple_lookup,
            course_lookup={},
            professor_lookup=professor_lookup,
            existing_classes_cache=[]
        )

        result = processor._find_professors_for_class('key1')

        assert len(result) == 1
        assert result[0][1] == 'John Smith'


class TestCompareValues:
    """Tests for _compare_values method."""

    def test_detects_change_in_values(self):
        """_compare_values should detect when values are different."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values('new_value', 'old_value')
        assert changed is True

    def test_detects_no_change_when_same(self):
        """_compare_values should detect when values are the same."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values('same_value', 'same_value')
        assert changed is False

    def test_handles_nan_values(self):
        """_compare_values should handle NaN values."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values(float('nan'), float('nan'))
        assert changed is False

    def test_handles_mixed_nan_and_value(self):
        """_compare_values should not flag as changed when new value is NaN (incoming empty)."""
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        # When new_value is NaN and old_value is 'some_value', should NOT change
        new_val, old_val, changed = processor._compare_values(float('nan'), 'some_value')
        assert changed is False


class TestProcessRow:
    """Tests for _process_row method."""

    def test_skips_row_without_acad_term_id(self):
        """_process_row should skip rows without acad_term_id."""
        raw_data = pd.DataFrame([{
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1'
        }])

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_lookup={},
            existing_classes_cache=[]
        )

        processor._process_row(raw_data.iloc[0])

        assert len(processor._new_classes) == 0
        assert len(processor._updated_classes) == 0

    def test_skips_row_without_class_boss_id(self):
        """_process_row should skip rows without class_boss_id."""
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'course_code': 'MGMT715',
            'section': 'G1'
        }])

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_lookup={},
            existing_classes_cache=[]
        )

        processor._process_row(raw_data.iloc[0])

        assert len(processor._new_classes) == 0

    def test_skips_row_when_course_not_found(self):
        """_process_row should skip rows when course not in course_lookup."""
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'NONEXISTENT',
            'section': 'G1'
        }])

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={},
            professor_lookup={},
            existing_classes_cache=[]
        )

        processor._process_row(raw_data.iloc[0])

        assert len(processor._new_classes) == 0


class TestClassDTO:
    """Tests for ClassDTO."""

    def test_creates_class_dto(self):
        """ClassDTO should be creatable with all fields."""
        now = datetime.now()
        dto = ClassDTO(
            id='class-uuid-123',
            section='G1',
            course_id='course-uuid-456',
            professor_id='prof-uuid-789',
            acad_term_id='AY202526T1',
            grading_basis='GRADED',
            course_outline_url='https://example.com/outline',
            boss_id=1001,
            warn_inaccuracy=False,
            created_at=now,
            updated_at=now
        )

        assert dto.id == 'class-uuid-123'
        assert dto.section == 'G1'
        assert dto.course_id == 'course-uuid-456'
        assert dto.acad_term_id == 'AY202526T1'
        assert dto.boss_id == 1001