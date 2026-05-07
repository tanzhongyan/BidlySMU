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
from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
from src.pipeline.dtos.professor_dto import ProfessorDTO


class MockProfessorResolutionService:
    """Mock implementation of ProfessorResolutionService for testing."""

    def __init__(self, professor_lookup: dict = None):
        self._professor_lookup = professor_lookup or {}
        self._resolved = []

    def resolve_professor_ids(self, record_key: str, multiple_rows: list) -> list:
        """Return professor IDs from pre-configured lookup."""
        if not record_key:
            return []
        rows = multiple_rows or []
        results = []
        for row in rows:
            prof_name = row.get('professor_name', '')
            upper_name = prof_name.upper() if prof_name else ''
            if upper_name in self._professor_lookup:
                prof_id = self._professor_lookup[upper_name]
                if prof_id not in [r[0] for r in results]:
                    results.append((prof_id, prof_name))
        return results

    def resolve_professor_name(self, name: str):
        upper = name.upper() if name else ''
        return self._professor_lookup.get(upper)


class TestClassProcessor:
    """Tests for ClassProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_lookups(self):
        """Processor should initialize with empty lookups when not provided."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )
        assert processor._multiple_lookup == {}
        assert processor._course_lookup == {}

    def test_process_returns_tuple(self):
        """process() should return (new_classes, updated_classes) tuple."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
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

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
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

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache
        )

        processor._build_existing_lookup()

        assert len(processor._existing_class_lookup) == 0


class TestCompareValues:
    """Tests for _compare_values method."""

    def test_detects_change_in_values(self):
        """_compare_values should detect when values are different."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values('new_value', 'old_value')
        assert changed is True

    def test_detects_no_change_when_same(self):
        """_compare_values should detect when values are the same."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values('same_value', 'same_value')
        assert changed is False

    def test_handles_nan_values(self):
        """_compare_values should handle NaN values."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )

        new_val, old_val, changed = processor._compare_values(float('nan'), float('nan'))
        assert changed is False

    def test_handles_mixed_nan_and_value(self):
        """_compare_values should not flag as changed when new value is NaN (incoming empty)."""
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[]
        )

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

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_resolution_service=mock_service,
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

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_resolution_service=mock_service,
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

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
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