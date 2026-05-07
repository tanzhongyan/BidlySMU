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


class TestBuildExistingGroupLookup:
    """Tests for _build_existing_group_lookup method."""

    def test_builds_group_lookup_from_cache(self):
        """_build_existing_group_lookup should group existing classes by (acad_term_id, boss_id)."""
        existing_cache = [
            {'acad_term_id': 'AY202526T1', 'boss_id': 1001, 'professor_id': 'prof1', 'id': 'class-1'},
            {'acad_term_id': 'AY202526T1', 'boss_id': 1001, 'professor_id': 'prof2', 'id': 'class-2'},
            {'acad_term_id': 'AY202526T1', 'boss_id': 1002, 'professor_id': 'prof3', 'id': 'class-3'},
        ]

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache,
            logger=Mock()
        )

        processor._build_existing_group_lookup()

        # Both prof1 and prof2 should be under the same group key
        group_key = ('AY202526T1', 1001)
        assert group_key in processor._existing_classes_by_group
        assert len(processor._existing_classes_by_group[group_key]) == 2
        assert processor._existing_classes_by_group[group_key][0]['id'] == 'class-1'
        assert processor._existing_classes_by_group[group_key][1]['id'] == 'class-2'

        # prof3 should be in a separate group
        other_key = ('AY202526T1', 1002)
        assert other_key in processor._existing_classes_by_group
        assert len(processor._existing_classes_by_group[other_key]) == 1

    def test_handles_missing_boss_id(self):
        """_build_existing_group_lookup should skip entries without boss_id."""
        existing_cache = [
            {'acad_term_id': 'AY202526T1', 'boss_id': None, 'professor_id': 'prof1', 'id': 'class-1'},
        ]

        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=pd.DataFrame(),
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache,
            logger=Mock()
        )

        processor._build_existing_group_lookup()

        assert len(processor._existing_classes_by_group) == 0


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


class TestProcessEndToEnd:
    """End-to-end tests for class processing via process() method."""

    def _make_processor(self, raw_data, existing_cache, course_lookup=None, prof_lookup=None, multiple_lookup=None):
        """Helper to create a ClassProcessor with standard configuration."""
        mock_service = MockProfessorResolutionService(professor_lookup=prof_lookup or {})
        course_dto = course_lookup or {}
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup or {},
            course_lookup=course_dto,
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache,
            logger=Mock()
        )
        return processor

    def test_creates_new_class_when_no_existing(self):
        """Should CREATE a new class when no existing class matches."""
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1',
            'grading_basis': 'GRADED',
            'course_outline_url': None,
            'record_key': 'key-1',
        }])
        course_lookup = {'MGMT715': MagicMock(id='course-123')}
        multiple_lookup = {'key-1': [{'professor_name': 'JOHN DOE'}]}
        prof_lookup = {'JOHN DOE': 'prof-1'}

        processor = self._make_processor(raw_data, [], course_lookup, prof_lookup, multiple_lookup)
        new_classes, updated_classes = processor.process()

        assert len(new_classes) == 1
        assert new_classes[0].course_id == 'course-123'
        assert new_classes[0].professor_id == 'prof-1'

    def test_updates_existing_class_with_same_professor(self):
        """Should UPDATE an existing class when professor matches and section changes."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-1', 'section': 'G2', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': False,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
        ]
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1',
            'grading_basis': 'GRADED',
            'course_outline_url': None,
            'record_key': 'key-1',
        }])
        course_lookup = {'MGMT715': MagicMock(id='course-123')}
        multiple_lookup = {'key-1': [{'professor_name': 'JOHN DOE'}]}
        prof_lookup = {'JOHN DOE': 'prof-1'}

        processor = self._make_processor(raw_data, existing_cache, course_lookup, prof_lookup, multiple_lookup)
        new_classes, updated_classes = processor.process()

        assert len(new_classes) == 0
        assert len(updated_classes) == 1
        assert updated_classes[0].id == 'class-1'
        assert updated_classes[0].section == 'G1'

    def test_many_to_zero_professor_transition_keeps_one_tba(self):
        """When all professors are removed, exactly 1 class should remain with professor_id=None."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-1', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
            {'id': 'class-2', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-2', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
        ]
        # Incoming data with no professor (empty multiple_lookup for this key)
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1',
            'grading_basis': 'GRADED',
            'course_outline_url': None,
            'record_key': 'key-1',
        }])
        course_lookup = {'MGMT715': MagicMock(id='course-123')}
        # No professor resolution results → empty incoming_profs → TBA fallback
        multiple_lookup = {'key-1': []}

        processor = self._make_processor(raw_data, existing_cache, course_lookup, multiple_lookup=multiple_lookup)
        new_classes, updated_classes = processor.process()

        # 1 class should be updated to professor_id=None (repurposed as TBA)
        tba_updates = [u for u in updated_classes if u.professor_id is None]
        assert len(tba_updates) == 1

        # 1 class should be marked for deactivation
        deactivations = processor.get_classes_to_deactivate()
        assert len(deactivations) == 1


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
