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


class TestProfessorTransitions:
    """Tests for all professor transition types during class reconciliation."""

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

    def test_zero_to_one_professor_transition(self):
        """0→1: TBA class (professor_id=None) gets professor assigned."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': None, 'section': 'G1', 'course_id': 'course-123',
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

        # Existing class should be UPDATED with new professor
        assert len(updated_classes) == 1
        assert updated_classes[0].id == 'class-1'
        assert updated_classes[0].professor_id == 'prof-1'
        assert len(new_classes) == 0
        assert len(processor.get_classes_to_deactivate()) == 0

    def test_one_to_many_professor_transition(self):
        """1→N: Single-professor class gets additional professors (one record per professor).

        The existing class (prof-1) is matched but has no changes, so it stays as-is
        (not added to updated_classes). A new class is CREATEd for the second professor.
        No existing classes should be deactivated.
        """
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-1', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
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
        multiple_lookup = {'key-1': [{'professor_name': 'JOHN DOE'}, {'professor_name': 'JANE DOE'}]}
        prof_lookup = {'JOHN DOE': 'prof-1', 'JANE DOE': 'prof-2'}

        processor = self._make_processor(raw_data, existing_cache, course_lookup, prof_lookup, multiple_lookup)
        new_classes, updated_classes = processor.process()

        # 1 new class should be created for the second professor
        assert len(new_classes) == 1
        assert new_classes[0].professor_id == 'prof-2'
        # No classes should be deactivated
        assert len(processor.get_classes_to_deactivate()) == 0
        # New class for multi-professor group should have warn_inaccuracy=True
        for cls in new_classes:
            assert cls.warn_inaccuracy is True

    def test_professor_swap_transition(self):
        """Professor swap: existing class gets different professor assigned."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'old-prof', 'section': 'G1', 'course_id': 'course-123',
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
        multiple_lookup = {'key-1': [{'professor_name': 'NEW PROF'}]}
        prof_lookup = {'NEW PROF': 'new-prof'}

        processor = self._make_processor(raw_data, existing_cache, course_lookup, prof_lookup, multiple_lookup)
        new_classes, updated_classes = processor.process()

        # Existing class should be repurposed with new professor
        assert len(updated_classes) >= 1
        updated_prof_ids = [u.professor_id for u in updated_classes]
        assert 'new-prof' in updated_prof_ids
        # No deactivation needed (same number of professors)
        assert len(processor.get_classes_to_deactivate()) == 0

    def test_deactivation_marks_excess_records(self):
        """Excess existing classes should be marked for soft deactivation."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-1', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
            {'id': 'class-2', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-2', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
            {'id': 'class-3', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-3', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': True,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
        ]
        # Only 1 incoming professor → 2 excess classes should be deactivated
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

        # 2 excess classes should be deactivated
        deactivations = processor.get_classes_to_deactivate()
        assert len(deactivations) == 2
        deactivated_ids = {d['id'] for d in deactivations}
        # class-1 should NOT be in deactivations (it got repurposed)
        # Exactly 2 of the 3 should be deactivated
        assert len(deactivated_ids & {'class-1', 'class-2', 'class-3'}) == 2

    def test_course_not_found_skips_group(self):
        """Group should be skipped when course not found in lookup."""
        existing_cache = [
            {'id': 'class-1', 'acad_term_id': 'AY202526T1', 'boss_id': 1001,
             'professor_id': 'prof-1', 'section': 'G1', 'course_id': 'course-123',
             'grading_basis': 'GRADED', 'course_outline_url': None, 'warn_inaccuracy': False,
             'created_at': datetime.now(), 'updated_at': datetime.now()},
        ]
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'UNKNOWN_COURSE',
            'section': 'G1',
            'grading_basis': 'GRADED',
            'course_outline_url': None,
            'record_key': 'key-1',
        }])
        # Empty course_lookup - course not found
        course_lookup = {}
        multiple_lookup = {'key-1': [{'professor_name': 'JOHN DOE'}]}
        prof_lookup = {'JOHN DOE': 'prof-1'}

        mock_logger = Mock()
        mock_service = MockProfessorResolutionService(professor_lookup=prof_lookup)
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup,
            course_lookup=course_lookup,
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache,
            logger=mock_logger
        )
        new_classes, updated_classes = processor.process()

        # Should skip this group entirely
        assert len(new_classes) == 0
        assert len(updated_classes) == 0
        # Warning should be logged
        assert mock_logger.warning.called


class TestBuildIncomingState:
    """Tests for _build_incoming_state method."""

    def test_identifies_multi_professor_groups(self):
        """Should correctly identify groups with multiple professors."""
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1',
            'record_key': 'key-1',
        }])
        mock_service = MockProfessorResolutionService(professor_lookup={'JOHN DOE': 'prof-1', 'JANE DOE': 'prof-2'})
        multiple_lookup = {'key-1': [{'professor_name': 'JOHN DOE'}, {'professor_name': 'JANE DOE'}]}

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup,
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_resolution_service=mock_service,
            existing_classes_cache=[],
            logger=Mock()
        )
        processor._build_incoming_state()

        # Group should be identified as multi-professor
        assert ('AY202526T1', 1001) in processor._multi_professor_groups

    def test_deduplicates_professor_ids_within_group(self):
        """Should deduplicate incoming professors by professor_id within a group."""
        raw_data = pd.DataFrame([
            {'acad_term_id': 'AY202526T1', 'class_boss_id': 1001, 'course_code': 'MGMT715',
             'section': 'G1', 'record_key': 'key-1'},
            {'acad_term_id': 'AY202526T1', 'class_boss_id': 1001, 'course_code': 'MGMT715',
             'section': 'G1', 'record_key': 'key-2'},
        ])
        mock_service = MockProfessorResolutionService(professor_lookup={'JOHN DOE': 'prof-1'})
        multiple_lookup = {
            'key-1': [{'professor_name': 'JOHN DOE'}],
            'key-2': [{'professor_name': 'JOHN DOE'}]  # Same professor again
        }

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup,
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_resolution_service=mock_service,
            existing_classes_cache=[],
            logger=Mock()
        )
        processor._build_incoming_state()

        # Same professor should only appear once in incoming state
        group_key = ('AY202526T1', 1001)
        incoming_profs = processor._incoming_state_by_group.get(group_key, [])
        prof_ids = [p[0] for p in incoming_profs]
        assert prof_ids.count('prof-1') == 1

    def test_skips_rows_with_missing_keys(self):
        """Should skip rows with NaN acad_term_id or boss_id."""
        raw_data = pd.DataFrame([
            {'acad_term_id': None, 'class_boss_id': 1001, 'course_code': 'MGMT715',
             'section': 'G1', 'record_key': 'key-1'},
            {'acad_term_id': 'AY202526T1', 'class_boss_id': None, 'course_code': 'MGMT715',
             'section': 'G1', 'record_key': 'key-2'},
        ])
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[],
            logger=Mock()
        )
        processor._build_incoming_state()

        # No valid groups should be created
        assert len(processor._incoming_state_by_group) == 0

    def test_tba_fallback_when_no_professors(self):
        """Should add (None, '') TBA entry when no professors resolved."""
        raw_data = pd.DataFrame([{
            'acad_term_id': 'AY202526T1',
            'class_boss_id': 1001,
            'course_code': 'MGMT715',
            'section': 'G1',
            'record_key': 'key-1',
        }])
        mock_service = MockProfessorResolutionService()
        # Empty multiple_lookup → no professors
        multiple_lookup = {'key-1': []}

        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup,
            course_lookup={'MGMT715': MagicMock(id='course-123')},
            professor_resolution_service=mock_service,
            existing_classes_cache=[],
            logger=Mock()
        )
        processor._build_incoming_state()

        group_key = ('AY202526T1', 1001)
        incoming_profs = processor._incoming_state_by_group.get(group_key, [])
        # Should have TBA entry
        assert (None, '') in incoming_profs


class TestRecordKeyMapping:
    """Tests for record_key → class_ids mapping built during processing."""

    def _make_processor(self, raw_data, existing_cache, course_lookup=None, prof_lookup=None, multiple_lookup=None):
        mock_service = MockProfessorResolutionService(professor_lookup=prof_lookup or {})
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup=multiple_lookup or {},
            course_lookup=course_lookup or {},
            professor_resolution_service=mock_service,
            existing_classes_cache=existing_cache,
            logger=Mock()
        )
        return processor

    def test_mapping_built_during_processing(self):
        """record_key_to_class_ids should be populated during process()."""
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
        processor.process()

        mapping = processor.get_record_key_to_class_ids_mapping()
        assert 'key-1' in mapping
        assert len(mapping['key-1']) >= 1

    def test_mapping_accessible_via_getter(self):
        """get_record_key_to_class_ids_mapping() should return the mapping dict."""
        raw_data = pd.DataFrame()
        mock_service = MockProfessorResolutionService()
        processor = ClassProcessor(
            raw_data=raw_data,
            multiple_lookup={},
            course_lookup={},
            professor_resolution_service=mock_service,
            existing_classes_cache=[],
            logger=Mock()
        )

        mapping = processor.get_record_key_to_class_ids_mapping()
        assert isinstance(mapping, dict)


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

    def test_to_csv_row_with_all_fields(self):
        """to_csv_row should return dict with all fields."""
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

        row = dto.to_csv_row()
        assert row['id'] == 'class-uuid-123'
        assert row['section'] == 'G1'
        assert row['course_id'] == 'course-uuid-456'
        assert row['acad_term_id'] == 'AY202526T1'
        assert row['boss_id'] == 1001

    def test_to_db_row_with_all_fields(self):
        """to_db_row should return dict with all fields."""
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

        row = dto.to_db_row()
        assert row['id'] == 'class-uuid-123'
        assert row['section'] == 'G1'
        assert row['professor_id'] == 'prof-uuid-789'

    def test_none_professor_id_serialization(self):
        """ClassDTO with None professor_id should serialize correctly."""
        now = datetime.now()
        dto = ClassDTO(
            id='class-uuid-tba',
            section='G1',
            course_id='course-uuid-456',
            professor_id=None,
            acad_term_id='AY202526T1',
            grading_basis='GRADED',
            course_outline_url=None,
            boss_id=1001,
            warn_inaccuracy=False,
            created_at=now,
            updated_at=now
        )

        csv_row = dto.to_csv_row()
        assert csv_row['professor_id'] is None

        db_row = dto.to_db_row()
        assert db_row['professor_id'] is None
