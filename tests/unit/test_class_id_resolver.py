"""
Unit tests for find_all_class_ids method on AbstractProcessor.
"""
import numpy as np
import pandas as pd

from src.pipeline.processors.abstract_processor import AbstractProcessor


class TestProcessor(AbstractProcessor):
    """Concrete processor for testing."""
    def process(self):
        pass


class TestFindAllClassIds:
    """Tests for AbstractProcessor.find_all_class_ids()."""

    def setup_method(self):
        self.processor = TestProcessor()

    # ----- NaN/None input tests -----

    def test_returns_empty_list_for_nan_acad_term_id(self):
        result = self.processor.find_all_class_ids(
            acad_term_id=float('nan'),
            class_boss_id="BOSS001",
            new_classes=[],
            existing_classes_cache=[]
        )
        assert result == []

    def test_returns_empty_list_for_np_nan_acad_term_id(self):
        result = self.processor.find_all_class_ids(
            acad_term_id=np.nan,
            class_boss_id="BOSS001",
            new_classes=[],
            existing_classes_cache=[]
        )
        assert result == []

    def test_returns_empty_list_for_nan_class_boss_id(self):
        result = self.processor.find_all_class_ids(
            acad_term_id="2025-26_T1",
            class_boss_id=float('nan'),
            new_classes=[],
            existing_classes_cache=[]
        )
        assert result == []

    def test_returns_empty_list_for_none_acad_term_id(self):
        result = self.processor.find_all_class_ids(
            acad_term_id=None,
            class_boss_id="BOSS001",
            new_classes=[],
            existing_classes_cache=[]
        )
        assert result == []

    def test_returns_empty_list_for_none_class_boss_id(self):
        result = self.processor.find_all_class_ids(
            acad_term_id="2025-26_T1",
            class_boss_id=None,
            new_classes=[],
            existing_classes_cache=[]
        )
        assert result == []

    # ----- No matches tests -----

    def test_returns_empty_list_when_no_matches_any_source(self):
        new_classes = [
            {'id': 1, 'acad_term_id': '2024-25_T1', 'boss_id': 'BOSS001'},
        ]
        existing_classes_cache = [
            {'id': 2, 'acad_term_id': '2024-25_T1', 'boss_id': 'BOSS002'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS999',
            new_classes=new_classes,
            existing_classes_cache=existing_classes_cache
        )
        assert result == []

    def test_returns_empty_list_when_only_new_classes_provided_and_no_match(self):
        new_classes = [
            {'id': 1, 'acad_term_id': '2024-25_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == []

    def test_returns_empty_list_when_only_existing_cache_provided_and_no_match(self):
        existing_cache = [
            {'id': 1, 'acad_term_id': '2024-25_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=[],
            existing_classes_cache=existing_cache
        )
        assert result == []

    # ----- Match in new_classes tests -----

    def test_finds_match_in_new_classes(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == [100]

    def test_finds_multiple_matches_in_new_classes(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 101, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 102, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == [100, 101, 102]

    def test_new_classes_boss_id_matching_with_different_types(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 123},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='123',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == [100]

    # ----- Match in existing_classes_cache tests -----

    def test_finds_match_in_existing_classes_cache(self):
        existing_cache = [
            {'id': 200, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS002'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS002',
            new_classes=[],
            existing_classes_cache=existing_cache
        )
        assert result == [200]

    def test_finds_multiple_matches_in_existing_classes_cache(self):
        existing_cache = [
            {'id': 200, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 201, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=[],
            existing_classes_cache=existing_cache
        )
        assert result == [200, 201]

    # ----- Combined sources tests -----

    def test_combines_results_from_both_sources(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        existing_cache = [
            {'id': 200, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=existing_cache
        )
        assert result == [100, 200]

    def test_deduplicates_when_class_id_in_both_sources(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        existing_cache = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=existing_cache
        )
        assert result == [100]

    # ----- Multi-professor class tests -----

    def test_multi_professor_returns_all_matching_class_ids(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 101, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 102, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        existing_cache = [
            {'id': 200, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 201, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=existing_cache
        )
        assert result == [100, 101, 102, 200, 201]

    # ----- Edge cases -----

    def test_handles_empty_new_classes_list(self):
        existing_cache = [
            {'id': 200, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=[],
            existing_classes_cache=existing_cache
        )
        assert result == [200]

    def test_handles_empty_existing_cache_list(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == [100]

    def test_boss_id_matching_is_case_sensitive(self):
        new_classes = [
            {'id': 100, 'acad_term_id': '2025-26_T1', 'boss_id': 'boss001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=[]
        )
        assert result == []

    def test_preserves_order_from_multiple_sources(self):
        new_classes = [
            {'id': 1, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 3, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        existing_cache = [
            {'id': 2, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
            {'id': 4, 'acad_term_id': '2025-26_T1', 'boss_id': 'BOSS001'},
        ]
        result = self.processor.find_all_class_ids(
            acad_term_id='2025-26_T1',
            class_boss_id='BOSS001',
            new_classes=new_classes,
            existing_classes_cache=existing_cache
        )
        assert result == [1, 3, 2, 4]
