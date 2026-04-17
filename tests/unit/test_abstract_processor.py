"""
Unit tests for abstract_processor module.
"""
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.db.database_helper import DatabaseHelper


class ConcreteTestProcessor(AbstractProcessor):
    """Concrete subclass of AbstractProcessor for testing the template method."""

    def __init__(self, context):
        super().__init__(context)
        self._do_process_called = False
        self._load_cache_called = False
        self._collect_results_called = False
        self._persist_called = False

    def process(self):
        """Template method under test."""
        self._load_cache()
        self._do_process()
        self._collect_results()
        self._persist()

    def _do_process(self):
        self._do_process_called = True

    def _load_cache(self):
        self._load_cache_called = True

    def _collect_results(self):
        self._collect_results_called = True

    def _persist(self):
        self._persist_called = True


class TestAbstractProcessor:
    """Tests for AbstractProcessor base class."""

    def setup_method(self):
        """Set up a mock context for each test."""
        self.mock_logger = Mock()
        self.mock_db_connection = Mock()
        self.context = Mock()
        self.context.logger = self.mock_logger
        self.context.db_connection = self.mock_db_connection

    def test_template_method_calls_steps_in_order(self):
        """process() should call load_cache, do_process, collect_results, persist in order."""
        processor = ConcreteTestProcessor(self.context)
        processor.process()

        assert processor._load_cache_called is True
        assert processor._do_process_called is True
        assert processor._collect_results_called is True
        assert processor._persist_called is True

    def test_context_sets_logger(self):
        """AbstractProcessor should set logger from context."""
        processor = ConcreteTestProcessor(self.context)
        assert processor._logger == self.mock_logger

    def test_context_sets_db_connection(self):
        """AbstractProcessor should store db_connection from context."""
        processor = ConcreteTestProcessor(self.context)
        assert processor.context.db_connection == self.mock_db_connection


class TestNeedsUpdate:
    """Tests for _needs_update() method."""

    def setup_method(self):
        """Set up a mock context for each test."""
        self.mock_logger = Mock()
        self.context = Mock()
        self.context.logger = self.mock_logger
        self.context.db_connection = Mock()
        self.processor = ConcreteTestProcessor(self.context)

    def test_returns_true_when_field_differs(self):
        """_needs_update() should return True when a field value differs."""
        existing_record = {'course_name': 'Math 101', 'professor_id': '123'}
        new_row = {'course_name': 'Math 102', 'professor_id': '123'}
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is True

    def test_returns_false_when_no_fields_differ(self):
        """_needs_update() should return False when all fields match."""
        existing_record = {'course_name': 'Math 101', 'professor_id': '123'}
        new_row = {'course_name': 'Math 101', 'professor_id': '123'}
        field_mapping = {'course_name': 'course_name', 'professor_id': 'professor_id'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is False

    def test_returns_false_when_new_value_is_none(self):
        """_needs_update() should return False when new value is None."""
        existing_record = {'course_name': 'Math 101'}
        new_row = {'course_name': None}
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is False

    def test_returns_false_when_new_value_is_nan(self):
        """_needs_update() should return False when new value is NaN."""
        existing_record = {'course_name': 'Math 101'}
        new_row = {'course_name': float('nan')}
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is False

    def test_returns_true_when_new_value_differs_from_existing(self):
        """_needs_update() should return True when new value differs from existing."""
        existing_record = {'course_name': 'Math 101'}
        new_row = {'course_name': 'Math 102'}
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is True

    def test_handles_dict_like_new_row_with_get(self):
        """_needs_update() should handle dict-like objects with get method."""
        existing_record = {'course_name': 'Math 101'}
        new_row = MagicMock()
        new_row.get.return_value = 'Math 102'
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is True

    def test_handles_new_row_without_get_method(self):
        """_needs_update() should return False when new_row has no get method."""
        existing_record = {'course_name': 'Math 101'}
        new_row = "not a dict"
        field_mapping = {'course_name': 'course_name'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is False

    def test_returns_true_when_multiple_fields_differ(self):
        """_needs_update() should return True when any field differs."""
        existing_record = {'course_name': 'Math 101', 'credits': 3}
        new_row = {'course_name': 'Math 102', 'credits': 4}
        field_mapping = {'course_name': 'course_name', 'credits': 'credits'}

        result = self.processor._needs_update(existing_record, new_row, field_mapping)
        assert result is True


class TestExecuteUpsert:
    """Tests for _execute_upsert() method."""

    def setup_method(self):
        """Set up a mock context for each test."""
        self.mock_logger = Mock()
        self.context = Mock()
        self.context.logger = self.mock_logger
        self.context.db_connection = Mock()
        self.processor = ConcreteTestProcessor(self.context)

    def test_does_nothing_when_records_empty(self, mocker):
        """_execute_upsert() should return early if records list is empty."""
        mock_upsert = mocker.patch.object(DatabaseHelper, 'upsert_df')
        self.processor._execute_upsert('test_table', [], ['id'])

        mock_upsert.assert_not_called()

    def test_calls_upsert_df_with_correct_arguments(self, mocker):
        """_execute_upsert() should call DatabaseHelper.upsert_df with correct args."""
        mock_upsert = mocker.patch.object(DatabaseHelper, 'upsert_df')

        records = [{'id': 1, 'name': 'Test'}]
        self.processor._execute_upsert('test_table', records, ['id'])

        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        assert call_kwargs['connection'] == self.context.db_connection
        assert isinstance(call_kwargs['df'], pd.DataFrame)
        assert call_kwargs['table_name'] == 'test_table'
        assert call_kwargs['index_elements'] == ['id']
        assert call_kwargs['logger'] == self.mock_logger

    def test_converts_records_to_dataframe(self, mocker):
        """_execute_upsert() should convert records list to DataFrame."""
        mock_upsert = mocker.patch.object(DatabaseHelper, 'upsert_df')

        records = [{'id': 1, 'name': 'Test'}, {'id': 2, 'name': 'Test2'}]
        self.processor._execute_upsert('test_table', records, ['id'])

        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        assert len(call_kwargs['df']) == 2
        assert list(call_kwargs['df']['id']) == [1, 2]
        assert list(call_kwargs['df']['name']) == ['Test', 'Test2']


class TestPersist:
    """Tests for _persist() method (default implementation)."""

    def setup_method(self):
        """Set up a mock context for each test."""
        self.mock_logger = Mock()
        self.context = Mock()
        self.context.logger = self.mock_logger
        self.context.db_connection = Mock()

    def test_default_persist_does_nothing(self):
        """Default _persist() should do nothing (can be overridden)."""
        processor = ConcreteTestProcessor(self.context)
        # Should not raise any exception
        processor._persist()
