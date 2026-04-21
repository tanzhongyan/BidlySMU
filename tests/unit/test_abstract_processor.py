"""
Unit tests for AbstractProcessor base class.
"""
import pytest
from unittest.mock import Mock

from src.pipeline.processors.abstract_processor import AbstractProcessor


class ConcreteTestProcessor(AbstractProcessor):
    """Concrete subclass of AbstractProcessor for testing."""

    def __init__(self, context=None, logger=None):
        super().__init__(logger)
        self._do_process_called = False

    def process(self):
        self._do_process_called = True


class TestAbstractProcessorInit:
    """Tests for AbstractProcessor.__init__."""

    def test_sets_logger_from_argument(self):
        """Should use logger argument when provided."""
        mock_logger = Mock()
        processor = ConcreteTestProcessor(logger=mock_logger)
        assert processor._logger is mock_logger

    def test_sets_default_logger_when_none(self):
        """Should create a default logger when none provided."""
        processor = ConcreteTestProcessor()
        assert processor._logger is not None

    def test_logger_named_after_module(self):
        """Logger should be named after the processor's module."""
        processor = ConcreteTestProcessor()
        assert processor._logger.name == ConcreteTestProcessor.__module__


class TestSafeInt:
    """Tests for AbstractProcessor.safe_int static method."""

    def test_returns_int_for_integer(self):
        assert AbstractProcessor.safe_int(42) == 42

    def test_returns_int_for_float_integer(self):
        assert AbstractProcessor.safe_int(42.0) == 42

    def test_returns_int_for_string_integer(self):
        assert AbstractProcessor.safe_int("123") == 123

    def test_returns_none_for_none(self):
        assert AbstractProcessor.safe_int(None) is None

    def test_returns_none_for_nan(self):
        import math
        assert AbstractProcessor.safe_int(float('nan')) is None
        assert AbstractProcessor.safe_int(math.nan) is None

    def test_returns_none_for_non_numeric_string(self):
        assert AbstractProcessor.safe_int("hello") is None

    def test_returns_none_for_complex(self):
        assert AbstractProcessor.safe_int(3+4j) is None


class TestSafeFloat:
    """Tests for AbstractProcessor.safe_float static method."""

    def test_returns_float_for_float(self):
        assert AbstractProcessor.safe_float(42.5) == 42.5

    def test_returns_float_for_integer(self):
        assert AbstractProcessor.safe_float(42) == 42.0

    def test_returns_float_for_string(self):
        assert AbstractProcessor.safe_float("123.5") == 123.5

    def test_returns_none_for_none(self):
        assert AbstractProcessor.safe_float(None) is None

    def test_returns_none_for_nan(self):
        import math
        assert AbstractProcessor.safe_float(float('nan')) is None
        assert AbstractProcessor.safe_float(math.nan) is None

    def test_returns_none_for_non_numeric_string(self):
        assert AbstractProcessor.safe_float("hello") is None


class TestProcessMethod:
    """Tests for the abstract process() method."""

    def test_process_is_abstract(self):
        """AbstractProcessor.process should be abstract."""
        with pytest.raises(TypeError):
            AbstractProcessor()

    def test_concrete_process_can_be_called(self):
        """Concrete subclass process() should be callable."""
        processor = ConcreteTestProcessor()
        processor.process()
        assert processor._do_process_called is True
