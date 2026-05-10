"""
Unit tests for SafetyFactorProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import os

from src.pipeline.processors.safety_factor_processor import SafetyFactorProcessor
from src.pipeline.dtos.bid_prediction_dto import SafetyFactorDTO


class TestSafetyFactorProcessor:
    """Tests for SafetyFactorProcessor."""

    def test_requires_expected_acad_term_id(self):
        """Processor should require expected_acad_term_id parameter."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1'
        )
        assert processor._expected_acad_term_id == 'AY202526T1'

    def test_has_default_cache_dir(self):
        """Processor should have default cache_dir of 'db_cache'."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1'
        )
        assert processor._cache_dir == 'db_cache'

    def test_initializes_with_custom_cache_dir(self):
        """Processor should accept custom cache_dir."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            cache_dir='custom_cache'
        )
        assert processor._cache_dir == 'custom_cache'

    def test_process_returns_list(self):
        """process() should return a list of SafetyFactorDTOs."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, list)


class TestAlreadyExists:
    """Tests for _already_exists method."""

    def test_returns_false_when_cache_not_exists(self, tmp_path):
        """_already_exists should return False when cache file doesn't exist."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            cache_dir=str(tmp_path)
        )

        result = processor._already_exists()

        assert result is False

    def test_returns_true_when_cache_exists(self, tmp_path):
        """_already_exists should return True when cache file exists."""
        # Create the cache file
        cache_path = tmp_path / 'safety_factors_cache.pkl'
        cache_path.write_text('dummy')

        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            cache_dir=str(tmp_path)
        )

        result = processor._already_exists()

        assert result is True


class TestProcessSkipsWhenExists:
    """Tests for process() skipping when cache exists."""

    @patch.object(SafetyFactorProcessor, '_already_exists', return_value=True)
    def test_process_returns_empty_when_cache_exists(self, mock_exists):
        """process() should return empty list when safety factors already exist."""
        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            logger=Mock()
        )

        result = processor.process()

        assert result == []
        mock_exists.assert_called_once()


class TestSafetyFactorDTO:
    """Tests for SafetyFactorDTO."""

    def test_creates_safety_factor_dto(self):
        """SafetyFactorDTO should be creatable with all fields."""
        dto = SafetyFactorDTO(
            acad_term_id='AY202526T1',
            prediction_type='median',
            beats_percentage=90,
            multiplier=1.5,
            multiplier_type='empirical'
        )

        assert dto.acad_term_id == 'AY202526T1'
        assert dto.prediction_type == 'median'
        assert dto.beats_percentage == 90
        assert dto.multiplier == 1.5
        assert dto.multiplier_type == 'empirical'

    def test_creates_from_row(self):
        """SafetyFactorDTO.from_row should create DTO from row data."""
        row = {
            'acad_term_id': 'AY202526T1',
            'prediction_type': 'min',
            'beats_percentage': 95,
            'multiplier': 2.0,
            'multiplier_type': 'theoretical'
        }

        dto = SafetyFactorDTO.from_row(**row)

        assert dto.acad_term_id == 'AY202526T1'
        assert dto.prediction_type == 'min'
        assert dto.beats_percentage == 95
        assert dto.multiplier == 2.0
        assert dto.multiplier_type == 'theoretical'


class TestProcessWithCalculator:
    """Tests for process() with SafetyFactorCalculator."""

    @patch('src.pipeline.processors.safety_factor_processor.SafetyFactorCalculator')
    @patch.object(SafetyFactorProcessor, '_already_exists', return_value=False)
    def test_process_calls_calculator(self, mock_exists, mock_calculator_class):
        """process() should call SafetyFactorCalculator when cache doesn't exist."""
        mock_calculator = MagicMock()
        mock_calculator.create_safety_factor_table.return_value = pd.DataFrame({
            'acad_term_id': ['AY202526T1'],
            'prediction_type': ['median'],
            'beats_percentage': [90],
            'multiplier': [1.5],
            'multiplier_type': ['empirical']
        })
        mock_calculator_class.return_value = mock_calculator

        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            logger=Mock()
        )

        result = processor.process()

        assert len(result) == 1
        assert result[0].prediction_type == 'median'
        mock_calculator.create_safety_factor_table.assert_called_once_with('AY202526T1')

    @patch('src.pipeline.processors.safety_factor_processor.SafetyFactorCalculator')
    @patch.object(SafetyFactorProcessor, '_already_exists', return_value=False)
    def test_process_returns_empty_when_calculator_returns_empty(self, mock_exists, mock_calculator_class):
        """process() should return empty list when calculator returns empty DataFrame."""
        mock_calculator = MagicMock()
        mock_calculator.create_safety_factor_table.return_value = pd.DataFrame()
        mock_calculator_class.return_value = mock_calculator

        processor = SafetyFactorProcessor(
            expected_acad_term_id='AY202526T1',
            logger=Mock()
        )

        result = processor.process()

        assert result == []