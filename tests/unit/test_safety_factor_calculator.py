"""
Unit tests for SafetyFactorCalculator in src/pipeline/safety_factor_calculator.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.pipeline.safety_factor_calculator import SafetyFactorCalculator


class TestSafetyFactorCalculatorInit:
    """Tests for SafetyFactorCalculator initialization."""

    def test_init_accepts_logger(self):
        """Test that __init__ accepts and stores logger."""
        mock_logger = Mock()
        calculator = SafetyFactorCalculator(logger=mock_logger)
        assert calculator._logger == mock_logger


class TestFitTDistribution:
    """Tests for fit_t_distribution() method."""

    @pytest.fixture
    def calculator(self, mocker):
        """Create a SafetyFactorCalculator instance."""
        mock_logger = mocker.Mock()
        return SafetyFactorCalculator(logger=mock_logger)

    def test_fit_t_distribution_returns_tuple(self, calculator):
        """Test that fit_t_distribution() returns (df, loc, scale, params) tuple."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculator.fit_t_distribution(errors)

        assert isinstance(result, tuple)
        assert len(result) == 4
        df, loc, scale, params = result
        assert isinstance(df, (float, int))
        assert isinstance(loc, (float, int))
        assert isinstance(scale, (float, int))
        assert isinstance(params, tuple)
        assert len(params) == 3  # stats.t.fit returns (df, loc, scale)

    def test_fit_t_distribution_with_zeros(self, calculator):
        """Test fit_t_distribution() with all zeros."""
        errors = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = calculator.fit_t_distribution(errors)

        df, loc, scale, params = result
        # With zero variance, scale should be 0 or near-zero
        assert scale >= 0

    def test_fit_t_distribution_with_negative_values(self, calculator):
        """Test fit_t_distribution() with negative error values."""
        errors = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = calculator.fit_t_distribution(errors)

        df, loc, scale, params = result
        assert isinstance(df, (float, int))
        assert isinstance(scale, (float, int))

    def test_fit_t_distribution_filters_nonfinite(self, calculator):
        """Test that fit_t_distribution() filters out non-finite values."""
        errors = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])
        df, loc, scale, params = calculator.fit_t_distribution(errors)

        # Should only use [1.0, 2.0, 3.0]
        assert np.isfinite(df)
        assert np.isfinite(loc)
        assert np.isfinite(scale)


class TestCalculatePercentileMultipliers:
    """Tests for calculate_percentile_multipliers() method."""

    @pytest.fixture
    def calculator(self, mocker):
        """Create a SafetyFactorCalculator instance."""
        mock_logger = mocker.Mock()
        return SafetyFactorCalculator(logger=mock_logger)

    def test_calculate_percentile_multipliers_returns_list(self, calculator):
        """Test that calculate_percentile_multipliers() returns a list."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df, loc, scale, _ = calculator.fit_t_distribution(errors)

        result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "median")

        assert isinstance(result, list)
        assert len(result) == 99  # percentiles 1-99

    def test_calculate_percentile_multipliers_correct_structure(self, calculator):
        """Test that each result entry has correct structure."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        df, loc, scale, _ = calculator.fit_t_distribution(errors)

        result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "median")

        first_entry = result[0]
        assert 'prediction_type' in first_entry
        assert 'beats_percentage' in first_entry
        assert 'empirical_multiplier' in first_entry
        assert 'theoretical_multiplier' in first_entry
        assert 'empirical_error_value' in first_entry
        assert 'theoretical_error_value' in first_entry

    def test_calculate_percentile_multipliers_prediction_type_set(self, calculator):
        """Test that prediction_type is correctly set in results."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        df, loc, scale, _ = calculator.fit_t_distribution(errors)

        result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "median")

        for entry in result:
            assert entry['prediction_type'] == 'median'

        min_result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "min")
        for entry in min_result:
            assert entry['prediction_type'] == 'min'

    def test_calculate_percentile_multipliers_percentiles_1_to_99(self, calculator):
        """Test that percentiles range from 1 to 99."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        df, loc, scale, _ = calculator.fit_t_distribution(errors)

        result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "median")

        percentiles = [entry['beats_percentage'] for entry in result]
        assert percentiles == list(range(1, 100))

    def test_calculate_percentile_multipliers_with_zero_scale(self, calculator):
        """Test calculate_percentile_multipliers() when scale is zero."""
        # Call directly with scale=0 to test the zero-scale branch
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        df, loc, scale = 5.0, 3.0, 0.0  # Explicitly set scale to 0

        result = calculator.calculate_percentile_multipliers(errors, df, loc, scale, "median")

        # With scale=0, BOTH multipliers should be 0 per implementation
        for entry in result:
            assert entry['empirical_multiplier'] == 0
            assert entry['theoretical_multiplier'] == 0


class TestCreateSafetyFactorTable:
    """Tests for create_safety_factor_table() method."""

    @pytest.fixture
    def calculator(self, mocker):
        """Create a SafetyFactorCalculator instance."""
        mock_logger = mocker.Mock()
        return SafetyFactorCalculator(logger=mock_logger)

    @pytest.fixture
    def mock_regression_data(self):
        """Create mock regression validation results."""
        np.random.seed(42)
        median_errors = np.random.normal(0, 1, 100)
        min_errors = np.random.normal(-0.5, 1.5, 100)

        median_df = pd.DataFrame({'residuals': median_errors})
        min_df = pd.DataFrame({'residuals': min_errors})

        return median_df, min_df

    def test_create_safety_factor_table_returns_dataframe(self, calculator, mocker, mock_regression_data):
        """Test that create_safety_factor_table() returns a DataFrame."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                assert isinstance(result, pd.DataFrame)

    def test_create_safety_factor_table_has_correct_entry_count(self, calculator, mocker, mock_regression_data):
        """Test that create_safety_factor_table() creates 396 entries (99 percentiles × 2 types × 2 prediction types)."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                assert len(result) == 396  # 99 percentiles × 2 multiplier types × 2 prediction types

    def test_create_safety_factor_table_columns(self, calculator, mocker, mock_regression_data):
        """Test that result DataFrame has correct columns."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                expected_columns = ['acad_term_id', 'prediction_type', 'beats_percentage', 'multiplier', 'multiplier_type']
                for col in expected_columns:
                    assert col in result.columns

    def test_create_safety_factor_table_acad_term_id(self, calculator, mocker, mock_regression_data):
        """Test that acad_term_id is correctly set."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                assert (result['acad_term_id'] == "AY202526T1").all()

    def test_create_safety_factor_table_prediction_types(self, calculator, mocker, mock_regression_data):
        """Test that both median and min prediction types are present."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                assert 'median' in result['prediction_type'].values
                assert 'min' in result['prediction_type'].values

    def test_create_safety_factor_table_multiplier_types(self, calculator, mocker, mock_regression_data):
        """Test that both empirical and theoretical multiplier types are present."""
        median_df, min_df = mock_regression_data

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = [median_df, min_df]

                result = calculator.create_safety_factor_table("AY202526T1")

                assert 'empirical' in result['multiplier_type'].values
                assert 'theoretical' in result['multiplier_type'].values

    def test_create_safety_factor_table_empty_dataframe_on_error(self, calculator, mocker):
        """Test that create_safety_factor_table() returns empty DataFrame on error."""
        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv', side_effect=Exception("File not found")):
                result = calculator.create_safety_factor_table("AY202526T1")

                assert isinstance(result, pd.DataFrame)
                assert result.empty


class TestEdgeCases:
    """Tests for edge cases in SafetyFactorCalculator."""

    @pytest.fixture
    def calculator(self, mocker):
        """Create a SafetyFactorCalculator instance."""
        mock_logger = mocker.Mock()
        return SafetyFactorCalculator(logger=mock_logger)

    def test_empty_data_fit_t_distribution(self, calculator):
        """Test fit_t_distribution() with empty array - should raise ValueError."""
        errors = np.array([])

        with pytest.raises(ValueError):
            calculator.fit_t_distribution(errors)

    def test_all_zeros_fit_t_distribution(self, calculator):
        """Test fit_t_distribution() with all zero values."""
        errors = np.zeros(100)

        df, loc, scale, params = calculator.fit_t_distribution(errors)

        # Scale should be 0 or very small for zero variance data
        assert scale >= 0

    def test_negative_values_fit_t_distribution(self, calculator):
        """Test fit_t_distribution() with all negative values."""
        errors = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

        df, loc, scale, params = calculator.fit_t_distribution(errors)

        assert np.isfinite(df) or np.isnan(df)
        assert np.isfinite(loc) or np.isnan(loc)
        assert np.isfinite(scale) or np.isnan(scale)

    def test_calculate_percentile_multipliers_empty_data(self, calculator):
        """Test calculate_percentile_multipliers() with empty data - raises IndexError."""
        errors = np.array([])

        # Empty data after filtering will cause np.percentile to raise IndexError
        with pytest.raises((ValueError, IndexError)):
            calculator.calculate_percentile_multipliers(errors, 1.0, 0.0, 1.0, "median")

    def test_calculate_percentile_multipliers_single_value(self, calculator):
        """Test calculate_percentile_multipliers() with single value."""
        errors = np.array([5.0])

        result = calculator.calculate_percentile_multipliers(errors, 1.0, 5.0, 0.0, "median")

        assert isinstance(result, list)
        assert len(result) == 99
        # When scale is 0, multipliers should be 0
        for entry in result:
            assert entry['empirical_multiplier'] == 0
            assert entry['theoretical_multiplier'] == 0
