"""
Unit tests for SMUBiddingTransformer in src/pipeline/transformer.py
"""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pipeline.transformer import SMUBiddingTransformer


class TestSMUBiddingTransformerInit:
    """Tests for SMUBiddingTransformer initialization."""

    def test_init_sets_defaults(self, mocker):
        """Test that __init__ sets correct default values."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        assert transformer.is_fitted is False
        assert transformer.categorical_features == []
        assert transformer.numeric_features == []
        assert transformer._professor_lookup_path == "script_input/professor_lookup.csv"
        mock_logger.assert_not_called()

    def test_init_accepts_custom_logger(self, mocker):
        """Test that custom logger is accepted."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)
        assert transformer._logger == mock_logger

    def test_init_accepts_custom_professor_lookup_path(self, mocker):
        """Test that custom professor_lookup_path is accepted."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(
            logger=mock_logger,
            professor_lookup_path="custom/path.csv"
        )
        assert transformer._professor_lookup_path == "custom/path.csv"


class TestSMUBiddingTransformerFit:
    """Tests for SMUBiddingTransformer.fit()"""

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame with all required columns."""
        return pd.DataFrame({
            'course_code': ['MGMT715', 'LEGAL501'],
            'course_name': ['Business Ethics', 'Corporate Law'],
            'acad_year_start': [2025, 2025],
            'term': ['1', '2'],
            'start_time': ['19:30', '18:00'],
            'day_of_week': ['Mon,Thu', 'Tuesday'],
            'before_process_vacancy': [50, 30],
            'bidding_window': ['Round 1 Window 1', 'Round 2 Window 2'],
            'instructor': ['JOHN DOE', 'JANE SMITH'],
            'section': ['A', 'B']
        })

    def test_fit_raises_valueerror_when_missing_required_columns(self, mocker):
        """Test that fit() raises ValueError when required columns are missing."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        df_missing_cols = pd.DataFrame({
            'course_code': ['MGMT715'],
            'course_name': ['Business Ethics']
            # Missing other required columns
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            transformer.fit(df_missing_cols)

    def test_fit_sets_is_fitted_true_on_success(self, mocker, valid_df):
        """Test that fit() sets is_fitted=True on successful fit."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit(valid_df)

        assert transformer.is_fitted is True
        mock_logger.info.assert_called_once()
        assert "Fitting transformer on 2 rows" in mock_logger.info.call_args[0][0]

    def test_fit_returns_self(self, mocker, valid_df):
        """Test that fit() returns the transformer instance."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        result = transformer.fit(valid_df)

        assert result is transformer


class TestSMUBiddingTransformerTransform:
    """Tests for SMUBiddingTransformer.transform()"""

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame with all required columns."""
        return pd.DataFrame({
            'course_code': ['MGMT715', 'LEGAL501'],
            'course_name': ['Business Ethics', 'Corporate Law'],
            'acad_year_start': [2025, 2025],
            'term': ['1', '2'],
            'start_time': ['19:30', '18:00'],
            'day_of_week': ['Mon,Thu', 'Tuesday'],
            'before_process_vacancy': [50, 30],
            'bidding_window': ['Round 1 Window 1', 'Round 2 Window 2'],
            'instructor': ['JOHN DOE', 'JANE SMITH'],
            'section': ['A', 'B']
        })

    def test_transform_raises_valueerror_when_not_fitted(self, mocker):
        """Test that transform() raises ValueError when transformer is not fitted."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        df = pd.DataFrame({
            'course_code': ['MGMT715'],
            'course_name': ['Business Ethics'],
            'acad_year_start': [2025],
            'term': ['1'],
            'start_time': ['19:30'],
            'day_of_week': ['Mon'],
            'before_process_vacancy': [50],
            'bidding_window': ['Round 1 Window 1'],
            'instructor': ['JOHN DOE'],
            'section': ['A']
        })

        with pytest.raises(ValueError, match="Transformer must be fitted before transform"):
            transformer.transform(df)

    def test_transform_returns_dataframe_when_fitted(self, mocker, valid_df):
        """Test that transform() returns a DataFrame when fitted."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit(valid_df)
        result = transformer.transform(valid_df)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestExtractCourseFeatures:
    """Tests for _extract_course_features() method."""

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_extract_course_features_standard_code(self, mocker, transformer):
        """Test extracting features from standard course code like MGMT715."""
        df = pd.DataFrame({'course_code': ['MGMT715']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'MGMT'
        assert result.loc[0, 'catalogue_no'] == 715

    def test_extract_course_features_hyphenated_code(self, mocker, transformer):
        """Test extracting features from hyphenated course code like COR-COMM175."""
        df = pd.DataFrame({'course_code': ['COR-COMM175']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'COR-COMM'
        assert result.loc[0, 'catalogue_no'] == 175

    def test_extract_course_features_another_hyphenated(self, mocker, transformer):
        """Test extracting features from another hyphenated code like LEGAL501."""
        df = pd.DataFrame({'course_code': ['LEGAL501']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'LEGAL'
        assert result.loc[0, 'catalogue_no'] == 501

    def test_extract_course_features_none_value(self, mocker, transformer):
        """Test extracting features when course_code is None."""
        df = pd.DataFrame({'course_code': [None]})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] is None
        assert result.loc[0, 'catalogue_no'] == 0

    def test_extract_course_features_nan_value(self, mocker, transformer):
        """Test extracting features when course_code is NaN."""
        df = pd.DataFrame({'course_code': [np.nan]})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] is None
        assert result.loc[0, 'catalogue_no'] == 0


class TestExtractRoundWindow:
    """Tests for _extract_round_window() method."""

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_extract_round_window_round1_window1(self, mocker, transformer):
        """Test extracting round and window from 'Round 1 Window 1'."""
        df = pd.DataFrame({'bidding_window': ['Round 1 Window 1']})

        result = transformer._extract_round_window(df)

        assert result.loc[0, 'round'] == '1'
        assert result.loc[0, 'window'] == 1

    def test_extract_round_window_incoming_freshmen(self, mocker, transformer):
        """Test extracting round and window from 'Incoming Freshmen Rnd 1 Win 4'."""
        df = pd.DataFrame({'bidding_window': ['Incoming Freshmen Rnd 1 Win 4']})

        result = transformer._extract_round_window(df)

        # Incoming Freshmen gets 'F' suffix appended to round (e.g., '1' -> '1F')
        assert result.loc[0, 'round'] == '1F'
        assert result.loc[0, 'window'] == 4


class TestProcessBasicFeatures:
    """Tests for _process_basic_features() method."""

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_process_basic_features_preserves_categorical(self, mocker, transformer):
        """Test that _process_basic_features() preserves categorical nature of term."""
        df = pd.DataFrame({
            'term': ['1', '2'],
            'start_time': ['19:30', '18:00'],
            'course_name': ['Business Ethics', 'Corporate Law'],
            'section': ['A', 'B'],
            'instructor': ['JOHN DOE', 'JANE SMITH'],
            'before_process_vacancy': [50, 30],
            'acad_year_start': [2025, 2025]
        })

        result = transformer._process_basic_features(df)

        assert 'term' in result.columns
        assert 'start_time' in result.columns
        assert result.loc[0, 'term'] == '1'
        assert result.loc[1, 'term'] == '2'


class TestCreateDayOneHotEncoding:
    """Tests for _create_day_one_hot_encoding() method."""

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_create_day_one_hot_comma_separated(self, mocker, transformer):
        """Test one-hot encoding for comma-separated days like 'Mon,Thu'."""
        df = pd.DataFrame({'day_of_week': ['Mon,Thu']})

        result = transformer._create_day_one_hot_encoding(df)

        assert result.loc[0, 'has_mon'] == 1
        assert result.loc[0, 'has_thu'] == 1
        assert result.loc[0, 'has_tue'] == 0
        assert result.loc[0, 'has_wed'] == 0
        assert result.loc[0, 'has_fri'] == 0
        assert result.loc[0, 'has_sat'] == 0
        assert result.loc[0, 'has_sun'] == 0

    def test_create_day_one_hot_single_day(self, mocker, transformer):
        """Test one-hot encoding for single day like 'Monday'."""
        df = pd.DataFrame({'day_of_week': ['Monday']})

        result = transformer._create_day_one_hot_encoding(df)

        assert result.loc[0, 'has_mon'] == 1
        assert result.loc[0, 'has_tue'] == 0
        assert result.loc[0, 'has_wed'] == 0
        assert result.loc[0, 'has_thu'] == 0
        assert result.loc[0, 'has_fri'] == 0

    def test_create_day_one_hot_json_array(self, mocker, transformer):
        """Test one-hot encoding for JSON array like '["Mon", "Thu"]'."""
        df = pd.DataFrame({'day_of_week': ['["Mon", "Thu"]']})

        result = transformer._create_day_one_hot_encoding(df)

        assert result.loc[0, 'has_mon'] == 1
        assert result.loc[0, 'has_thu'] == 1
        assert result.loc[0, 'has_tue'] == 0
        assert result.loc[0, 'has_wed'] == 0
        assert result.loc[0, 'has_fri'] == 0
        assert result.loc[0, 'has_sat'] == 0
        assert result.loc[0, 'has_sun'] == 0


class TestProcessInstructorNames:
    """Tests for _process_instructor_names() method."""

    @pytest.fixture
    def transformer(self, mocker):
        """Create a transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_process_instructor_names_single_professor(self, mocker, transformer):
        """Test processing a single professor name."""
        result = transformer._process_instructor_names('JOHN DOE')
        # Without lookup, should return JSON array with original name
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_process_instructor_names_tba(self, mocker, transformer):
        """Test that TBA returns None."""
        result = transformer._process_instructor_names('TBA')
        assert result is None

    def test_process_instructor_names_none(self, mocker, transformer):
        """Test that None input returns None."""
        result = transformer._process_instructor_names(None)
        assert result is None

    def test_process_instructor_names_empty_string(self, mocker, transformer):
        """Test that empty string returns None."""
        result = transformer._process_instructor_names('')
        assert result is None

    def test_process_instructor_names_with_lookup(self, mocker, transformer):
        """Test processing with professor_lookup mapping."""
        # Create mock professor lookup CSV
        mock_lookup_df = pd.DataFrame({
            'boss_name': ['JOHN DOE'],
            'afterclass_name': ['DOE, JOHN']
        })

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv', return_value=mock_lookup_df):
                result = transformer._process_instructor_names('JOHN DOE')
                # With lookup, should return mapped name
                assert result is not None
                parsed = json.loads(result)
                assert 'DOE, JOHN' in parsed

    def test_process_instructor_names_comma_separated(self, mocker, transformer):
        """Test processing comma-separated professor names."""
        result = transformer._process_instructor_names('JOHN DOE, JANE SMITH')
        # Should return JSON array
        assert result is not None
        parsed = json.loads(result)
        assert isinstance(parsed, list)


class TestGetCategoricalFeatures:
    """Tests for get_categorical_features() method."""

    def test_get_categorical_features_returns_copy(self, mocker):
        """Test that get_categorical_features() returns a copy."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)
        transformer.categorical_features = ['subject_area', 'catalogue_no']

        result = transformer.get_categorical_features()

        assert result == ['subject_area', 'catalogue_no']
        assert result is not transformer.categorical_features


class TestGetNumericFeatures:
    """Tests for get_numeric_features() method."""

    def test_get_numeric_features_returns_copy(self, mocker):
        """Test that get_numeric_features() returns a copy."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)
        transformer.numeric_features = ['before_process_vacancy', 'window']

        result = transformer.get_numeric_features()

        assert result == ['before_process_vacancy', 'window']
        assert result is not transformer.numeric_features


class TestFitTransform:
    """Tests for fit_transform() method."""

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame with all required columns."""
        return pd.DataFrame({
            'course_code': ['MGMT715'],
            'course_name': ['Business Ethics'],
            'acad_year_start': [2025],
            'term': ['1'],
            'start_time': ['19:30'],
            'day_of_week': ['Mon'],
            'before_process_vacancy': [50],
            'bidding_window': ['Round 1 Window 1'],
            'instructor': ['JOHN DOE'],
            'section': ['A']
        })

    def test_fit_transform_fits_and_transforms(self, mocker, valid_df):
        """Test that fit_transform() fits and then transforms."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        result = transformer.fit_transform(valid_df)

        assert transformer.is_fitted is True
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestGetFeatureNames:
    """Tests for get_feature_names() method."""

    def test_get_feature_names_raises_when_not_fitted(self, mocker):
        """Test that get_feature_names() raises ValueError when not fitted."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        with pytest.raises(ValueError, match="Transformer must be fitted"):
            transformer.get_feature_names()

    def test_get_feature_names_returns_combined_features(self, mocker):
        """Test that get_feature_names() returns categorical + numeric features."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)
        transformer.is_fitted = True
        transformer.categorical_features = ['subject_area', 'catalogue_no']
        transformer.numeric_features = ['window', 'before_process_vacancy']

        result = transformer.get_feature_names()

        assert result == ['subject_area', 'catalogue_no', 'window', 'before_process_vacancy']
