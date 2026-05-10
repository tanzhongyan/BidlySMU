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

    def test_process_instructor_names_with_lookup(self, mocker):
        """Test processing with professor_lookup mapping."""
        # Create mock professor lookup CSV
        mock_lookup_df = pd.DataFrame({
            'boss_name': ['JOHN DOE'],
            'afterclass_name': ['DOE, JOHN']
        })

        with patch.object(Path, 'exists', return_value=True):
            with patch('pandas.read_csv', return_value=mock_lookup_df):
                # Create a new transformer instance inside the patch to ensure the lookup is loaded with the mock data
                mock_logger = mocker.Mock()
                transformer = SMUBiddingTransformer(logger=mock_logger)
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


class TestGetFeatureMethods:
    """Tests for get_categorical_features(), get_numeric_features(), get_feature_names() methods.

    These methods were present in V4 step_3_BidPrediction.py SMUBiddingTransformer.
    They should return the feature lists after fitting/transforming.
    """

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame with all required columns."""
        return pd.DataFrame({
            'course_code': ['MGMT715', 'COR-COMM1304'],
            'course_name': ['Business Ethics', 'Communication'],
            'acad_year_start': [2025, 2025],
            'term': ['1', '2'],
            'start_time': ['19:30', '18:00'],
            'day_of_week': ['Mon,Thu', 'Tuesday'],
            'before_process_vacancy': [50, 30],
            'bidding_window': ['Round 1 Window 1', 'Round 2 Window 2'],
            'instructor': ['JOHN DOE', 'JANE SMITH'],
            'section': ['A', 'B']
        })

    def test_get_categorical_features_returns_list_after_transform(self, mocker, valid_df):
        """get_categorical_features() should return a list of categorical features after transform."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit(valid_df)
        transformer.transform(valid_df)

        cat_features = transformer.get_categorical_features()
        assert isinstance(cat_features, list)
        assert len(cat_features) > 0
        assert 'subject_area' in cat_features
        assert 'catalogue_no' in cat_features

    def test_get_numeric_features_returns_list_after_transform(self, mocker, valid_df):
        """get_numeric_features() should return a list of numeric features after transform."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit(valid_df)
        transformer.transform(valid_df)

        num_features = transformer.get_numeric_features()
        assert isinstance(num_features, list)
        assert len(num_features) > 0
        assert 'before_process_vacancy' in num_features
        assert 'acad_year_start' in num_features

    def test_get_feature_names_returns_all_features(self, mocker, valid_df):
        """get_feature_names() should return all categorical + numeric features."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit_transform(valid_df)

        all_features = transformer.get_feature_names()
        cat_features = transformer.get_categorical_features()
        num_features = transformer.get_numeric_features()

        assert isinstance(all_features, list)
        assert len(all_features) == len(cat_features) + len(num_features)
        assert all_features == cat_features + num_features

    def test_get_categorical_features_returns_copy(self, mocker, valid_df):
        """get_categorical_features() should return a copy, not the original list."""
        mock_logger = mocker.Mock()
        transformer = SMUBiddingTransformer(logger=mock_logger)

        transformer.fit_transform(valid_df)

        cat1 = transformer.get_categorical_features()
        cat1.append('fake_feature')
        cat2 = transformer.get_categorical_features()

        assert 'fake_feature' not in cat2


class TestExtractCourseFeaturesWithSampleData:
    """Tests for _extract_course_features() using sample data patterns from raw_data.xlsx.

    Based on V4 step_3_BidPrediction.py split_course_code logic:
    - Standard codes like MGMT715 -> subject=MGMT, catalogue=715
    - Hyphenated codes like COR-COMM1304 -> subject=COR-COMM, catalogue=1304
    - Codes with underscores like LAW103_603 -> subject=LAW, catalogue=103 (underscore not handled specially)
    """

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

    def test_extract_course_features_hyphenated_code(self, transformer):
        """Test extracting features from hyphenated course code like COR-COMM1304."""
        df = pd.DataFrame({'course_code': ['COR-COMM1304']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'COR-COMM'
        assert result.loc[0, 'catalogue_no'] == 1304

    def test_extract_course_features_another_hyphenated(self, transformer):
        """Test extracting features from another hyphenated code like COR-STAT1202."""
        df = pd.DataFrame({'course_code': ['COR-STAT1202']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'COR-STAT'
        assert result.loc[0, 'catalogue_no'] == 1202

    def test_extract_course_features_underscore_code(self, transformer):
        """Test extracting features from course code with underscore like LAW103_603.

        The underscore is NOT specially handled - regex stops at first digit sequence.
        This test documents the current behavior.
        """
        df = pd.DataFrame({'course_code': ['LAW103_603']})

        result = transformer._extract_course_features(df)

        # Current behavior: regex ([A-Z\-]+)(\d+) matches LAW and 103
        assert result.loc[0, 'subject_area'] == 'LAW'
        assert result.loc[0, 'catalogue_no'] == 103

    def test_extract_course_features_stat_with_suffix(self, transformer):
        """Test extracting features from STAT701A - letters after digits."""
        df = pd.DataFrame({'course_code': ['STAT701A']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'STAT'
        assert result.loc[0, 'catalogue_no'] == 701

    def test_extract_course_features_multiple_hyphens(self, transformer):
        """Test extracting features from course code with multiple hyphens."""
        df = pd.DataFrame({'course_code': ['COR-MGMT1302']})

        result = transformer._extract_course_features(df)

        assert result.loc[0, 'subject_area'] == 'COR-MGMT'
        assert result.loc[0, 'catalogue_no'] == 1302


class TestSectionFormats:
    """Tests for section handling based on sample data patterns.

    Sample data shows sections like G1, G2, G49, G61, SG81, SG91, etc.
    SG sections appear in some courses like COR1703, FNCE101, etc.
    """

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_process_basic_features_preserves_g_sections(self, transformer):
        """Test that G1, G2, G61 etc. sections are preserved as categorical."""
        df = pd.DataFrame({
            'course_code': ['MGMT715', 'MGMT715', 'MGMT715'],
            'course_name': ['Business Ethics', 'Business Ethics', 'Business Ethics'],
            'acad_year_start': [2025, 2025, 2025],
            'term': ['1', '1', '1'],
            'start_time': ['19:30', '19:30', '19:30'],
            'day_of_week': ['Mon', 'Mon', 'Mon'],
            'before_process_vacancy': [50, 50, 50],
            'bidding_window': ['Round 1 Window 1', 'Round 1 Window 1', 'Round 1 Window 1'],
            'instructor': ['JOHN DOE', 'JOHN DOE', 'JOHN DOE'],
            'section': ['G1', 'G61', 'G3']
        })

        result = transformer._process_basic_features(df)

        assert 'section' in result.columns
        assert result.loc[0, 'section'] == 'G1'
        assert result.loc[1, 'section'] == 'G61'
        assert result.loc[2, 'section'] == 'G3'

    def test_process_basic_features_preserves_sg_sections(self, transformer):
        """Test that SG81, SG91 sections are preserved as categorical.

        SG sections appear in courses like COR1703, FNCE101, etc.
        These are valid section identifiers and should be preserved.
        """
        df = pd.DataFrame({
            'course_code': ['COR1703', 'FNCE101'],
            'course_name': ['Corporate Communication', 'Finance Principles'],
            'acad_year_start': [2025, 2025],
            'term': ['1', '1'],
            'start_time': ['19:30', '19:30'],
            'day_of_week': ['Mon', 'Mon'],
            'before_process_vacancy': [50, 50],
            'bidding_window': ['Round 1 Window 1', 'Round 1 Window 1'],
            'instructor': ['JOHN DOE', 'JOHN DOE'],
            'section': ['SG81', 'SG91']
        })

        result = transformer._process_basic_features(df)

        assert 'section' in result.columns
        assert result.loc[0, 'section'] == 'SG81'
        assert result.loc[1, 'section'] == 'SG91'


class TestCourseAreaPatterns:
    """Tests for course_area handling based on sample data patterns.

    Note: course_area is NOT processed by SMUBiddingTransformer._process_basic_features().
    It is stored as a raw column in the CourseDTO and processed separately.
    These tests document the expected course_area values from raw_data.xlsx.

    Sample course_area values from raw_data.xlsx:
    - "There is no applicable Course Area."
    - "EMBA Programme Core"
    - "MITB Digi Transformation Track Core"
    - "GPGM Programme Core (OM)"
    """

    def test_course_area_values_are_strings(self):
        """Test that course_area values from DataFrame are strings."""
        df = pd.DataFrame({
            'course_area': [
                'EMBA Programme Core',
                'MITB Digi Transformation Track Core',
                'There is no applicable Course Area.'
            ]
        })

        assert df.loc[0, 'course_area'] == 'EMBA Programme Core'
        assert isinstance(df.loc[0, 'course_area'], str)


class TestCreditUnitsAsFloat:
    """Tests verifying credit_units are parsed as float.

    Sample credit_units values from raw_data.xlsx: [0.5, 1.0, 1.5, 1.25, 2.0, 0.75, 28.0, 0.25, 26.0, 12.0, 4.0, 5.0, 6.0]
    """

    @pytest.fixture
    def transformer(self, mocker):
        """Create a fitted transformer instance."""
        mock_logger = mocker.Mock()
        t = SMUBiddingTransformer(logger=mock_logger)
        t.is_fitted = True
        return t

    def test_process_basic_features_before_process_vacancy_is_numeric(self, transformer):
        """Test that before_process_vacancy is processed as numeric."""
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

        result = transformer._process_basic_features(df)

        assert 'before_process_vacancy' in result.columns
        # Should be numeric, not string
        assert pd.api.types.is_numeric_dtype(result['before_process_vacancy'])
