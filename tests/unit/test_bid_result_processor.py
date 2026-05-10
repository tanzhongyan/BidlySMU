"""
Unit tests for BidResultProcessor.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import os

from src.pipeline.processors.bid_result_processor import BidResultProcessor
from src.pipeline.dtos.bid_result_dto import BidResultDTO


class TestBidResultProcessor:
    """Tests for BidResultProcessor."""

    def test_requires_raw_data(self):
        """Processor should require raw_data parameter."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )
        assert processor._raw_data is not None

    def test_initializes_with_empty_lookups(self):
        """Processor should initialize with empty lookups when not provided."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={},
            course_lookup={}
        )
        assert processor._class_lookup == {}
        assert processor._bid_window_lookup == {}

    @patch('src.pipeline.processors.bid_result_processor.CURRENT_WINDOW_NAME', None)
    @patch('src.pipeline.processors.bid_result_processor.PREVIOUS_WINDOW_NAME', None)
    def test_process_returns_tuple(self):
        """process() should return (new_results, updated_results) tuple."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )
        result = processor.process()
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch('src.pipeline.processors.bid_result_processor.CURRENT_WINDOW_NAME', None)
    @patch('src.pipeline.processors.bid_result_processor.PREVIOUS_WINDOW_NAME', None)
    def test_process_handles_no_windows(self):
        """process() should return empty lists when no windows configured."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )
        new_results, updated_results = processor.process()
        assert new_results == []
        assert updated_results == []


class TestLoadOverallResults:
    """Tests for _load_overall_results method."""

    def test_returns_none_when_file_not_found(self):
        """_load_overall_results should return None when file doesn't exist."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='nonexistent_path.xlsx',
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )

        result = processor._load_overall_results()

        assert result is None

    def test_loads_excel_file(self, tmp_path):
        """_load_overall_results should load and return DataFrame from Excel."""
        # Create a mock Excel file
        excel_path = tmp_path / "overall_results.xlsx"
        df = pd.DataFrame({
            'Bidding Window': ['Round 1 Window 1'],
            'Term': ['AY202526T1'],
            'Course Code': ['MGMT715'],
            'Section': ['G1'],
            'Median Bid': [100.0],
            'Min Bid': [50.0]
        })
        df.to_excel(excel_path, index=False)

        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path=str(excel_path),
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )

        result = processor._load_overall_results()

        assert result is not None
        assert len(result) == 1
        assert 'Bidding Window' in result.columns


class TestSafeConversions:
    """Tests for safe value conversion methods."""

    def test_safe_int_handles_nan(self):
        """_safe_int should handle NaN values."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )

        result = processor._safe_int(float('nan'))
        assert result is None

    def test_safe_int_handles_none(self):
        """_safe_int should handle None values."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )

        result = processor._safe_int(None)
        assert result is None

    def test_safe_int_converts_valid_values(self):
        """_safe_int should convert valid integer strings."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )

        assert processor._safe_int('42') == 42
        assert processor._safe_int(42.5) == 42

    def test_safe_float_handles_nan(self):
        """_safe_float should handle NaN values."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )

        result = processor._safe_float(float('nan'))
        assert result is None

    def test_safe_float_converts_valid_values(self):
        """_safe_float should convert valid float strings."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={}
        )

        assert processor._safe_float('3.14') == 3.14
        assert processor._safe_float(42) == 42.0


class TestProcessPreviousWindow:
    """Tests for _process_previous_window method."""

    @patch('src.pipeline.processors.bid_result_processor.PREVIOUS_WINDOW_NAME', None)
    def test_skips_when_no_previous_window(self):
        """_process_previous_window should skip when PREVIOUS_WINDOW_NAME is None."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={},
            logger=Mock()
        )

        processor._process_previous_window()

        assert processor._new_bid_results == []
        assert processor._updated_bid_results == []

    @patch('src.pipeline.processors.bid_result_processor.PREVIOUS_WINDOW_NAME', 'Round 2 Window 1')
    def test_warns_and_skips_when_previous_window_file_missing(self):
        """_process_previous_window should log warning and skip when file is missing."""
        mock_logger = Mock()
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='nonexistent_path.xlsx',
            class_lookup={},
            bid_window_lookup={},
            logger=mock_logger
        )

        processor._process_previous_window()

        # Should log a warning, not raise
        mock_logger.warning.assert_called_once()
        assert "overallBossResults file not found" in mock_logger.warning.call_args[0][0]
        assert processor._new_bid_results == []
        assert processor._updated_bid_results == []


class TestFindAllClassIds:
    """Tests for _find_all_class_ids method."""

    def test_finds_class_ids_by_term_and_boss(self):
        """_find_all_class_ids should find classes by acad_term_id and boss_id."""
        from src.pipeline.dtos.class_dto import ClassDTO

        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-uuid-1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-uuid-2'),
            ('AY202526T1', 1002, 'prof1'): MagicMock(id='class-uuid-3'),
        }

        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup=class_lookup,
            bid_window_lookup={}
        )

        result = processor._find_all_class_ids('AY202526T1', 1001)

        assert len(result) == 2
        assert 'class-uuid-1' in result
        assert 'class-uuid-2' in result

    def test_returns_empty_when_no_match(self):
        """_find_all_class_ids should return empty list when no classes match."""
        class_lookup = {
            ('AY202526T1', 1001, None): MagicMock(id='class-uuid-1'),
        }

        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup=class_lookup,
            bid_window_lookup={}
        )

        result = processor._find_all_class_ids('AY202526T2', 9999)

        assert result == []


class TestFindAllClassIdsByCourseSection:
    """Tests for _find_all_class_ids_by_course_section method."""

    def test_finds_by_course_code_and_section(self):
        """_find_all_class_ids_by_course_section should find by course code and section."""
        from src.pipeline.dtos.class_dto import ClassDTO
        from src.pipeline.dtos.course_dto import CourseDTO

        course_lookup = {
            'MGMT715': MagicMock(id='course-uuid')
        }

        class_lookup = {
            ('AY202526T1', 1001, 'prof1'): MagicMock(id='class-1', course_id='course-uuid', section='G1'),
            ('AY202526T1', 1001, 'prof2'): MagicMock(id='class-2', course_id='course-uuid', section='G1'),
            ('AY202526T1', 1001, 'prof3'): MagicMock(id='class-3', course_id='course-uuid', section='G2'),
        }

        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup=class_lookup,
            bid_window_lookup={},
            course_lookup=course_lookup
        )

        result = processor._find_all_class_ids_by_course_section('AY202526T1', 'MGMT715', 'G1')

        assert len(result) == 2
        assert 'class-1' in result
        assert 'class-2' in result

    def test_returns_empty_when_course_not_found(self):
        """_find_all_class_ids_by_course_section should return empty when course not in lookup."""
        processor = BidResultProcessor(
            raw_data=pd.DataFrame(),
            overall_results_path='',
            class_lookup={},
            bid_window_lookup={},
            course_lookup={}
        )

        result = processor._find_all_class_ids_by_course_section('AY202526T1', 'NONEXISTENT', 'G1')

        assert result == []


class TestBidResultDTO:
    """Tests for BidResultDTO."""

    def test_create_bid_result_dto(self):
        """BidResultDTO should be creatable with all fields."""
        dto = BidResultDTO(
            bid_window_id='bw-123',
            class_id='class-456',
            vacancy=30,
            opening_vacancy=40,
            before_process_vacancy=35,
            d_i_c_e=1,
            after_process_vacancy=25,
            enrolled_students=15,
            median=100.0,
            min=50.0
        )

        assert dto.bid_window_id == 'bw-123'
        assert dto.class_id == 'class-456'
        assert dto.median == 100.0
        assert dto.min == 50.0

    def test_to_csv_row(self):
        """BidResultDTO.to_csv_row should return dict with all fields."""
        dto = BidResultDTO(
            bid_window_id='bw-123',
            class_id='class-456',
            vacancy=30,
            opening_vacancy=40,
            before_process_vacancy=35,
            d_i_c_e=1,
            after_process_vacancy=25,
            enrolled_students=15,
            median=100.0,
            min=50.0
        )

        row = dto.to_csv_row()
        assert row['bid_window_id'] == 'bw-123'
        assert row['class_id'] == 'class-456'
        assert row['vacancy'] == 30
        assert row['median'] == 100.0
        assert row['min'] == 50.0

    def test_to_db_row(self):
        """BidResultDTO.to_db_row should return dict with all fields."""
        dto = BidResultDTO(
            bid_window_id='bw-123',
            class_id='class-456',
            vacancy=30,
            opening_vacancy=40,
            before_process_vacancy=35,
            d_i_c_e=1,
            after_process_vacancy=25,
            enrolled_students=15,
            median=100.0,
            min=50.0
        )

        row = dto.to_db_row()
        assert row['bid_window_id'] == 'bw-123'
        assert row['class_id'] == 'class-456'
        assert row['median'] == 100.0

    def test_optional_fields_with_none(self):
        """BidResultDTO should handle None for optional fields."""
        dto = BidResultDTO(
            bid_window_id='bw-123',
            class_id='class-456',
            vacancy=None,
            opening_vacancy=None,
            before_process_vacancy=None,
            d_i_c_e=None,
            after_process_vacancy=None,
            enrolled_students=None,
            median=None,
            min=None
        )

        assert dto.vacancy is None
        assert dto.median is None
        assert dto.min is None

        csv_row = dto.to_csv_row()
        assert csv_row['vacancy'] is None
        assert csv_row['median'] is None