"""
Unit tests for HTMLDataExtractor.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.scraper.html_data_extractor import HTMLDataExtractor, ExtractionResult


class TestHTMLDataExtractor:
    """Tests for HTMLDataExtractor."""

    def test_requires_no_config(self):
        """HTMLDataExtractor should not require config."""
        extractor = HTMLDataExtractor()
        assert extractor is not None

    def test_initializes_with_driver(self, mock_webdriver):
        """HTMLDataExtractor should accept driver in constructor."""
        extractor = HTMLDataExtractor(driver=mock_webdriver)
        assert extractor._driver is mock_webdriver


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self):
        """ExtractionResult should have correct defaults."""
        result = ExtractionResult()
        assert result.files_processed == 0
        assert result.files_successful == 0
        assert result.errors == []
        assert result.standalone_records == []
        assert result.multiple_records == []

    def test_tracks_files_processed(self):
        """ExtractionResult should track files_processed."""
        result = ExtractionResult(files_processed=10)
        assert result.files_processed == 10

    def test_tracks_files_successful(self):
        """ExtractionResult should track files_successful."""
        result = ExtractionResult(files_successful=8)
        assert result.files_successful == 8

    def test_tracks_errors(self):
        """ExtractionResult should track errors list."""
        error = {"file": "test.html", "error": "Parse failed"}
        result = ExtractionResult(errors=[error])
        assert len(result.errors) == 1
        assert result.errors[0]["file"] == "test.html"


class TestHTMLDataExtractorBiddingWindow:
    """Tests for HTMLDataExtractor bidding window extraction."""

    def test_extract_r2w1_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W1."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("2025-26_T1_R2W1")
        assert result == 'Round 2 Window 1'

    def test_extract_r2w2_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W2."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("2025-26_T1_R2W2")
        assert result == 'Round 2 Window 2'

    def test_extract_r2w3_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W3."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("2025-26_T1_R2W3")
        assert result == 'Round 2 Window 3'

    def test_extract_r2w4_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W4."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("2025-26_T3B_R2W4")
        assert result == 'Round 2 Window 4'

    def test_extract_unknown_returns_original(self):
        """_extract_bidding_window_from_folder should return original for unknown."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("UNKNOWN")
        assert result == "UNKNOWN"


class TestHTMLDataExtractorRun:
    """Tests for HTMLDataExtractor.run() method."""

    def test_run_returns_bool(self, mock_webdriver):
        """run() should return a boolean."""
        extractor = HTMLDataExtractor(driver=mock_webdriver)

        with patch.object(extractor, '_process_all_files'):
            with patch.object(extractor, '_save_to_excel'):
                result = extractor.run(output_path='test.xlsx')
                assert isinstance(result, bool)

    def test_run_sets_up_driver_if_none(self, mock_webdriver):
        """run() should set up driver if _driver is None."""
        extractor = HTMLDataExtractor()
        assert extractor._driver is None

        with patch('src.scraper.html_data_extractor.ChromeDriverFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create.return_value = mock_webdriver
            mock_factory_class.return_value = mock_factory

            with patch.object(extractor, '_process_all_files'):
                with patch.object(extractor, '_save_to_excel'):
                    extractor.run(output_path='test.xlsx')
                    mock_factory.create.assert_called_once()
