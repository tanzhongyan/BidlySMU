"""
Unit tests for ClassScraper.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from src.scraper.class_scraper import ClassScraper, ClassScraperConfig


class TestClassScraperConfig:
    """Tests for ClassScraperConfig."""

    def test_config_defaults(self):
        """Config should have correct default values."""
        config = ClassScraperConfig(bidding_schedules={})
        assert config.min_class_number == 1000
        assert config.max_class_number == 5000
        assert config.consecutive_empty_threshold == 300
        assert config.delay_between_requests == 1.0
        assert config.max_retries == 3

    def test_config_requires_bidding_schedules(self):
        """Config should require bidding_schedules parameter."""
        # This test verifies the structure - actual validation happens at runtime
        config = ClassScraperConfig(bidding_schedules={"2025-26_T1": []})
        assert config.bidding_schedules is not None


class TestClassScraper:
    """Tests for ClassScraper."""

    def test_scraper_requires_config(self):
        """ClassScraper should raise ValueError if config is None."""
        with pytest.raises(ValueError, match="config is required"):
            ClassScraper(config=None)

    def test_scraper_initializes_with_config(self, mock_webdriver, mock_logger):
        """ClassScraper should initialize with provided config."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)
        assert scraper._config is config
        assert scraper._driver is mock_webdriver

    def test_term_code_map(self):
        """ClassScraper should have correct TERM_CODE_MAP."""
        assert ClassScraper.TERM_CODE_MAP == {
            "T1": "10",
            "T2": "20",
            "T3A": "31",
            "T3B": "32",
        }

    def test_config_property(self, mock_webdriver):
        """config property should return the config object."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver)
        assert scraper.config is config

    def test_scrape_returns_scraping_result(self, mock_webdriver, mock_logger):
        """scrape() should return a ScrapingResult object."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        # Mock the internal methods
        scraper._scrape_range = Mock(return_value=0)

        result = scraper.scrape(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        assert hasattr(result, "ay_term")
        assert hasattr(result, "round_folder")
        assert hasattr(result, "files_saved")
        assert hasattr(result, "is_success")

    def test_generate_scraped_filepaths_csv(self, mock_webdriver, tmp_path):
        """generate_scraped_filepaths_csv should create CSV with filepaths."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver)

        # Create a temporary directory with some HTML files
        base_dir = tmp_path / "html_files"
        base_dir.mkdir()
        (base_dir / "test1.html").write_text("<html>")
        (base_dir / "test2.html").write_text("<html>")

        output_csv = tmp_path / "filepaths.csv"
        scraper.generate_scraped_filepaths_csv(
            base_dir=str(base_dir), output_csv=str(output_csv)
        )

        assert output_csv.exists()
