"""
Unit tests for OverallResultsScraper.
"""
import pytest
from unittest.mock import Mock

from src.scraper.overall_results_scraper import OverallResultsScraper, OverallResultsConfig
from src.models.dto.scraping_result import ScrapingResult


class TestOverallResultsConfig:
    """Tests for OverallResultsConfig."""

    def test_requires_bidding_schedules(self):
        """OverallResultsConfig should require bidding_schedules."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        assert config.bidding_schedules == {}

    def test_requires_start_ay_term(self):
        """OverallResultsConfig should require start_ay_term."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        assert config.start_ay_term == "2025-26_T1"

    def test_default_values(self):
        """OverallResultsConfig should have correct defaults."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        assert config.base_url == "https://boss.intranet.smu.edu.sg/OverallResults.aspx"
        assert config.delay == 5
        assert config.headless is False
        assert config.page_size == 50
        assert config.max_retries == 3

    def test_desired_columns_preset(self):
        """OverallResultsConfig should have desired_columns preset."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        assert 'Term' in config.desired_columns
        assert 'Course Code' in config.desired_columns
        assert 'Median Bid' in config.desired_columns


class TestOverallResultsScraper:
    """Tests for OverallResultsScraper."""

    def test_requires_config(self):
        """OverallResultsScraper should raise ValueError if config is None."""
        with pytest.raises(ValueError, match="config is required"):
            OverallResultsScraper(config=None)

    def test_initializes_with_config(self):
        """OverallResultsScraper should initialize with config."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        scraper = OverallResultsScraper(config=config)
        assert scraper._config is config

    def test_term_map_values(self):
        """TERM_MAP should have correct values."""
        assert OverallResultsScraper.TERM_MAP == {
            'Term 1': 'T1',
            'Term 2': 'T2',
            'Term 3A': 'T3A',
            'Term 3B': 'T3B'
        }

    def test_desired_columns_preset(self):
        """DESIRED_COLUMNS should be set correctly."""
        assert 'Term' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Course Code' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Median Bid' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Min Bid' in OverallResultsScraper.DESIRED_COLUMNS


class TestOverallResultsScraperScrape:
    """Tests for OverallResultsScraper.scrape()."""

    def test_scrape_returns_scraping_result(self, mock_webdriver):
        """scrape() should return a ScrapingResult object."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="2025-26_T1"
        )
        scraper = OverallResultsScraper(config=config, driver=mock_webdriver)

        # Mock the run method
        scraper.run = Mock(return_value=ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
        ))

        result = scraper.scrape()

        assert isinstance(result, ScrapingResult)
        assert result.files_saved == 10
