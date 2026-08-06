"""
Unit tests for OverallResultsScraper.
"""
import pytest
from unittest.mock import Mock

from src.scraper.overall_results_scraper import OverallResultsScraper, OverallResultsConfig
from src.scraper.dtos.scraping_result import ScrapingResult


class TestOverallResultsConfig:
    """Tests for OverallResultsConfig."""

    def test_requires_bidding_schedules(self):
        """OverallResultsConfig should require bidding_schedules."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="AY202526T1"
        )
        assert config.bidding_schedules == {}

    def test_requires_start_ay_term(self):
        """OverallResultsConfig should require start_ay_term."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="AY202526T1"
        )
        assert config.start_ay_term == "AY202526T1"

    def test_default_values(self):
        """OverallResultsConfig should have correct defaults."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="AY202526T1"
        )
        assert config.base_url == "https://boss.intranet.smu.edu.sg/OverallResults.aspx"
        assert config.delay == 5
        assert config.headless is True
        assert config.page_size == 50
        assert config.max_retries == 3

    def test_desired_columns_preset(self):
        """OverallResultsScraper should have DESIRED_COLUMNS preset."""
        assert 'Term' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Course Code' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Median Bid' in OverallResultsScraper.DESIRED_COLUMNS


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
            start_ay_term="AY202526T1"
        )
        scraper = OverallResultsScraper(config=config)
        assert scraper._config is config

    def test_term_map_values(self):
        """_TERM_DISPLAY_MAP should have correct values."""
        assert OverallResultsScraper._TERM_DISPLAY_MAP == {
            'T1': 'Term 1',
            'T2': 'Term 2',
            'T3A': 'Term 3A',
            'T3B': 'Term 3B'
        }

    def test_desired_columns_preset(self):
        """DESIRED_COLUMNS should be set correctly."""
        assert 'Term' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Course Code' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Median Bid' in OverallResultsScraper.DESIRED_COLUMNS
        assert 'Min Bid' in OverallResultsScraper.DESIRED_COLUMNS


class TestGenerateFilename:
    """Tests for _generate_filename — AY-format term consistency."""

    def test_ay_term_returns_ay_filename(self):
        """AY term 'AY202526T1' -> 'AY202526T1.xlsx' (no conversion)."""
        config = OverallResultsConfig(bidding_schedules={}, start_ay_term="AY202526T1")
        scraper = OverallResultsScraper(config=config)
        assert scraper._generate_filename("AY202526T1") == "AY202526T1.xlsx"

    def test_ay_term_returns_ay_filename_2(self):
        """AY term 'AY202627T1' -> 'AY202627T1.xlsx' (no conversion)."""
        config = OverallResultsConfig(bidding_schedules={}, start_ay_term="AY202627T1")
        scraper = OverallResultsScraper(config=config)
        assert scraper._generate_filename("AY202627T1") == "AY202627T1.xlsx"


class TestScrapePassesAyTerm:
    """scrape()/_scrape_term_data must keep the AY term as the filename source.

    Regression: the display term ("2026-27 Term 1") used for the BOSS dropdown
    must not leak into the output filename — files must be AY202627T1.xlsx.
    """

    def test_scrape_passes_ay_term_to_scrape_term_data(self, mock_webdriver):
        """scrape() calls _scrape_term_data with the AY term, not the display form."""
        config = OverallResultsConfig(bidding_schedules={}, start_ay_term="AY202627T1")
        scraper = OverallResultsScraper(config=config, driver=mock_webdriver)

        captured = {}

        def fake_scrape_term_data(**kwargs):
            captured['term'] = kwargs.get('term')
            return []

        scraper._scrape_term_data = fake_scrape_term_data

        result = scraper.scrape(term="AY202627T1", bid_round="1", bid_window="1",
                                output_dir="tmp", authenticator=None)

        assert captured['term'] == "AY202627T1"
        assert result.files_saved == 0

    def test_scrape_term_data_uses_display_for_dropdown_ay_for_filename(self, mock_webdriver):
        """_scrape_term_data selects the display term in the dropdown but saves with the AY term."""
        config = OverallResultsConfig(bidding_schedules={}, start_ay_term="AY202627T1")
        scraper = OverallResultsScraper(config=config, driver=mock_webdriver)

        for m in ["_navigate_to_overall_results", "_select_course_career", "_select_bid_round",
                  "_select_bid_window", "_click_search", "_set_page_size_to_50",
                  "_sort_by_bidding_window", "_click_next_page", "_has_next_page"]:
            setattr(scraper, m, Mock(return_value=None))
        scraper._is_no_records_found = Mock(return_value=False)
        scraper._get_current_page_info = Mock(return_value=(1, 1, 0))
        scraper._extract_table_data = Mock(return_value=([{'a': 1}], False, None))
        scraper._select_term = Mock()
        scraper._save_to_excel = Mock()

        scraper._scrape_term_data("AY202627T1", "1", "1", "out")

        # dropdown receives the display term
        assert scraper._select_term.call_args[0][0] == "2026-27 Term 1"
        # filename source stays AY
        assert scraper._save_to_excel.call_args[0][1] == "AY202627T1"


class TestOverallResultsScraperScrape:
    """Tests for OverallResultsScraper.scrape()."""

    def test_scrape_returns_scraping_result(self, mock_webdriver):
        """scrape() should return a ScrapingResult object."""
        config = OverallResultsConfig(
            bidding_schedules={},
            start_ay_term="AY202526T1"
        )
        scraper = OverallResultsScraper(config=config, driver=mock_webdriver)

        # Mock the scrape method
        scraper.scrape = Mock(return_value=ScrapingResult(
            ay_term="AY202526T1",
            round_folder="R1W1",
            files_saved=10,
        ))

        result = scraper.scrape(term="AY202526T1")

        assert isinstance(result, ScrapingResult)
        assert result.files_saved == 10
