"""
Unit tests for ClassScraper.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime

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

        result = scraper.scrape(acad_term_id="AY202526T3A")

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


class TestScrapeSingleClass:
    """Tests for _scrape_single_class method."""

    def test_returns_true_when_file_saved(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_single_class should return True when HTML is saved successfully."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock driver.get to capture the page
        mock_webdriver.get.return_value = None
        mock_webdriver.page_source = "<html>Class Details Here</html>"

        # Mock wait_for_any_of to not raise
        scraper.wait_for_any_of = Mock()

        result = scraper._scrape_single_class(
            mock_webdriver, output_dir, "25", "31", 1001
        )

        # Should return True when file is saved
        assert result is True

        # File should exist
        expected_file = output_dir / "SelectedAcadTerm=2531&SelectedClassNumber=1001.html"
        assert expected_file.exists()

    def test_returns_false_when_no_record_found(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_single_class should return False when page shows 'No record found'."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock page_source with "No record found"
        mock_webdriver.page_source = "<html><body>No record found</body></html>"
        mock_webdriver.get.return_value = None

        scraper.wait_for_any_of = Mock()

        result = scraper._scrape_single_class(
            mock_webdriver, output_dir, "25", "31", 1001
        )

        assert result is False

        # No file should be created
        expected_file = output_dir / "SelectedAcadTerm=2531&SelectedClassNumber=1001.html"
        assert not expected_file.exists()

    def test_returns_none_on_error(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_single_class should return None when an exception occurs."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock driver.get to raise an exception
        mock_webdriver.get.side_effect = Exception("Network error")

        result = scraper._scrape_single_class(
            mock_webdriver, output_dir, "25", "31", 1001
        )

        assert result is None

    def test_generates_correct_url(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_single_class should generate URL with correct format."""
        config = ClassScraperConfig(
            bidding_schedules={},
            base_url="https://boss.intranet.smu.edu.sg"
        )
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_webdriver.get.return_value = None
        mock_webdriver.page_source = "<html>Saved</html>"
        scraper.wait_for_any_of = Mock()

        scraper._scrape_single_class(mock_webdriver, output_dir, "26", "10", 1234)

        # Verify URL was called with correct parameters
        call_args = mock_webdriver.get.call_args
        url_called = call_args[0][0]
        assert "SelectedClassNumber=1234" in url_called
        assert "SelectedAcadTerm=2610" in url_called
        assert "ClassDetails.aspx" in url_called

    def test_handles_stale_element_retry(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_single_class should retry on StaleElementError."""
        from src.scraper.abstract_scraper import StaleElementError

        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # First call raises StaleElementError, second succeeds
        mock_webdriver.page_source = "<html>Success</html>"
        scraper.wait_for_any_of = Mock(side_effect=[StaleElementError(), None])

        call_count = [0]

        def get_side_effect(*_args):
            call_count[0] += 1
            return None

        mock_webdriver.get.side_effect = get_side_effect

        result = scraper._scrape_single_class(
            mock_webdriver, output_dir, "25", "31", 1001
        )

        # Should have retried (get called twice)
        assert call_count[0] == 2
        assert result is True


class TestScrapeRange:
    """Tests for _scrape_range method."""

    def test_stops_at_consecutive_empty_threshold(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_range should stop after consecutive_empty_threshold empty records."""
        config = ClassScraperConfig(
            bidding_schedules={},
            consecutive_empty_threshold=5
        )
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock _scrape_single_class to return False (empty) every time
        mock_fn = Mock(side_effect=[False]*5)
        scraper._scrape_single_class = mock_fn

        files_saved = scraper._scrape_range(mock_webdriver, "2025-26_T1", output_dir)

        # Should have stopped at threshold
        assert files_saved == 0
        # Should have been called exactly threshold times (stops AT threshold)
        assert mock_fn.call_count == 5

    def test_counts_files_saved(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_range should count files_saved correctly."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock: return True for first 3, then False
        mock_fn = Mock(side_effect=[True, True, True, False, False, False])
        scraper._scrape_single_class = mock_fn

        files_saved = scraper._scrape_range(mock_webdriver, "2025-26_T1", output_dir)

        assert files_saved == 3

    def test_resets_consecutive_empty_on_success(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_range should reset consecutive_empty counter on successful save."""
        config = ClassScraperConfig(
            bidding_schedules={},
            consecutive_empty_threshold=5
        )
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock: True, False, False, False, False, False, True (resets after success)
        scraper._scrape_single_class = Mock(side_effect=[True, False, False, False, False, True, False])

        files_saved = scraper._scrape_range(mock_webdriver, "2025-26_T1", output_dir)

        # Should not have stopped at threshold because of the True in middle
        assert files_saved == 2

    def test_breaks_on_error(self, mock_webdriver, mock_logger, tmp_path):
        """_scrape_range should break when _scrape_single_class returns None (error)."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock: False, False, None (error on 3rd call)
        scraper._scrape_single_class = Mock(side_effect=[False, False, None])

        files_saved = scraper._scrape_range(mock_webdriver, "2025-26_T1", output_dir)

        # Should stop at None (error)
        assert files_saved == 0
        assert scraper._scrape_single_class.call_count == 3


class TestScrapeMethod:
    """Tests for scrape() method edge cases."""

    def test_skips_when_not_in_bidding_window(self, mock_webdriver, mock_logger):
        """scrape() should return early when not in a bidding window."""
        # Create a schedule where no dates are in the future (effectively no active window)
        past_schedule = [
            (datetime(2020, 1, 1), "Round 1", "WIN1"),  # All past dates
        ]

        config = ClassScraperConfig(
            bidding_schedules={"2025-26_T1": past_schedule},
            start_ay_term="2025-26_T1"
        )
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        # This would skip because no round folder is determined
        result = scraper.scrape(acad_term_id="AY202526T1")

        # Should return without scraping
        assert result.files_saved == 0

    def test_requires_acad_term_id(self, mock_webdriver, mock_logger):
        """scrape() should handle missing acad_term_id gracefully."""
        config = ClassScraperConfig(bidding_schedules={})
        scraper = ClassScraper(config=config, driver=mock_webdriver, logger=mock_logger)

        result = scraper.scrape(acad_term_id=None)

        # Should return an error result
        assert result.ay_term == "unknown"
