"""
Unit tests for ScraperCoordinator.
"""
import pytest
from unittest.mock import MagicMock, Mock
from datetime import datetime

from src.scraper.coordinator import ScraperCoordinator
from src.driver.driver_factory import ChromeDriverFactory
from src.models.dto.scraping_result import ScrapingResult


class TestScraperCoordinator:
    """Tests for ScraperCoordinator."""

    def test_requires_driver_factory(self):
        """ScraperCoordinator should require driver_factory."""
        mock_scraper = MagicMock()
        coordinator = ScraperCoordinator(
            driver_factory=MagicMock(),
            authenticator=None,
            scraper=mock_scraper,
        )
        assert coordinator._driver_factory is not None

    def test_requires_scraper(self):
        """ScraperCoordinator should require scraper."""
        coordinator = ScraperCoordinator(
            driver_factory=MagicMock(),
            authenticator=None,
            scraper=MagicMock(),
        )
        assert coordinator._scraper is not None

    def test_accepts_none_authenticator(self):
        """ScraperCoordinator should accept None authenticator."""
        coordinator = ScraperCoordinator(
            driver_factory=MagicMock(),
            authenticator=None,
            scraper=MagicMock(),
        )
        assert coordinator._authenticator is None

    def test_run_creates_driver(self):
        """run() should create driver via factory."""
        mock_driver = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_driver

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
        )

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=None,
            scraper=mock_scraper,
        )

        coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        mock_factory.create.assert_called_once()
        mock_scraper.connect.assert_called_once_with(mock_driver)

    def test_run_calls_scraper_scrape(self):
        """run() should call scraper.scrape() with kwargs."""
        mock_factory = MagicMock()
        mock_factory.create.return_value = MagicMock()

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
        )

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=None,
            scraper=mock_scraper,
        )

        result = coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        mock_scraper.scrape.assert_called_once()
        assert result.files_saved == 10

    def test_run_quits_driver_on_success(self):
        """run() should quit driver on successful scrape."""
        mock_driver = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_driver

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
        )

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=None,
            scraper=mock_scraper,
        )

        coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        mock_driver.quit.assert_called_once()

    def test_run_raises_on_scraper_failure(self):
        """run() should raise Exception when scraper returns failed result."""
        mock_factory = MagicMock()
        mock_factory.create.return_value = MagicMock()

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=0,
            errors=[],
        )

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=None,
            scraper=mock_scraper,
        )

        with pytest.raises(Exception, match="Scrape failed"):
            coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

    def test_run_uses_authenticator_when_provided(self):
        """run() should use authenticator to login if provided."""
        mock_driver = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_driver

        mock_authenticator = MagicMock()

        mock_scraper = MagicMock()
        mock_scraper.scrape.return_value = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
        )

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=mock_authenticator,
            scraper=mock_scraper,
        )

        coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        mock_authenticator.login.assert_called_once_with(mock_driver)

    def test_run_quits_driver_on_exception(self):
        """run() should quit driver even when exception occurs."""
        mock_driver = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_driver

        mock_scraper = MagicMock()
        mock_scraper.connect.side_effect = Exception("Connection failed")

        coordinator = ScraperCoordinator(
            driver_factory=mock_factory,
            authenticator=None,
            scraper=mock_scraper,
        )

        with pytest.raises(Exception):
            coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")

        mock_driver.quit.assert_called_once()
