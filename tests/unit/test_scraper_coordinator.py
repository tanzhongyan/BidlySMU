"""Unit tests for ScraperCoordinator.run() with mocked driver/authenticator/scraper.

The coordinator owns driver creation (ChromeDriverFactory) internally, so we
mock the factory's ``create`` to return a fake driver and verify the full
workflow: driver -> connect -> login -> connect -> scrape -> quit, plus the
error path when ``result.is_success`` is False.
"""
from unittest import mock

import pytest

from src.scraper.scraper_coordinator import ScraperCoordinator


def _make_scraper(is_success=True, errors=None):
    """Return a (scraper_mock, result_mock) pair with a fake _config."""
    scraper = mock.Mock()
    scraper._config = mock.Mock(headless=True)
    result = mock.Mock(is_success=is_success, errors=errors)
    scraper.scrape.return_value = result
    return scraper, result


def test_run_drives_full_workflow():
    """With an authenticator: create driver, connect, login, reconnect, scrape, quit."""
    driver = mock.Mock()
    scraper, result = _make_scraper()
    authenticator = mock.Mock()
    authenticator.login.return_value = driver

    coordinator = ScraperCoordinator(
        authenticator=authenticator, scraper=scraper, logger=mock.Mock()
    )
    with mock.patch.object(coordinator._driver_factory, "create", return_value=driver):
        out = coordinator.run(acad_term_id="AY202627T1")

    assert out is result
    driver.get.assert_called_once_with("https://boss.intranet.smu.edu.sg/")
    authenticator.login.assert_called_once_with(driver)
    # connect is called twice: before login and again with the returned driver
    assert scraper.connect.call_count == 2
    scraper.scrape.assert_called_once_with(acad_term_id="AY202627T1")
    driver.quit.assert_called_once()


def test_run_without_authenticator_skips_login():
    """With authenticator=None: no BOSS navigation, no login, still scrapes."""
    driver = mock.Mock()
    scraper, result = _make_scraper()

    coordinator = ScraperCoordinator(authenticator=None, scraper=scraper, logger=mock.Mock())
    with mock.patch.object(coordinator._driver_factory, "create", return_value=driver):
        out = coordinator.run(acad_term_id="AY202627T1")

    assert out is result
    driver.get.assert_not_called()
    scraper.connect.assert_called_once_with(driver)
    scraper.scrape.assert_called_once_with(acad_term_id="AY202627T1")
    driver.quit.assert_called_once()


def test_run_raises_when_scrape_fails_and_still_quits_driver():
    """A failed scrape raises AND the driver is still cleaned up in finally."""
    driver = mock.Mock()
    scraper, _ = _make_scraper(is_success=False, errors="boom")

    coordinator = ScraperCoordinator(authenticator=None, scraper=scraper, logger=mock.Mock())
    with mock.patch.object(coordinator._driver_factory, "create", return_value=driver):
        with pytest.raises(Exception, match="Scrape failed: boom"):
            coordinator.run(acad_term_id="AY202627T1")

    driver.quit.assert_called_once()
