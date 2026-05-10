"""
Unit tests for AbstractScraper.
"""
import pytest
from unittest.mock import Mock
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
)

from src.scraper.abstract_scraper import AbstractScraper


class DummyScraper(AbstractScraper):
    """Concrete implementation of AbstractScraper for testing."""

    def scrape(self, **kwargs):
        return Mock(files_saved=0, is_success=True, errors=[])


class TestAbstractScraper:
    """Tests for AbstractScraper."""

    def test_driver_property_raises_when_not_initialized(self):
        """Accessing driver property without driver should raise RuntimeError."""
        scraper = DummyScraper()
        with pytest.raises(RuntimeError, match="WebDriver not initialized"):
            _ = scraper.driver

    def test_driver_property_returns_driver_when_set(self, mock_webdriver):
        """Accessing driver property should return the injected driver."""
        scraper = DummyScraper()
        scraper.driver = mock_webdriver
        assert scraper.driver is mock_webdriver

    def test_connect_sets_driver(self, mock_webdriver):
        """connect() should set the internal driver."""
        scraper = DummyScraper()
        scraper.connect(mock_webdriver)
        assert scraper._driver is mock_webdriver

    def test_disconnect_quits_driver(self, mock_webdriver):
        """disconnect() should quit the driver and set it to None."""
        scraper = DummyScraper()
        scraper.connect(mock_webdriver)
        scraper.disconnect()
        mock_webdriver.quit.assert_called_once()
        assert scraper._driver is None

    def test_safe_get_text_returns_text(self, mock_webdriver, mock_webelement):
        """safe_get_text should return element text."""
        mock_webelement.text = "Test Element Text"
        mock_webdriver.find_element.return_value = mock_webelement

        scraper = DummyScraper()
        scraper.connect(mock_webdriver)

        result = scraper.safe_get_text(By.ID, "test-element")
        assert result == "Test Element Text"

    def test_safe_get_text_returns_default_on_no_such_element(
        self, mock_webdriver
    ):
        """safe_get_text should return default when element not found."""
        mock_webdriver.find_element.side_effect = NoSuchElementException("Not found")

        scraper = DummyScraper()
        scraper.connect(mock_webdriver)

        result = scraper.safe_get_text(By.ID, "nonexistent", default="DEFAULT")
        assert result == "DEFAULT"

    def test_safe_get_attribute(self, mock_webdriver, mock_webelement):
        """safe_get_attribute should return element attribute."""
        mock_webelement.get_attribute.return_value = "href-value"
        mock_webdriver.find_element.return_value = mock_webelement

        scraper = DummyScraper()
        scraper.connect(mock_webdriver)

        result = scraper.safe_get_attribute(By.ID, "test-element", "href")
        assert result == "href-value"

    def test_safe_click_clicks_element(self, mock_webdriver, mock_webelement):
        """safe_click should click the element."""
        mock_webdriver.find_element.return_value = mock_webelement
        mock_webelement.is_enabled.return_value = True
        mock_webelement.is_displayed.return_value = True

        scraper = DummyScraper()
        scraper.connect(mock_webdriver)

        scraper.safe_click(By.ID, "test-button")
        mock_webelement.click.assert_called_once()

    def test_context_manager_calls_disconnect(self, mock_webdriver):
        """Using scraper as context manager should call disconnect on exit."""
        scraper = DummyScraper()
        with scraper:
            scraper.connect(mock_webdriver)
        mock_webdriver.quit.assert_called_once()
