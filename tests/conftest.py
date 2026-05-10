"""
Shared pytest fixtures for unit tests.
"""
import pytest
from unittest.mock import MagicMock, Mock

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from src.driver.driver_factory import ChromeDriverFactory
from src.driver.authenticator import AuthCredentials


@pytest.fixture
def mock_webdriver():
    """Mock WebDriver for unit tests."""
    mock = MagicMock(spec=WebDriver)
    mock.get.return_value = None
    mock.page_source = "<html><body>Test Page</body></html>"
    mock.quit.return_value = None
    return mock


@pytest.fixture
def mock_webelement():
    """Mock WebElement for unit tests."""
    mock = MagicMock(spec=WebElement)
    mock.text = "Sample Text"
    mock.get_attribute.return_value = "value"
    mock.is_enabled.return_value = True
    mock.is_displayed.return_value = True
    mock.click.return_value = None
    mock.send_keys.return_value = None
    mock.clear.return_value = None
    return mock


@pytest.fixture
def chrome_driver_factory():
    """ChromeDriverFactory instance for testing."""
    return ChromeDriverFactory(headless=True)


@pytest.fixture
def auth_credentials():
    """Valid test credentials."""
    return AuthCredentials(
        email="test@business.smu.edu.sg",
        password="test_password",
        mfa_secret="SECRET123SECRET123"  # Valid base32 string
    )


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock = MagicMock()
    mock.info.return_value = None
    mock.warning.return_value = None
    mock.error.return_value = None
    mock.debug.return_value = None
    return mock
