"""
Shared pytest fixtures for unit tests.
"""
import pytest
from unittest.mock import MagicMock, Mock
import requests

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from src.driver.driver_factory import ChromeDriverFactory
from src.driver.authenticator import AuthCredentials
from src.scraper.trumba_client import TrubaConfig


# ==============================================================================
# Selenium/Web Driver Fixtures (for BOSS scraper tests)
# ==============================================================================

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


# ==============================================================================
# Truba API Fixtures (for calendar API tests)
# ==============================================================================

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock = MagicMock()
    mock.info.return_value = None
    mock.warning.return_value = None
    mock.error.return_value = None
    mock.debug.return_value = None
    return mock


@pytest.fixture
def truba_config():
    """TrubaConfig for testing."""
    return TrubaConfig(
        api_url="https://test.trumba.url/api.json",
        months_ahead=6,
        timeout=30
    )


@pytest.fixture
def mock_requests():
    """Mock requests.get for HTTP testing."""
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None

    mock_get = MagicMock(return_value=mock_response)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(requests, "get", mock_get)
        yield {"get": mock_get, "response": mock_response}


@pytest.fixture
def sample_bidding_schedules():
    """Sample bidding_schedules.json data."""
    return {
        "AY202627T1": [
            ["2026-07-08T14:00:00", "Round 1 Window 1", "R1W1"],
            ["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1"]
        ]
    }


# ==============================================================================
# AWS/Supabase Fixtures (for Lambda tests)
# ==============================================================================

@pytest.fixture
def mock_eventbridge_client():
    """Mock boto3 EventBridge scheduler client."""
    mock = MagicMock()
    mock.exceptions.ResourceNotFoundException = type('ResourceNotFoundException', (Exception,), {})
    mock.get_schedule.side_effect = mock.exceptions.ResourceNotFoundException()
    mock.create_schedule.return_value = {}
    return mock


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for storage operations."""
    mock = MagicMock()
    mock.storage.from_.return_value.download.return_value = b'{}'
    mock.storage.from_.return_value.upload.return_value = {}
    return mock