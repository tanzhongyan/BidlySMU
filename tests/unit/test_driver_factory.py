"""
Unit tests for ChromeDriverFactory.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from src.driver.driver_factory import ChromeDriverFactory


class TestChromeDriverFactory:
    """Tests for ChromeDriverFactory."""

    def test_init_with_defaults(self):
        """Factory should initialize with default values."""
        factory = ChromeDriverFactory()
        assert factory.headless is False
        assert factory.no_sandbox is True
        assert factory.disable_dev_shm_usage is True
        assert factory.disable_gpu is True
        assert factory.window_size is None
        assert factory.user_agent is None
        assert factory.arguments == []

    def test_init_with_custom_values(self):
        """Factory should accept custom values."""
        factory = ChromeDriverFactory(
            headless=True,
            window_size="1920,1080",
            user_agent="TestAgent/1.0",
            arguments=["--some-flag"],
        )
        assert factory.headless is True
        assert factory.window_size == "1920,1080"
        assert factory.user_agent == "TestAgent/1.0"
        assert "--some-flag" in factory.arguments

    def test_create_options_headless(self):
        """_create_options should add --headless=new when headless=True."""
        factory = ChromeDriverFactory(headless=True)
        options = factory._create_options()
        args = options.arguments
        assert "--headless=new" in args

    def test_create_options_no_headless(self):
        """_create_options should not add headless flag when headless=False."""
        factory = ChromeDriverFactory(headless=False)
        options = factory._create_options()
        args = options.arguments
        assert "--headless=new" not in args

    def test_create_options_adds_no_sandbox(self):
        """_create_options should add --no-sandbox by default."""
        factory = ChromeDriverFactory()
        options = factory._create_options()
        args = options.arguments
        assert "--no-sandbox" in args

    def test_create_options_adds_disable_dev_shm_usage(self):
        """_create_options should add --disable-dev-shm-usage by default."""
        factory = ChromeDriverFactory()
        options = factory._create_options()
        args = options.arguments
        assert "--disable-dev-shm-usage" in args

    def test_create_options_adds_disable_gpu(self):
        """_create_options should add --disable-gpu by default."""
        factory = ChromeDriverFactory()
        options = factory._create_options()
        args = options.arguments
        assert "--disable-gpu" in args

    def test_create_options_with_window_size(self):
        """_create_options should add --window-size when specified."""
        factory = ChromeDriverFactory(window_size="1920,1080")
        options = factory._create_options()
        args = options.arguments
        assert "--window-size=1920,1080" in args

    def test_create_options_with_user_agent(self):
        """_create_options should add --user-agent when specified."""
        factory = ChromeDriverFactory(user_agent="CustomAgent/1.0")
        options = factory._create_options()
        args = options.arguments
        assert "--user-agent=CustomAgent/1.0" in args

    def test_create_options_adds_experimental_options(self):
        """_create_options should add BOSS-specific experimental options."""
        factory = ChromeDriverFactory()
        options = factory._create_options()
        # Check excludeSwitches is set
        prefs = options.experimental_options
        assert "excludeSwitches" in prefs
        assert "enable-logging" in prefs["excludeSwitches"]

    @patch("src.driver.driver_factory.Chrome")
    @patch("src.driver.driver_factory.ChromeDriverManager")
    def test_create_returns_chrome_instance(
        self, mock_driver_manager, mock_chrome
    ):
        """create() should return a Chrome WebDriver instance."""
        mock_service = MagicMock()
        mock_driver_manager.return_value.install.return_value = "/path/to/chromedriver"
        mock_chrome_instance = MagicMock()
        mock_chrome.return_value = mock_chrome_instance

        factory = ChromeDriverFactory()
        result = factory.create()

        mock_chrome.assert_called_once()
        assert result is mock_chrome_instance

    @patch("src.driver.driver_factory.Chrome")
    @patch("src.driver.driver_factory.ChromeDriverManager")
    def test_create_with_options_passes_options(
        self, mock_driver_manager, mock_chrome
    ):
        """create_with_options() should pass the provided Options object."""
        mock_driver_manager.return_value.install.return_value = "/path/to/chromedriver"
        mock_chrome_instance = MagicMock()
        mock_chrome.return_value = mock_chrome_instance

        custom_options = Options()
        custom_options.add_argument("--custom-flag")

        factory = ChromeDriverFactory()
        factory.create_with_options(custom_options)

        # Check that Chrome was called with the custom options
        call_kwargs = mock_chrome.call_args[1]
        assert call_kwargs["options"] is custom_options
