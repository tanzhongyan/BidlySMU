"""
Chrome WebDriver factory for creating configured driver instances.
"""
from typing import Optional, List

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class ChromeDriverFactory:
    """
    Factory for creating Chrome WebDriver instances.

    Usage:
        factory = ChromeDriverFactory(headless=True, window_size="1920,1080")
        driver = factory.create()

    Or with default config:
        factory = ChromeDriverFactory()
        driver = factory.create()
    """

    def __init__(
        self,
        headless: bool = False,
        no_sandbox: bool = True,
        disable_dev_shm_usage: bool = True,
        disable_gpu: bool = True,
        window_size: Optional[str] = None,
        user_agent: Optional[str] = None,
        arguments: Optional[List[str]] = None,
    ):
        self.headless = headless
        self.no_sandbox = no_sandbox
        self.disable_dev_shm_usage = disable_dev_shm_usage
        self.disable_gpu = disable_gpu
        self.window_size = window_size
        self.user_agent = user_agent
        self.arguments = arguments or []

    def create(self) -> Chrome:
        """
        Create and return a new Chrome WebDriver instance.

        Returns:
            Configured Chrome WebDriver instance

        Uses webdriver-manager to automatically handle ChromeDriver installation.
        """
        service = Service(ChromeDriverManager().install())
        return Chrome(service=service, options=self._create_options())

    def create_with_options(self, options: Options) -> Chrome:
        """
        Create driver with custom Options object.

        Args:
            options: Custom Chrome Options object

        Returns:
            Chrome WebDriver with provided options
        """
        service = Service(ChromeDriverManager().install())
        return Chrome(service=service, options=options)

    def create_with_defaults(self) -> Chrome:
        """
        Create driver with settings suitable for BOSS scraping.

        Returns:
            Chrome WebDriver configured for BOSS
        """
        factory = ChromeDriverFactory(
            headless=False,  # BOSS requires interactive mode
            window_size="1920,1080",
        )
        return factory.create()

    def _create_options(self) -> Options:
        """Convert factory settings to Chrome Options object."""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")
        if self.no_sandbox:
            options.add_argument("--no-sandbox")
        if self.disable_dev_shm_usage:
            options.add_argument("--disable-dev-shm-usage")
        if self.disable_gpu:
            options.add_argument("--disable-gpu")
        if self.window_size:
            options.add_argument(f"--window-size={self.window_size}")
        if self.user_agent:
            options.add_argument(f"--user-agent={self.user_agent}")

        for arg in self.arguments:
            options.add_argument(arg)

        # BOSS-specific options
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)

        return options