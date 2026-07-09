"""
Chrome WebDriver factory for creating configured driver instances.
"""
import os
import uuid
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
        driver = Chrome(service=service, options=self._create_options())
        if self.headless:
            self._apply_anti_detection_cdp(driver)
        return driver

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

        # Anti-detection: hide automation signals from anti-bot systems.
        # Microsoft Entra ID detects headless Chrome and refuses to set
        # session cookies unless we mask these indicators.
        # Uses excludeSwitches + CDP navigator.webdriver patch only —
        # avoids --disable-blink-features (causes crashes in Chrome 150)
        # and cross-OS user-agent spoofing (confuses Linux Chrome).
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation", "enable-logging"]
        )
        options.add_experimental_option("useAutomationExtension", False)

        if self.user_agent:
            options.add_argument(f"--user-agent={self.user_agent}")

        # Stability: let Chrome use default multi-process architecture.
        # A single renderer crash (e.g. from heavy ASP.NET pages) won't
        # kill the entire browser. Memory usage verified at ~350MB peak
        # with 2GB allocated — ample headroom.
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")

        for arg in self.arguments:
            options.add_argument(arg)

        # Unique profile per instance prevents lock contention when
        # parallel Chrome instances run (e.g. Stream A + Stream B).
        profile_dir = f"/tmp/chrome-profile-{uuid.uuid4().hex[:8]}"
        options.add_argument(f"--user-data-dir={profile_dir}")

        return options

    def _apply_anti_detection_cdp(self, driver: Chrome) -> None:
        """Hide navigator.webdriver flag via CDP to avoid bot detection."""
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": (
                        "Object.defineProperty(navigator, 'webdriver', "
                        "{get: () => undefined})"
                    )
                },
            )
        except Exception:
            pass  # CDP may not be available in all Chrome versions