"""
Abstract scraper base class with Smart Wait Wrapper for handling StaleElementReferenceException.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
import logging

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    ElementNotInteractableException,
)

from src.logging.logger import get_logger
from src.scraper.dtos.scraping_result import ScrapingResult


class StaleElementError(Exception):
    """Raised when element remains stale after max retries."""
    pass


class AbstractScraper(ABC):
    """
    Abstract base class for all BOSS scrapers.

    Provides:
    - Encapsulated WebDriver management via @property
    - Smart Wait Wrapper with automatic StaleElementReferenceException recovery
    - Standard wait strategies
    - Logging via LoggerFactory (supports Sentry)

    SUBCLASSES MUST IMPLEMENT (abstract):
    - connect(driver) - establish connection to BOSS
    - scrape(**kwargs) - perform scraping operation

    SUBCLASSES INHERIT (concrete - don't override unless needed):
    - disconnect() - cleans up and quits driver (same for all scrapers)
    - safe_click(), safe_send_keys(), safe_get_text() - Smart Wait wrappers
    - take_screenshot() - debugging utility

    Usage:
        class MyScraper(AbstractScraper):
            def connect(self, driver):
                self._driver = driver
            def scrape(self, **kwargs):
                return ScrapingResult(ay_term="...", round_folder="...")

        # Create and use
        scraper = MyScraper()
        scraper.logger.info("Starting...")

        # Dependency injection
        mock_driver = Mock()
        scraper.driver = mock_driver  # Uses @driver.setter
    """

    @dataclass
    class ScraperConfig:
        """Configuration for scraper behavior."""
        headless: bool = False
        timeout: int = 30
        delay_between_requests: float = 1.0
        max_retries: int = 3
        base_url: str = "https://boss.intranet.smu.edu.sg"
        screenshot_on_error: bool = True

    def __init__(
        self,
        driver: Optional[WebDriver] = None,
        config: Optional[ScraperConfig] = None,
        logger: Optional["logging.Logger"] = None,
    ):
        self._driver = driver
        self._config = config or self.ScraperConfig()
        # Use injected logger or create one using LoggerFactory (supports Sentry)
        self._logger = logger or get_logger(self.__class__.__name__)
        self._max_retries = self._config.max_retries

    # ==================== Properties ====================

    @property
    def driver(self) -> WebDriver:
        """Encapsulated driver access - raises if not initialized."""
        if self._driver is None:
            raise RuntimeError(
                "WebDriver not initialized. Call connect(driver) first or inject via setter."
            )
        return self._driver

    @driver.setter
    def driver(self, value: WebDriver) -> None:
        """
        Allow dependency injection of pre-configured driver.

        Usage:
            scraper = MyScraper()
            scraper.driver = mock_driver  # Injects mock for testing
        """
        self._driver = value

    @property
    def config(self) -> "AbstractScraper.ScraperConfig":
        """Access configuration."""
        return self._config

    @property
    def logger(self) -> "logging.Logger":
        """Access logger."""
        return self._logger

    # ==================== Smart Wait Wrapper ====================

    def wait_for_presence(
        self,
        by: By,
        value: str,
        timeout: Optional[int] = None,
    ) -> "WebElement":
        """
        Waits for element to be present in DOM with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            timeout: Optional override for default timeout

        Returns:
            WebElement if found

        Raises:
            StaleElementError: If element remains stale after max retries
            TimeoutException: If element not found within timeout
        """
        return self._smart_wait(
            EC.presence_of_element_located((by, value)),
            timeout=timeout,
        )

    def wait_for_visibility(
        self,
        by: By,
        value: str,
        timeout: Optional[int] = None,
    ) -> "WebElement":
        """
        Waits for element to be visible with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            timeout: Optional override for default timeout

        Returns:
            WebElement if visible

        Raises:
            StaleElementError: If element remains stale after max retries
            TimeoutException: If element not visible within timeout
        """
        return self._smart_wait(
            EC.visibility_of_element_located((by, value)),
            timeout=timeout,
        )

    def wait_for_clickable(
        self,
        by: By,
        value: str,
        timeout: Optional[int] = None,
    ) -> "WebElement":
        """
        Waits for element to be clickable with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            timeout: Optional override for default timeout

        Returns:
            WebElement if clickable

        Raises:
            StaleElementError: If element remains stale after max retries
            TimeoutException: If element not clickable within timeout
        """
        return self._smart_wait(
            EC.element_to_be_clickable((by, value)),
            timeout=timeout,
        )

    def wait_for_any_of(
        self,
        *conditions,
        timeout: Optional[int] = None,
    ) -> "WebElement":
        """
        Wait for any of multiple conditions to be met.

        Args:
            conditions: Multiple EC conditions
            timeout: Optional override for default timeout

        Returns:
            First WebElement that satisfies any condition
        """
        return self._smart_wait(EC.any_of(*conditions), timeout=timeout)

    def wait_for_none(
        self,
        by: By,
        value: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for element to no longer be present in DOM.

        Args:
            by: Selenium By locator type
            value: Locator value
            timeout: Optional override for default timeout

        Returns:
            True if element gone, False if still present
        """
        wait = WebDriverWait(self.driver, timeout or self._config.timeout)
        try:
            wait.until(EC.invisibility_of_element_located((by, value)))
            return True
        except TimeoutException:
            return False

    def _smart_wait(
        self,
        condition,
        timeout: Optional[int] = None,
    ) -> "WebElement":
        """
        Internal smart wait with automatic stale element recovery.

        Repeatedly tries the condition until it succeeds or max retries reached.
        _smart_wait handles ALL stale element retries - safe_click should NOT add
        another retry layer on top.
        """
        wait = WebDriverWait(
            self.driver,
            timeout or self._config.timeout,
            ignored_exceptions=[StaleElementReferenceException],
        )

        last_exception = None
        for attempt in range(self._max_retries):
            try:
                return wait.until(condition)
            except StaleElementReferenceException:
                last_exception = StaleElementReferenceException(
                    f"Element remained stale after {attempt + 1} attempts"
                )
                self._logger.warning(
                    f"Stale element detected, retrying ({attempt + 1}/{self._max_retries})"
                )
                time.sleep(0.5)
            except TimeoutException:
                # Re-raise timeout as-is (not a stale element issue)
                raise

        # All retries exhausted
        raise StaleElementError(
            f"Element remained stale after {self._max_retries} retries"
        ) from last_exception

    # ==================== Safe Operations ====================

    def safe_click(
        self,
        by: By,
        value: str,
    ) -> None:
        """
        Clicks element with automatic stale element recovery.

        Note: The _smart_wait in wait_for_clickable already handles stale element
        retries. This method performs the click - if click fails due to stale element
        post-wait, the exception propagates up.

        Args:
            by: Selenium By locator type
            value: Locator value

        Raises:
            StaleElementError: If element remains stale after retries from _smart_wait
            ElementNotInteractableException: If element cannot be clicked
            ElementClickInterceptedException: If element is covered by another
        """
        element = self.wait_for_clickable(by, value)
        try:
            element.click()
        except ElementNotInteractableException:
            # Try scrolling into view first
            self._logger.warning("Element not interactable, trying scroll into view")
            raw_element = self.driver.find_element(by, value)
            self.driver.execute_script("arguments[0].scrollIntoView(true);", raw_element)
            time.sleep(0.5)
            raw_element.click()

    def safe_send_keys(
        self,
        by: By,
        value: str,
        text: str,
        clear_first: bool = True,
    ) -> None:
        """
        Sends keys to element with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            text: Text to send
            clear_first: Whether to clear existing text first
        """
        element = self.wait_for_visibility(by, value)
        if clear_first:
            element.clear()
        element.send_keys(text)

    def safe_get_text(
        self,
        by: By,
        value: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Gets text from element with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            default: Default value if text not found

        Returns:
            Element text or default if not found
        """
        try:
            element = self.wait_for_presence(by, value)
            return element.text.strip() if element.text else default
        except (TimeoutException, StaleElementError, NoSuchElementException):
            return default

    def safe_get_attribute(
        self,
        by: By,
        value: str,
        attribute: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Gets attribute from element with automatic stale element recovery.

        Args:
            by: Selenium By locator type
            value: Locator value
            attribute: Attribute name
            default: Default value if not found

        Returns:
            Attribute value or default if not found
        """
        try:
            element = self.wait_for_presence(by, value)
            return element.get_attribute(attribute) or default
        except (TimeoutException, StaleElementError, NoSuchElementException):
            return default

    # ==================== Screenshot Utility ====================

    def take_screenshot(self, path: str) -> None:
        """
        Takes a screenshot for debugging purposes.

        Args:
            path: Full path including filename for screenshot
        """
        try:
            self.driver.save_screenshot(path)
            self._logger.info(f"Screenshot saved to {path}")
        except Exception as e:
            self._logger.error(f"Failed to take screenshot: {e}")

    # ==================== Concrete Methods ====================

    def connect(self, driver: WebDriver) -> None:
        """
        CONCRETE METHOD: Assign the provided WebDriver.

        All scrapers connect the same way - inherited automatically.
        Subclasses can override if they need custom connect logic,
        but should call super().connect() to ensure proper assignment.
        """
        self._driver = driver

    def disconnect(self) -> None:
        """
        CONCRETE METHOD: Clean up connections and close driver.

        All scrapers disconnect the same way - inherited automatically.
        Subclasses can override if they need custom cleanup, but should
        call super().disconnect() to ensure proper driver cleanup.
        """
        if self._driver:
            self._driver.quit()
        self._driver = None

    @abstractmethod
    def scrape(self, **kwargs) -> "ScrapingResult":
        """
        Perform scraping operation.

        Returns:
            ScrapingResult with operation outcome

        Raises:
            NotImplementedError: If driver not connected
        """
        pass

    # ==================== Context Manager ====================

    def __enter__(self) -> "AbstractScraper":
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures disconnect."""
        self.disconnect()
        return False  # Don't suppress exceptions