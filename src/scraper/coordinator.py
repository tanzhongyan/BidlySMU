"""
ScraperCoordinator - wires DriverFactory + Authenticator + ScraperEngine.

Usage:
    credentials = AuthCredentials.from_environment()
    coordinator = ScraperCoordinator(
        driver_factory=ChromeDriverFactory(),
        authenticator=AutomatedLogin(credentials),
        scraper=ClassScraper(),
    )
    result = coordinator.run(start_ay_term="2025-26_T1", end_ay_term="2025-26_T1")
"""
from typing import Optional

from src.driver.driver_factory import ChromeDriverFactory
from src.driver.authenticator import Authenticator
from src.base.base_scraper import BaseScraper
from src.logging.logger import get_logger


class ScraperCoordinator:
    """
    Wires DriverFactory + Authenticator + ScraperEngine.

    Equivalent to original run_full_scraping_process():
    1. Create driver
    2. Login (manual or automated via Authenticator)
    3. Call scraper.scrape()
    4. Cleanup driver
    """

    def __init__(
        self,
        driver_factory: ChromeDriverFactory,
        authenticator: Optional[Authenticator],
        scraper: BaseScraper,
        logger=None,
    ):
        """
        Initialize coordinator.

        Args:
            driver_factory: Factory to create WebDriver
            authenticator: Authenticator instance (ManualLogin or AutomatedLogin)
            scraper: ScraperEngine implementation
            logger: Optional logger
        """
        self._driver_factory = driver_factory
        self._authenticator = authenticator
        self._scraper = scraper
        self._logger = logger or get_logger(self.__class__.__name__)

    def run(self, **kwargs):
        """
        Run the scraping workflow.

        Kwargs passed to scraper.scrape().

        Returns:
            Result from scraper.scrape()
        """
        driver = None

        try:
            # Create driver
            self._logger.info("Creating WebDriver...")
            driver = self._driver_factory.create()

            # Connect scraper to driver
            self._scraper.connect(driver)
            self._logger.info("Scraper connected to driver")

            # Login if authenticator provided
            if self._authenticator:
                self._logger.info("Navigating to BOSS...")
                driver.get("https://boss.intranet.smu.edu.sg/")

                self._logger.info("Performing login...")
                self._authenticator.login(driver)
                self._logger.info("Login successful")

            # Run scraping
            self._logger.info("Starting scrape...")
            result = self._scraper.scrape(**kwargs)

            if not result.is_success:
                error_msg = f"Scrape failed: {result.errors}"
                self._logger.error(error_msg)
                raise Exception(error_msg)

            self._logger.info(f"Scrape completed: {result}")
            return result

        except Exception as e:
            self._logger.error(f"Scraping workflow failed: {e}")
            raise

        finally:
            # Cleanup
            if driver:
                try:
                    driver.quit()
                    self._logger.info("WebDriver quit")
                except Exception as e:
                    self._logger.warning(f"Error quitting driver: {e}")
