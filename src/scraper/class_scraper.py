"""
Refactored ClassScraper - implements ScraperEngine for BOSS class details.

Usage:
    # With manual login
    scraper = ClassScraper()
    with scraper:
        driver = factory.create()
        scraper.connect(driver)
        authenticator = ManualLogin()
        authenticator.login(driver)
        result = scraper.scrape(start_ay_term="2024-25_T1", end_ay_term="2024-25_T1")

    # With automated login
    credentials = AuthCredentials.from_environment()
    scraper = ClassScraper()
    with scraper:
        driver = factory.create()
        scraper.connect(driver)
        authenticator = AutomatedLogin(credentials)
        authenticator.login(driver)
        result = scraper.scrape(start_ay_term="2024-25_T1", end_ay_term="2024-25_T1")
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import csv
import os
import time
from pathlib import Path

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from src.base.base_scraper import BaseScraper, StaleElementError
from src.driver.authenticator import Authenticator, ManualLogin
from src.driver.driver_factory import ChromeDriverFactory
from src.models.dto.scraping_result import ScrapingResult, ScraperError, ErrorType
from src.utils.term_resolver import generate_academic_year_range
from src.utils.schedule_resolver import get_bidding_round_info_for_term


@dataclass(frozen=True)
class ClassScraperConfig:
    """
    Configuration for ClassScraper - ALL dependencies explicit.

    bidding_schedules is REQUIRED to determine folder naming.
    """
    bidding_schedules: dict  # REQUIRED - no default!
    min_class_number: int = 1000
    max_class_number: int = 5000
    consecutive_empty_threshold: int = 300
    base_url: str = "https://boss.intranet.smu.edu.sg"
    delay_between_requests: float = 1.0
    max_retries: int = 3


class ClassScraper(BaseScraper):
    """
    Scraper engine for BOSS class details.

    Handles scanning class numbers 1000-5000 and saving HTML files.
    Login is handled separately via Authenticator injection.

    Usage:
        from src.config import BIDDING_SCHEDULES
        config = ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES)
        scraper = ClassScraper(config=config)
        result = scraper.scrape(start_ay_term="2024-25_T1", end_ay_term="2024-25_T1")
    """

    TERM_CODE_MAP = {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}

    def __init__(
        self,
        config: ClassScraperConfig,
        driver: Optional[WebDriver] = None,
        logger=None,
    ):
        if config is None:
            raise ValueError(
                "config is required. Use: ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES)"
            )
        self._config = config
        super().__init__(
            driver=driver,
            config=self._config,
            logger=logger,
        )
        self._all_terms = ['T1', 'T2', 'T3A', 'T3B']
        self._current_term = None
        self._current_round_folder = None

    @property
    def config(self) -> ClassScraperConfig:
        """Access ClassScraper-specific configuration."""
        return self._config

    def scrape(
        self,
        start_ay_term: str = None,
        end_ay_term: str = None,
        base_dir: str = 'script_input/classTimingsFull',
        authenticator: Optional[Authenticator] = None,
        driver: Optional[WebDriver] = None,
    ) -> ScrapingResult:
        """
        Perform full scraping process for class details.

        Args:
            start_ay_term: Start academic year term (e.g., '2024-25_T1')
            end_ay_term: End academic year term
            base_dir: Output directory for HTML files
            authenticator: Optional Authenticator for login
            driver: Optional WebDriver (will use injected driver if not provided)

        Returns:
            ScrapingResult with operation outcome
        """
        # Use provided driver or existing
        target_driver = driver or self._driver
        if target_driver is None:
            raise RuntimeError("No WebDriver available. Call connect(driver) first or pass driver argument.")

        # Login if authenticator provided
        if authenticator:
            try:
                authenticator.login(target_driver)
            except Exception as e:
                result = ScrapingResult(
                    ay_term=start_ay_term or "unknown",
                    round_folder="unknown",
                )
                result.add_error(ScraperError.create(
                    url_attempted=self._config.base_url,
                    error_type=ErrorType.AUTHENTICATION,
                    message=f"Login failed: {e}"
                ))
                result.finalize()
                return result

        # Generate list of terms to scrape
        try:
            ay_terms_to_scrape = generate_academic_year_range(start_ay_term, end_ay_term)
        except ValueError as e:
            result = ScrapingResult(
                ay_term=start_ay_term or "unknown",
                round_folder="unknown",
            )
            self._logger.error(str(e))
            result.finalize()
            return result

        total_files_saved = 0
        now = datetime.now()

        for ay_term in ay_terms_to_scrape:
            self._logger.info(f"\nProcessing Academic Term: {ay_term}")

            round_folder = get_bidding_round_info_for_term(ay_term, now, self._config.bidding_schedules)
            if not round_folder:
                self._logger.info(f"Not in a bidding window for {ay_term} at this time. Skipping.")
                continue

            target_path = Path(base_dir) / ay_term / round_folder
            target_path.mkdir(parents=True, exist_ok=True)

            self._logger.info(f"Scraping to: {target_path}")

            # Perform the scanning for this term
            try:
                files_saved = self._scrape_range(
                    target_driver,
                    ay_term,
                    target_path,
                )
                total_files_saved += files_saved
            except Exception as e:
                self._logger.error(f"Scraping error for {ay_term}: {e}")

        # Generate CSV of scraped filepaths after all terms complete
        self.generate_scraped_filepaths_csv(base_dir)

        self._logger.info("\nScraping process completed.")

        result = ScrapingResult(
            ay_term=start_ay_term or "unknown",
            round_folder=round_folder or "unknown",
            files_saved=total_files_saved,
        )
        result.finalize()
        return result

    # ==================== Internal Methods ====================

    def _scrape_range(
        self,
        driver: WebDriver,
        ay_term: str,
        output_dir: Path,
    ) -> int:
        """
        Scrape all class numbers in configured range.

        Returns:
            int: Number of files saved
        """
        if not ay_term:
            ay_term = "2025-26_T1"

        ay, term = ay_term.split('_')
        ay_short = ay[2:4]  # "25" from "2025"
        term_code = self.TERM_CODE_MAP.get(term, '10')

        files_saved = 0
        consecutive_empty = 0

        self._logger.info(f"Performing full scan for {ay_term}.")
        for class_num in range(self._config.min_class_number, self._config.max_class_number + 1):
            try:
                saved = self._scrape_single_class(
                    driver,
                    output_dir,
                    ay_short,
                    term_code,
                    class_num,
                )

                if saved is None:  # Error
                    break
                elif not saved:  # Empty
                    consecutive_empty += 1
                    if consecutive_empty >= self._config.consecutive_empty_threshold:
                        self._logger.info(f"Stopping scan after {consecutive_empty} consecutive empty records.")
                        break
                else:
                    consecutive_empty = 0
                    files_saved += 1

            except Exception as e:
                self._logger.error(f"Error at class {class_num}: {e}")

        return files_saved

    def _scrape_single_class(
        self,
        driver: WebDriver,
        output_dir: Path,
        ay_short: str,
        term_code: str,
        class_num: int,
    ) -> Optional[bool]:
        """
        Scrape a single class number.

        Returns:
            True if saved, False if empty, None if error
        """
        filename = f"SelectedAcadTerm={ay_short}{term_code}&SelectedClassNumber={class_num:04}.html"
        filepath = output_dir / filename

        url = (
            f"{self._config.base_url}/ClassDetails.aspx"
            f"?SelectedClassNumber={class_num:04}"
            f"&SelectedAcadTerm={ay_short}{term_code}"
            f"&SelectedAcadCareer=UGRD"
        )

        try:
            driver.get(url)

            # Wait for content with smart wait (handles stale elements)
            try:
                self.wait_for_any_of(
                    EC.visibility_of_element_located((By.ID, "RadGrid_MeetingInfo_ctl00")),
                    EC.presence_of_element_located((By.ID, "lblErrorDetails")),
                    timeout=10,
                )
            except StaleElementError:
                self._logger.warning(f"Stale element at class {class_num}, retrying...")
                time.sleep(1)
                driver.get(url)
                self.wait_for_any_of(
                    EC.visibility_of_element_located((By.ID, "RadGrid_MeetingInfo_ctl00")),
                    EC.presence_of_element_located((By.ID, "lblErrorDetails")),
                    timeout=10,
                )

            page_source = driver.page_source

            if "No record found" in page_source:
                return False

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_source)

            time.sleep(self._config.delay_between_requests)
            return True

        except Exception as e:
            self._logger.error(f"Error processing {url}: {e}")
            time.sleep(5)
            return None

    def generate_scraped_filepaths_csv(
        self,
        base_dir: str = 'script_input/classTimingsFull',
        output_csv: str = 'script_input/scraped_filepaths.csv',
    ) -> None:
        """
        Generate CSV of all scraped HTML file paths.

        Args:
            base_dir: Directory to search for HTML files
            output_csv: Output CSV file path
        """
        base_path = Path(base_dir)
        existing_paths = set()

        if Path(output_csv).exists():
            with open(output_csv, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip header
                    for row in reader:
                        if row:
                            existing_paths.add(row[0])
                except (IOError, StopIteration):
                    pass

        new_paths = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith('.html'):
                    filepath = os.path.join(root, file)
                    if filepath not in existing_paths:
                        new_paths.append(filepath)

        if not new_paths:
            self._logger.info("No new valid HTML files found to add to the CSV.")
            return

        mode = 'a' if existing_paths else 'w'
        with open(output_csv, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if mode == 'w':
                writer.writerow(['Filepath'])
            for path in new_paths:
                writer.writerow([path])

        self._logger.info(f"CSV updated. Total valid files now: {len(existing_paths) + len(new_paths)}")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    from src.config import BIDDING_SCHEDULES, START_AY_TERM, END_AY_TERM
    from src.driver.driver_factory import ChromeDriverFactory
    from src.driver.authenticator import AutomatedLogin, AuthCredentials, ManualLogin
    from src.scraper.coordinator import ScraperCoordinator
    
    config = ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES)
    scraper = ClassScraper(config=config)
    factory = ChromeDriverFactory(headless=False)
    
    try:
        credentials = AuthCredentials.from_environment()
        authenticator = AutomatedLogin(credentials)
    except ValueError:
        authenticator = ManualLogin()

    coordinator = ScraperCoordinator(
        driver_factory=factory,
        authenticator=authenticator,
        scraper=scraper
    )

    try:
        coordinator.run(start_ay_term=START_AY_TERM, end_ay_term=END_AY_TERM)
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)