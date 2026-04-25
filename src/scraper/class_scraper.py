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
        result = scraper.scrape(acad_term_id="AY202526T3A")

    # With automated login
    credentials = AuthCredentials.from_environment()
    scraper = ClassScraper()
    with scraper:
        driver = factory.create()
        scraper.connect(driver)
        authenticator = AutomatedLogin(credentials)
        authenticator.login(driver)
        result = scraper.scrape(acad_term_id="AY202526T3A")
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import csv
import os
import time
from pathlib import Path

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from src.scraper.abstract_scraper import AbstractScraper, StaleElementError
from src.driver.authenticator import Authenticator
from src.scraper.dtos.scraping_result import ScrapingResult, ScraperError, ErrorType


@dataclass(frozen=True)
class ClassScraperConfig:
    """
    Configuration for ClassScraper - ALL dependencies explicit.

    bidding_schedules is REQUIRED to determine folder naming.
    start_ay_term is the academic term in dash format (e.g., '2025-26_T3A') from config.START_AY_TERM.
    """
    bidding_schedules: dict  # REQUIRED - no default!
    start_ay_term: str = ''  # Dash format from config.START_AY_TERM
    min_class_number: int = 1000
    max_class_number: int = 5000
    consecutive_empty_threshold: int = 300
    base_url: str = "https://boss.intranet.smu.edu.sg"
    delay_between_requests: float = 1.0
    max_retries: int = 3


class ClassScraper(AbstractScraper):
    """
    Scraper engine for BOSS class details.

    Handles scanning class numbers 1000-5000 and saving HTML files.
    Login is handled separately via Authenticator injection.

    Usage:
        from src.config import BIDDING_SCHEDULES, START_AY_TERM
        config = ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=START_AY_TERM)
        scraper = ClassScraper(config=config)
        result = scraper.scrape(acad_term_id="AY202526T3A")
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
                "config is required. Use: ClassScraperConfig(bidding_schedules=BIDDING_SCHEDULES, start_ay_term=START_AY_TERM)"
            )
        self._config = config
        super().__init__(
            driver=driver,
            config=self._config,
            logger=logger,
        )

    @property
    def config(self) -> ClassScraperConfig:
        """Access ClassScraper-specific configuration."""
        return self._config

    def scrape(
        self,
        acad_term_id: str = None,
        base_dir: str = 'script_input/classTimingsFull',
        authenticator: Optional[Authenticator] = None,
        driver: Optional[WebDriver] = None,
    ) -> ScrapingResult:
        """
        Perform full scraping process for class details.

        Args:
            acad_term_id: Academic term ID in BOSS format (e.g., 'AY202526T3A')
            base_dir: Output directory for HTML files
            authenticator: Optional Authenticator for login
            driver: Optional WebDriver (will use injected driver if not provided)

        Returns:
            ScrapingResult with operation outcome
        """
        target_driver = driver or self._driver
        if target_driver is None:
            raise RuntimeError("No WebDriver available. Call connect(driver) first or pass driver argument.")

        # Login if authenticator provided
        if authenticator:
            try:
                authenticator.login(target_driver)
            except Exception as e:
                result = ScrapingResult(
                    ay_term=acad_term_id or "unknown",
                    round_folder="unknown",
                )
                result.add_error(ScraperError.create(
                    url_attempted=self._config.base_url,
                    error_type=ErrorType.AUTHENTICATION,
                    message=f"Login failed: {e}"
                ))
                result.finalize()
                return result

        if not acad_term_id:
            result = ScrapingResult(
                ay_term="unknown",
                round_folder="unknown",
            )
            self._logger.error("acad_term_id is required")
            result.finalize()
            return result

        total_files_saved = 0
        now = datetime.now()

        ay_term = acad_term_id
        self._logger.info(f"\nProcessing Academic Term: {ay_term}")

        # Use start_ay_term from config (already in dash format from config.START_AY_TERM)
        schedule_key = self._config.start_ay_term

        # Determine round folder name - inlined get_bidding_round_info_for_term logic
        schedule = self._config.bidding_schedules.get(schedule_key)
        round_folder = None
        if schedule:
            for results_date, *rest in schedule:
                if now < results_date:
                    suffix = rest[1] if len(rest) > 1 else None
                    round_folder = f"{schedule_key}_{suffix}" if suffix else None
                    break


        if not round_folder:
            self._logger.info(f"Not in a bidding window for {ay_term} at this time. Skipping.")
            result = ScrapingResult(
                ay_term=acad_term_id or "unknown",
                round_folder="unknown",
                files_saved=0,
            )
            result.finalize()
            return result

        target_path = Path(base_dir) / ay_term / round_folder
        target_path.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Scraping to: {target_path}")

        files_saved = 0
        try:
            files_saved = self._scrape_range(
                target_driver,
                ay_term,
                target_path,
            )
            total_files_saved += files_saved
        except Exception as e:
            self._logger.error(f"Scraping error for {ay_term}: {e}")

        self.generate_scraped_filepaths_csv(base_dir)

        self._logger.info("\nScraping process completed.")

        result = ScrapingResult(
            ay_term=acad_term_id or "unknown",
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