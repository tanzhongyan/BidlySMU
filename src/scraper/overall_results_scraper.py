"""
Overall Results Scraper - scrapes bidding results from BOSS.

Usage:
    scraper = OverallResultsScraper()
    with scraper:
        authenticator = AutomatedLogin(credentials)
        scraper.connect(driver)
        authenticator.login(driver)
        result = scraper.run(term="2025-26_T1")
"""
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

from src.scraper.abstract_scraper import AbstractScraper
from src.driver.authenticator import Authenticator
from src.logging.logger import get_logger
from src.scraper.dtos.scraping_result import ScrapingResult, ScraperError, ErrorType


@dataclass
class OverallResultsConfig:
    """Configuration for OverallResultsScraper."""
    bidding_schedules: dict  # REQUIRED
    start_ay_term: str  # REQUIRED
    base_url: str = "https://boss.intranet.smu.edu.sg/OverallResults.aspx"
    delay: int = 5
    headless: bool = False
    page_size: int = 50
    max_retries: int = 3


class OverallResultsScraper(AbstractScraper):
    """
    Scraper for BOSS Overall Results page.

    Handles form interaction, pagination, and data extraction.
    Login is handled separately via Authenticator injection.
    """

    DESIRED_COLUMNS = [
        'Term', 'Session', 'Bidding Window', 'Course Code', 'Description',
        'Section', 'Vacancy', 'Opening Vacancy', 'Before Process Vacancy',
        'D.I.C.E', 'After Process Vacancy', 'Enrolled Students',
        'Median Bid', 'Min Bid', 'Instructor', 'School/Department'
    ]

    _TERM_DISPLAY_MAP = {
        'T1': 'Term 1',
        'T2': 'Term 2',
        'T3A': 'Term 3A',
        'T3B': 'Term 3B'
    }

    @staticmethod
    def _transform_term_format(short_term: str) -> str:
        """Convert '2025-26_T1' -> '2025-26 Term 1' for BOSS website dropdown."""
        year_part, term_part = short_term.split('_')
        full_term_name = OverallResultsScraper._TERM_DISPLAY_MAP.get(term_part, term_part)
        return f"{year_part} {full_term_name}"

    def __init__(
        self,
        config: OverallResultsConfig,
        driver=None,
        logger=None,
    ):
        if config is None:
            raise ValueError("config is required.")
        self._config = config
        super().__init__(driver=driver, config=self._config, logger=logger)

    # ==================== AbstractScraper Implementation ====================

    def scrape(
        self,
        term: str,
        bid_round: Optional[str] = None,
        bid_window: Optional[str] = None,
        output_dir: str = "./script_input/overallBossResults",
        authenticator: Optional[Authenticator] = None,
    ) -> ScrapingResult:
        """
        Run the scraper for a single term.

        Args:
            term: Term to scrape (e.g., '2025-26_T1')
            bid_round: Specific bid round to filter by
            bid_window: Specific bid window to filter by
            output_dir: Directory to save output files
            authenticator: Optional Authenticator for login

        Returns:
            ScrapingResult with operation outcome
        """

        result = ScrapingResult(
            ay_term=term or "unknown",
            round_folder=bid_round or "unknown",
        )

        try:
            website_term = self._transform_term_format(term)

            # Auto-detect phase if not specified
            if bid_round is None or bid_window is None:
                detected_round, detected_window = self._determine_current_bidding_phase()
                if detected_round:
                    bid_round = bid_round or detected_round
                    bid_window = bid_window or detected_window

            self._logger.info(f"Scraping {website_term} - Round {bid_round}, Window {bid_window}")

            # Navigate and login
            self._driver.get("https://boss.intranet.smu.edu.sg/")

            if authenticator:
                authenticator.login(self._driver)

            # Scrape data
            data = self._scrape_term_data(
                term=website_term,
                bid_round=bid_round,
                bid_window=bid_window,
                output_dir=output_dir,
            )

            result.files_saved = len(data)
            self._logger.info(f"Scraping completed! Collected {len(data)} records")
            result.finalize()
            return result

        except Exception as e:
            self._logger.error(f"Error during scraping: {e}")
            result.add_error(ScraperError.create(
                url_attempted=self._config.base_url,
                error_type=ErrorType.UNKNOWN,
                message=str(e)
            ))
            result.finalize()
            return result

    def _determine_current_bidding_phase(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine the current bidding phase based on current time.

        Returns:
            tuple: (round, window) or (None, None) if no active phase
        """
        current_time = datetime.now()
        self._logger.info(f"Current time: {current_time}")

        if self._config.start_ay_term not in self._config.bidding_schedules:
            return None, None

        schedule = self._config.bidding_schedules[self._config.start_ay_term]

        active_phase = None
        for schedule_time, phase_name, _ in schedule:
            if current_time >= schedule_time:
                active_phase = (schedule_time, phase_name)
            else:
                break

        if active_phase is None:
            self._logger.warning("No active bidding phase found")
            return None, None

        schedule_time, phase_name = active_phase
        self._logger.info(f"Current bidding phase: {phase_name}")

        # Parse phase name to extract round and window
        normalized = phase_name.replace("Incoming Exchange ", "").replace("Incoming Freshmen ", "")
        normalized = normalized.replace("Rnd ", "Round ").replace("Win ", "Window ")

        match = re.search(r'Round\s+(\d+[A-Z]*)\s+Window\s+(\d+)', normalized)
        if match:
            return match.group(1), match.group(2)

        return None, None

    def _scrape_term_data(
        self,
        term: str,
        bid_round: Optional[str],
        bid_window: Optional[str],
        output_dir: str,
    ) -> List[dict]:
        """
        Scrape data for a specific term with pagination.

        Returns:
            List of data records
        """
        self._navigate_to_overall_results()
        self._select_course_career("Undergraduate")
        self._select_term(term)
        self._select_bid_round(bid_round)
        self._select_bid_window(bid_window)
        self._click_search()
        self._set_page_size_to_50()
        self._sort_by_bidding_window()

        # Collect all data from all pages
        all_data = []
        page_num = 1
        max_pages = 200
        last_bidding_window = None

        # Get initial page information
        current_page, total_pages, total_items = self._get_current_page_info()
        if total_pages:
            self._logger.info(f"Starting scrape: {total_items} total items across {total_pages} pages")

        while page_num <= max_pages:
            # Get current page info
            current_page, total_pages, _ = self._get_current_page_info()

            if current_page and total_pages:
                self._logger.info(f"Scraping page {current_page} of {total_pages} (iteration {page_num})...")

            page_data, should_stop, current_bidding_window = self._extract_table_data(
                stop_on_bidding_window_change=True,
                last_bidding_window=last_bidding_window,
            )

            if current_bidding_window:
                last_bidding_window = current_bidding_window

            if should_stop:
                self._logger.info("Early termination due to bidding window change")
                break

            if page_data:
                all_data.extend(page_data)
                self._logger.info(f"Page {page_num}: Found {len(page_data)} records")
            else:
                self._logger.warning(f"Page {page_num}: No data found")
                break

            # Check if we've reached the last page
            if current_page and total_pages and current_page >= total_pages:
                self._logger.info(f"Reached last page ({current_page}/{total_pages})")
                break

            if not self._has_next_page():
                break

            if not self._click_next_page():
                break

            page_num += 1
            time.sleep(self._config.delay)

        # Save to Excel
        if all_data:
            self._save_to_excel(all_data, term, output_dir)

        return all_data

    # ==================== Navigation Methods ====================

    def _navigate_to_overall_results(self) -> None:
        """Navigate to the Overall Results page."""
        self._driver.get(self._config.base_url)
        wait = WebDriverWait(self._driver, 30)
        wait.until(EC.presence_of_element_located((By.ID, "rcboCourseCareer")))
        self._logger.info("Successfully navigated to Overall Results page")
        time.sleep(2)

    def _select_course_career(self, career: str = "Undergraduate") -> None:
        """Select course career."""
        career_input = self._driver.find_element(By.ID, "rcboCourseCareer_Input")
        current_value = career_input.get_attribute("value")

        if current_value != career:
            dropdown_arrow = self._driver.find_element(By.ID, "rcboCourseCareer_Arrow")
            dropdown_arrow.click()
            time.sleep(1)

            option = self._driver.find_element(By.XPATH, f"//li[@class='rcbItem' and text()='{career}']")
            option.click()
            time.sleep(1)

        self._logger.info(f"Course career set to: {career}")

    def _select_term(self, term: str) -> None:
        """Select term from dropdown."""
        wait = WebDriverWait(self._driver, 10)

        current_term_input = self._driver.find_element(By.ID, "rcboTerm_Input")
        current_term_value = current_term_input.get_attribute("value").strip()

        if current_term_value == term:
            self._logger.info(f"Term '{term}' is already selected")
            return

        self._logger.info(f"Changing term from '{current_term_value}' to '{term}'")

        term_arrow = wait.until(EC.element_to_be_clickable((By.ID, "rcboTerm_Arrow")))
        self._driver.execute_script("arguments[0].click();", term_arrow)

        dropdown_div = wait.until(EC.visibility_of_element_located((By.ID, "rcboTerm_DropDown")))

        selected_checkboxes = dropdown_div.find_elements(By.XPATH, ".//input[@type='checkbox' and @checked='checked']")
        for checkbox in selected_checkboxes:
            self._driver.execute_script("arguments[0].checked = false;", checkbox)

        term_checkbox_xpath = f"//div[@id='rcboTerm_DropDown']//label[contains(., '{term}')]/input[@type='checkbox']"
        term_checkbox = wait.until(EC.presence_of_element_located((By.XPATH, term_checkbox_xpath)))
        self._driver.execute_script("arguments[0].click();", term_checkbox)

        self._driver.find_element(By.TAG_NAME, "body").click()
        time.sleep(1)
        self._logger.info(f"Term selected: {term}")

    def _select_bid_round(self, round_value: Optional[str] = None) -> None:
        """Select bid round."""
        if round_value is None:
            self._logger.info("Bid round left as default")
            return

        # Map Freshmen Round 1 to Round 1
        if round_value == '1F':
            round_value = '1'

        round_arrow = self._driver.find_element(By.ID, "rcboBidRound_Arrow")
        round_arrow.click()
        time.sleep(1)

        round_option = self._driver.find_element(
            By.XPATH,
            f"//div[@id='rcboBidRound_DropDown']//li[@class='rcbItem' and text()='{round_value}']"
        )
        round_option.click()
        time.sleep(1)
        self._logger.info(f"Bid round selected: {round_value}")

    def _select_bid_window(self, window_value: Optional[str] = None) -> None:
        """Select bid window."""
        if window_value is None:
            self._logger.info("Bid window left as default")
            return

        window_arrow = self._driver.find_element(By.ID, "rcboBidWindow_Arrow")
        window_arrow.click()
        time.sleep(1)

        window_option = self._driver.find_element(
            By.XPATH,
            f"//div[@id='rcboBidWindow_DropDown']//li[@class='rcbItem' and text()='{window_value}']"
        )
        window_option.click()
        time.sleep(1)
        self._logger.info(f"Bid window selected: {window_value}")

    def _click_search(self) -> None:
        """Click the search button."""
        search_button = self._driver.find_element(By.ID, "RadButton_Search_input")
        search_button.click()

        wait = WebDriverWait(self._driver, 30)
        wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
        self._logger.info("Search completed successfully")
        time.sleep(3)

    def _set_page_size_to_50(self) -> None:
        """Set page size to 50 records per page."""
        try:
            page_size_arrow = self._driver.find_element(
                By.ID, "RadGrid_OverallResults_ctl00_ctl03_ctl01_PageSizeComboBox_Arrow"
            )
            page_size_arrow.click()
            time.sleep(1)

            option_50 = self._driver.find_element(
                By.XPATH,
                "//div[@id='RadGrid_OverallResults_ctl00_ctl03_ctl01_PageSizeComboBox_DropDown']//li[text()='50']"
            )
            option_50.click()
            time.sleep(5)

            self._logger.info("Page size set to 50 records per page")
        except Exception as e:
            self._logger.warning(f"Failed to set page size to 50: {e}")

    def _sort_by_bidding_window(self) -> None:
        """Sort results by bidding window."""
        try:
            sort_link = self._driver.find_element(
                By.XPATH,
                "//a[contains(@onclick, 'EVENT_TYPE_DESCRIPTION') and contains(text(), 'Bidding Window')]"
            )
            sort_link.click()
            time.sleep(3)

            wait = WebDriverWait(self._driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
            self._logger.info("Sorted by Bidding Window")
        except Exception as e:
            self._logger.warning(f"Failed to sort by bidding window: {e}")

    # ==================== Extraction Methods ====================

    def _extract_table_data(
        self,
        stop_on_bidding_window_change: bool = False,
        last_bidding_window: Optional[str] = None,
    ) -> Tuple[List[dict], bool, Optional[str]]:
        """
        Extract data from the current page table.

        Returns:
            tuple: (page_data, should_stop, current_bidding_window)
        """
        wait = WebDriverWait(self._driver, 15)
        wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
        time.sleep(2)

        table = self._driver.find_element(By.ID, "RadGrid_OverallResults_ctl00")
        all_rows = table.find_elements(By.TAG_NAME, "tr")

        data_rows = []
        for row in all_rows:
            try:
                row_class = row.get_attribute("class") or ""
                if "rgRow" in row_class or "rgAltRow" in row_class:
                    data_rows.append(row)
            except StaleElementReferenceException:
                continue

        if not data_rows:
            self._logger.warning("No data rows found")
            return [], False, None

        page_data = []
        current_bidding_window = None
        should_stop = False

        for i, row in enumerate(data_rows):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")

                if len(cells) < 16:
                    continue

                row_bidding_window = cells[2].text.strip()
                if current_bidding_window is None:
                    current_bidding_window = row_bidding_window

                if stop_on_bidding_window_change and last_bidding_window:
                    if (last_bidding_window.startswith("Incoming Freshmen") and
                        not row_bidding_window.startswith("Incoming Freshmen")):
                        self._logger.info(f"Bidding window changed from '{last_bidding_window}' to '{row_bidding_window}'")
                        should_stop = True
                        break

                # Extract section from cell 5
                section_cell = cells[5]
                section_text = ""
                try:
                    link = section_cell.find_element(By.TAG_NAME, "a")
                    section_text = link.get_attribute("title") or link.text.strip()
                except NoSuchElementException:
                    section_text = section_cell.text.strip()

                # Build record
                record = {
                    'Term': cells[0].text.strip(),
                    'Session': cells[1].text.strip(),
                    'Bidding Window': cells[2].text.strip(),
                    'Course Code': cells[3].text.strip(),
                    'Description': cells[4].text.strip(),
                    'Section': section_text,
                    'Median Bid': cells[6].text.strip(),
                    'Min Bid': cells[7].text.strip(),
                    'Vacancy': cells[8].text.strip(),
                    'Opening Vacancy': cells[9].text.strip(),
                    'Before Process Vacancy': cells[10].text.strip(),
                    'After Process Vacancy': cells[11].text.strip(),
                    'D.I.C.E': cells[12].text.strip(),
                    'Enrolled Students': cells[13].text.strip(),
                    'Instructor': cells[14].text.strip(),
                    'School/Department': cells[15].text.strip()
                }

                # Clean record
                cleaned_record = self._clean_record(record)

                if cleaned_record['Course Code'] and cleaned_record['Course Code'] != '-':
                    page_data.append(cleaned_record)

            except Exception as e:
                self._logger.warning(f"Error processing row {i}: {e}")
                continue

        self._logger.info(f"Extracted {len(page_data)} valid records")
        return page_data, should_stop, current_bidding_window

    def _clean_record(self, record: dict) -> dict:
        """Clean record values."""
        cleaned = {}
        for key, value in record.items():
            cleaned_value = re.sub(r'\s+', ' ', str(value)).strip()
            cleaned_value = cleaned_value.replace('\u00a0', ' ').replace('&nbsp;', ' ')

            if cleaned_value == '' or cleaned_value == ' ':
                cleaned_value = '-'

            if key in ['Median Bid', 'Min Bid'] and cleaned_value == '-':
                cleaned_value = '0'

            cleaned[key] = cleaned_value
        return cleaned

    # ==================== Pagination Methods ====================

    def _has_next_page(self) -> bool:
        """Check if there is a next page available."""
        try:
            next_buttons = self._driver.find_elements(
                By.XPATH, "//input[@title='Next Page' and contains(@name, 'RadGrid_OverallResults')]"
            )

            if next_buttons:
                next_button = next_buttons[0]
                is_enabled = next_button.is_enabled()
                button_class = next_button.get_attribute("class") or ""
                return is_enabled and "disabled" not in button_class.lower()

            next_buttons_by_class = self._driver.find_elements(By.CLASS_NAME, "rgPageNext")
            if next_buttons_by_class:
                next_button = next_buttons_by_class[0]
                is_enabled = next_button.is_enabled()
                button_class = next_button.get_attribute("class") or ""
                return is_enabled and "disabled" not in button_class.lower()

            return False

        except Exception as e:
            self._logger.error(f"Error checking for next page: {e}")
            return False

    def _click_next_page(self) -> bool:
        """Click the next page button."""
        try:
            next_buttons = self._driver.find_elements(
                By.XPATH, "//input[@title='Next Page' and contains(@name, 'RadGrid_OverallResults')]"
            )

            if not next_buttons:
                next_buttons = self._driver.find_elements(By.CLASS_NAME, "rgPageNext")

            if not next_buttons:
                return False

            next_button = next_buttons[0]
            if not next_button.is_enabled():
                return False

            next_button.click()
            time.sleep(self._config.delay)

            wait = WebDriverWait(self._driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
            time.sleep(2)

            self._logger.info("Navigated to next page")
            return True

        except Exception as e:
            self._logger.error(f"Failed to click next page: {e}")
            return False

    def _get_current_page_info(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get current page number and total pages.

        Returns:
            tuple: (current_page, total_pages, total_items)
        """
        try:
            info_elements = self._driver.find_elements(By.CLASS_NAME, "rgInfoPart")
            if info_elements:
                info_text = info_elements[0].text
                match = re.search(r'(\d+)\s+items\s+in\s+(\d+)\s+pages', info_text)
                if match:
                    total_items = int(match.group(1))
                    total_pages = int(match.group(2))

                    current_page_elements = self._driver.find_elements(By.CLASS_NAME, "rgCurrentPage")
                    if current_page_elements:
                        current_page_text = current_page_elements[0].text.strip()
                        if current_page_text.isdigit():
                            return int(current_page_text), total_pages, total_items

            return None, None, None

        except Exception:
            return None, None, None

    # ==================== Output Methods ====================

    def _generate_filename(self, term: str) -> str:
        """Generate filename based on term."""
        filename = term
        for full_term, short_term in self.TERM_MAP.items():
            if full_term in term:
                filename = term.replace(full_term, short_term)
                break
        return filename + '.xlsx'

    def _save_to_excel(self, data: List[dict], term: str, output_dir: str) -> None:
        """Save data to Excel file."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            new_df = pd.DataFrame(data)
            new_df = new_df[self.DESIRED_COLUMNS]

            filename = self._generate_filename(term)
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                existing_df = pd.read_excel(filepath, engine='openpyxl')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates().reset_index(drop=True)
                self._logger.info(f"Combined {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total")
            else:
                combined_df = new_df

            combined_df.to_excel(filepath, index=False, engine='openpyxl')
            self._logger.info(f"Data saved to {filepath}")

        except Exception as e:
            self._logger.error(f"Failed to save data to Excel: {e}")
            raise
