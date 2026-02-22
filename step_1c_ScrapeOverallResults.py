# Import global configuration settings
from config import *

# Import shared utilities
from util import (
    setup_driver,
    wait_for_manual_login,
    perform_automated_login,
    transform_term_format,
    get_bidding_round_info_for_term,
    setup_logger
)

# Import dependencies
import os
import re
import sys
import time
import pandas as pd
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException


class ScrapeOverallResults:
    """
    BOSS Overall Results Scraper using Selenium
    
    This class scrapes course bidding results from the BOSS system with proper
    authentication, form interaction, and data extraction capabilities.
    """
    
    def __init__(self, headless=False, delay=5):
        """
        Initialize the scraper with configuration parameters.
        
        Args:
            headless (bool): Run browser in headless mode
            delay (int): Delay between requests in seconds
        """
        self.driver = None
        self.delay = delay
        self.headless = headless
        self.base_url = "https://boss.intranet.smu.edu.sg/OverallResults.aspx"
        
        # Column mapping and ordering as specified
        self.desired_columns = [
            'Term', 'Session', 'Bidding Window', 'Course Code', 'Description',
            'Section', 'Vacancy', 'Opening Vacancy', 'Before Process Vacancy',
            'D.I.C.E', 'After Process Vacancy', 'Enrolled Students',
            'Median Bid', 'Min Bid', 'Instructor', 'School/Department'
        ]
        
        # Use the global bidding schedule, extracting only the date and name
        # Assuming we are targeting the start term for this scraper.
        self.boss_schedule = []
        if START_AY_TERM in BIDDING_SCHEDULES:
            schedule_for_term = BIDDING_SCHEDULES[START_AY_TERM]
            # Keep only the datetime and full name for this class's logic
            self.boss_schedule = [(dt, name) for dt, name, suffix in schedule_for_term]
        
        # Setup logging
        self.logger = setup_logger(__name__)
        
    def _determine_current_bidding_phase(self):
        """
        Determine the current bidding phase based on current time
        
        Returns:
            tuple: (round, window) or (None, None) if no active phase
        """
        current_time = datetime.now()
        self.logger.info(f"Current time: {current_time}")
        
        # Find the most recent bidding phase that has started
        active_phase = None
        for schedule_time, phase_name in self.boss_schedule:
            if current_time >= schedule_time:
                active_phase = (schedule_time, phase_name)
            else:
                break
        
        if active_phase is None:
            self.logger.warning("No active bidding phase found - before first scheduled phase")
            return None, None
        
        schedule_time, phase_name = active_phase
        self.logger.info(f"Current bidding phase: {phase_name} (started at {schedule_time})")
        
        # Parse the phase name to extract round and window
        # Handle different formats:
        # "Round 1 Window 1" -> ("1", "1")
        # "Round 1A Window 2" -> ("1A", "2")
        # "Round 2A Window 3" -> ("2A", "3")
        # "Incoming Exchange Rnd 1C Win 1" -> ("1C", "1")
        # "Incoming Freshmen Rnd 1 Win 1" -> ("1", "1")
        
        try:
            # Remove prefixes and normalize
            normalized = phase_name.replace("Incoming Exchange ", "").replace("Incoming Freshmen ", "")
            normalized = normalized.replace("Rnd ", "Round ").replace("Win ", "Window ")
            
            # Extract round and window using regex
            match = re.search(r'Round\s+(\d+[A-Z]*)\s+Window\s+(\d+)', normalized)
            
            if match:
                round_value = match.group(1)
                window_value = match.group(2)
                
                self.logger.info(f"Parsed phase: Round {round_value}, Window {window_value}")
                return round_value, window_value
            else:
                self.logger.warning(f"Could not parse phase name: {phase_name}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error parsing phase name '{phase_name}': {str(e)}")
            return None, None
    
    def _setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        try:
            self.driver = setup_driver(headless=self.headless)
            self.logger.info("Chrome WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise
    
    def _navigate_to_overall_results(self):
        """Navigate to the Overall Results page"""
        try:
            self.driver.get(self.base_url)
            
            # Wait for page to load
            wait = WebDriverWait(self.driver, 30)
            wait.until(EC.presence_of_element_located((By.ID, "rcboCourseCareer")))
            
            self.logger.info("Successfully navigated to Overall Results page")
            time.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Failed to navigate to Overall Results page: {str(e)}")
            raise
    
    def _select_course_career(self, career="Undergraduate"):
        """Select course career (default: Undergraduate)"""
        try:
            # The dropdown is already set to Undergraduate by default
            career_input = self.driver.find_element(By.ID, "rcboCourseCareer_Input")
            current_value = career_input.get_attribute("value")
            
            if current_value != career:
                # If we need to change it, click the dropdown arrow
                dropdown_arrow = self.driver.find_element(By.ID, "rcboCourseCareer_Arrow")
                dropdown_arrow.click()
                time.sleep(1)
                
                # Select the desired option
                option = self.driver.find_element(By.XPATH, f"//li[@class='rcbItem' and text()='{career}']")
                option.click()
                time.sleep(1)
            
            self.logger.info(f"Course career set to: {career}")
            
        except Exception as e:
            self.logger.error(f"Failed to select course career: {str(e)}")
            raise
    
    def _select_term(self, term):
        """
        Selects a term ONLY if it's not already the selected term.
        
        Args:
            term (str): The full-text term to select (e.g., '2025-26 Term 1').
        """
        try:
            wait = WebDriverWait(self.driver, 10)
            
            # 1. First, check the currently displayed value in the term input box.
            current_term_input = self.driver.find_element(By.ID, "rcboTerm_Input")
            current_term_value = current_term_input.get_attribute("value").strip()
            
            # 2. Compare with the desired term. If they match, do nothing.
            if current_term_value == term:
                self.logger.info(f"Term '{term}' is already selected. Skipping interaction.")
                return

            # 3. If the term needs to be changed, proceed with the selection logic.
            self.logger.info(f"Current term is '{current_term_value}', changing to '{term}'.")
            term_arrow = wait.until(EC.element_to_be_clickable((By.ID, "rcboTerm_Arrow")))
            self.driver.execute_script("arguments[0].click();", term_arrow)
            
            dropdown_div = wait.until(EC.visibility_of_element_located((By.ID, "rcboTerm_DropDown")))
            
            selected_checkboxes = dropdown_div.find_elements(By.XPATH, ".//input[@type='checkbox' and @checked='checked']")
            for checkbox in selected_checkboxes:
                self.driver.execute_script("arguments[0].checked = false;", checkbox)
            
            term_checkbox_xpath = f"//div[@id='rcboTerm_DropDown']//label[contains(., '{term}')]/input[@type='checkbox']"
            term_checkbox = wait.until(EC.presence_of_element_located((By.XPATH, term_checkbox_xpath)))

            self.driver.execute_script("arguments[0].click();", term_checkbox)
            
            self.driver.find_element(By.TAG_NAME, "body").click()
            time.sleep(1)
            
            self.logger.info(f"Term selected: {term}")
            
        except (NoSuchElementException, TimeoutException) as e:
            self.logger.error(f"Failed to select term '{term}'. The element could not be found or timed out.")
            # Make sure to import these exceptions at the top of your script:
            # from selenium.common.exceptions import NoSuchElementException, TimeoutException
            with open("error_page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            self.logger.info("Page HTML at the time of error saved to 'error_page_source.html'.")
            raise e
    
    def _select_bid_round(self, round_value=None):
        """
        Select bid round
        
        Args:
            round_value (str): Round to select (e.g., '1', '1A', '2', '1F')
                            If None, leave as default (empty)
                            Note: '1F' is automatically mapped to '1' for dropdown selection
        """
        try:
            if round_value is None:
                self.logger.info("Bid round left as default (empty)")
                return
            
            # Map round values that don't exist in dropdown to valid options
            original_round_value = round_value
            if round_value == '1F':  # Freshmen Round 1 maps to Round 1
                round_value = '1'
                self.logger.info(f"Mapped round '{original_round_value}' to '{round_value}' for dropdown selection")
            
            # Click the round dropdown arrow
            round_arrow = self.driver.find_element(By.ID, "rcboBidRound_Arrow")
            round_arrow.click()
            time.sleep(1)
            
            # Select the round option
            round_option = self.driver.find_element(
                By.XPATH, 
                f"//div[@id='rcboBidRound_DropDown']//li[@class='rcbItem' and text()='{round_value}']"
            )
            round_option.click()
            time.sleep(1)
            
            self.logger.info(f"Bid round selected: {round_value} (original: {original_round_value})")
            
        except Exception as e:
            self.logger.error(f"Failed to select bid round '{original_round_value}' (mapped to '{round_value}'): {str(e)}")
            raise

    def _select_bid_window(self, window_value=None):
        """
        Select bid window
        
        Args:
            window_value (str): Window to select (e.g., '1', '2', '3')
                               If None, leave as default (empty)
        """
        try:
            if window_value is None:
                self.logger.info("Bid window left as default (empty)")
                return
            
            # Click the window dropdown arrow
            window_arrow = self.driver.find_element(By.ID, "rcboBidWindow_Arrow")
            window_arrow.click()
            time.sleep(1)
            
            # Select the window option
            window_option = self.driver.find_element(
                By.XPATH, 
                f"//div[@id='rcboBidWindow_DropDown']//li[@class='rcbItem' and text()='{window_value}']"
            )
            window_option.click()
            time.sleep(1)
            
            self.logger.info(f"Bid window selected: {window_value}")
            
        except Exception as e:
            self.logger.error(f"Failed to select bid window '{window_value}': {str(e)}")
            raise
    
    def _click_search(self):
        """Click the search button to submit the form"""
        try:
            search_button = self.driver.find_element(By.ID, "RadButton_Search_input")
            search_button.click()
            
            # Wait for results to load
            wait = WebDriverWait(self.driver, 30)
            wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
            
            self.logger.info("Search completed successfully")
            time.sleep(3)  # Give extra time for data to load
            
        except Exception as e:
            self.logger.error(f"Failed to click search or load results: {str(e)}")
            raise
    
    def _set_page_size_to_50(self):
        """Set the page size dropdown to 50 records per page"""
        try:
            # Click the page size dropdown arrow
            page_size_arrow = self.driver.find_element(
                By.ID, "RadGrid_OverallResults_ctl00_ctl03_ctl01_PageSizeComboBox_Arrow"
            )
            page_size_arrow.click()
            time.sleep(1)
            
            # Select 50 from the dropdown
            option_50 = self.driver.find_element(
                By.XPATH, 
                "//div[@id='RadGrid_OverallResults_ctl00_ctl03_ctl01_PageSizeComboBox_DropDown']//li[text()='50']"
            )
            option_50.click()
            
            # Wait for page to reload with new page size
            time.sleep(5)  # Extended wait for page reload
            
            self.logger.info("Page size set to 50 records per page")
            
        except Exception as e:
            self.logger.error(f"Failed to set page size to 50: {str(e)}")
            # Continue anyway, might work with default page size

    def _sort_by_bidding_window(self):
        """Sort the results by bidding window to get Incoming Freshmen first"""
        try:
            # Click the Bidding Window header to sort
            sort_link = self.driver.find_element(
                By.XPATH, 
                "//a[contains(@onclick, 'EVENT_TYPE_DESCRIPTION') and contains(text(), 'Bidding Window')]"
            )
            sort_link.click()
            
            # Wait for table to reload after sorting
            time.sleep(3)
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
            
            self.logger.info("Successfully sorted by Bidding Window")
            
        except Exception as e:
            self.logger.error(f"Failed to sort by bidding window: {str(e)}")
            # Continue without sorting rather than failing completely

    def _extract_table_data(self, stop_on_bidding_window_change=False, last_bidding_window=None):
        """
        Extract data from the current page table with improved robustness
        
        Args:
            stop_on_bidding_window_change (bool): Whether to stop when bidding window changes
            last_bidding_window (str): The last bidding window seen (for change detection)
        
        Returns:
            tuple: (page_data, should_stop, current_bidding_window)
        """
        try:
            # Wait for table to be fully loaded
            wait = WebDriverWait(self.driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
            
            # Additional wait for data to populate
            time.sleep(2)
            
            # Find the main table
            table = self.driver.find_element(By.ID, "RadGrid_OverallResults_ctl00")
            
            # Get all rows in the table
            all_rows = table.find_elements(By.TAG_NAME, "tr")
            self.logger.info(f"Found {len(all_rows)} total rows in table")
            
            # Filter for data rows only (rgRow and rgAltRow classes)
            data_rows = []
            for row in all_rows:
                try:
                    row_class = row.get_attribute("class") or ""
                    if "rgRow" in row_class or "rgAltRow" in row_class:
                        data_rows.append(row)
                except StaleElementReferenceException:
                    continue
            
            self.logger.info(f"Found {len(data_rows)} data rows")
            
            if len(data_rows) == 0:
                self.logger.warning("No data rows found - checking table structure")
                self._debug_table_content()
                return []
                        
            page_data = []
            current_bidding_window = None
            should_stop = False

            for i, row in enumerate(data_rows):
                try:
                    # Get all cells in the row
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) < 16:
                        self.logger.warning(f"Row {i} has only {len(cells)} cells, expected 16")
                        continue
                    
                    # Extract current row's bidding window for change detection
                    row_bidding_window = cells[2].text.strip()
                    if current_bidding_window is None:
                        current_bidding_window = row_bidding_window
                    
                    # Check for bidding window change if requested
                    if stop_on_bidding_window_change and last_bidding_window:
                        if (last_bidding_window.startswith("Incoming Freshmen") and 
                            not row_bidding_window.startswith("Incoming Freshmen")):
                            self.logger.info(f"Bidding window changed from '{last_bidding_window}' to '{row_bidding_window}' - stopping extraction")
                            should_stop = True
                            break
                    
                    # Extract section text from cell 5 (which contains a link)
                    section_cell = cells[5]
                    section_text = ""
                    try:
                        # Try to get link text first
                        link = section_cell.find_element(By.TAG_NAME, "a")
                        section_text = link.get_attribute("title") or link.text.strip()
                    except NoSuchElementException:
                        # If no link, get cell text directly
                        section_text = section_cell.text.strip()
                    
                    # Create record with proper column mapping
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
                    
                    # Clean up data
                    cleaned_record = {}
                    for key, value in record.items():
                        # Remove extra whitespace and handle special characters
                        cleaned_value = re.sub(r'\s+', ' ', str(value)).strip()
                        cleaned_value = cleaned_value.replace('\u00a0', ' ')  # Remove &nbsp;
                        cleaned_value = cleaned_value.replace('&nbsp;', ' ')
                        
                        # Handle empty values
                        if cleaned_value == '' or cleaned_value == ' ':
                            cleaned_value = '-'
                        
                        # Convert '-' to '0' for Median Bid and Min Bid fields
                        if key in ['Median Bid', 'Min Bid'] and cleaned_value == '-':
                            cleaned_value = '0'
                        
                        cleaned_record[key] = cleaned_value
                    
                    # Only add record if it has a valid course code
                    if cleaned_record['Course Code'] and cleaned_record['Course Code'] != '-':
                        page_data.append(cleaned_record)
                        
                        # Log first record for verification
                        if len(page_data) == 1:
                            self.logger.info(f"Sample record: {cleaned_record['Course Code']} - {cleaned_record['Description'][:30]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {i}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully extracted {len(page_data)} valid records")
            return page_data, should_stop, current_bidding_window
            
        except Exception as e:
            self.logger.error(f"Failed to extract table data: {str(e)}")
            self._debug_table_content()
            return []
    
    def _debug_table_content(self):
        """Debug method to inspect table content when extraction fails"""
        try:
            self.logger.info("=== TABLE DEBUG INFO ===")
            
            # Check if main table exists
            try:
                table = self.driver.find_element(By.ID, "RadGrid_OverallResults_ctl00")
                self.logger.info("✓ Main table found")
            except NoSuchElementException:
                self.logger.error("✗ Main table NOT found")
                return
            
            # Check all rows
            all_rows = table.find_elements(By.TAG_NAME, "tr")
            self.logger.info(f"Total rows in table: {len(all_rows)}")
            
            # Analyze first few rows
            for i, row in enumerate(all_rows[:5]):
                try:
                    row_class = row.get_attribute("class") or "no-class"
                    cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                    cell_count = len(cells)
                    
                    first_cell_text = ""
                    if cells:
                        first_cell_text = cells[0].text.strip()[:50]
                    
                    is_data_row = "rgRow" in row_class or "rgAltRow" in row_class
                    
                    self.logger.info(f"Row {i}: class='{row_class}', cells={cell_count}, data_row={is_data_row}")
                    self.logger.info(f"  First cell: '{first_cell_text}'")
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing row {i}: {str(e)}")
            
            # Check for error messages in the page
            error_elements = self.driver.find_elements(By.CLASS_NAME, "error")
            if error_elements:
                for error in error_elements:
                    if error.text.strip():
                        self.logger.warning(f"Error message found: {error.text}")
            
            # Check if there's a "no data" message
            no_data_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'No record') or contains(text(), 'no data') or contains(text(), 'No data')]")
            if no_data_elements:
                for element in no_data_elements:
                    self.logger.warning(f"No data message: {element.text}")
            
            self.logger.info("=== END TABLE DEBUG ===")
            
        except Exception as e:
            self.logger.error(f"Debug method failed: {str(e)}")
    
    def _has_next_page(self):
        """Check if there is a next page available using flexible selectors"""
        try:
            # Method 1: Look for Next Page button by title attribute (most reliable)
            next_buttons = self.driver.find_elements(
                By.XPATH, "//input[@title='Next Page' and contains(@name, 'RadGrid_OverallResults')]"
            )
            
            if next_buttons:
                next_button = next_buttons[0]
                is_enabled = next_button.is_enabled()
                button_class = next_button.get_attribute("class") or ""
                is_not_disabled = "disabled" not in button_class.lower()
                
                self.logger.debug(f"Next button found: enabled={is_enabled}, class='{button_class}'")
                return is_enabled and is_not_disabled
            
            # Method 2: Look for Next Page button by class
            next_buttons_by_class = self.driver.find_elements(By.CLASS_NAME, "rgPageNext")
            if next_buttons_by_class:
                next_button = next_buttons_by_class[0]
                is_enabled = next_button.is_enabled()
                button_class = next_button.get_attribute("class") or ""
                is_not_disabled = "disabled" not in button_class.lower()
                
                self.logger.debug(f"Next button (by class) found: enabled={is_enabled}, class='{button_class}'")
                return is_enabled and is_not_disabled
            
            # Method 3: Check pagination info to see if we're on the last page
            try:
                # Look for pagination info like "1129 items in 23 pages"
                info_elements = self.driver.find_elements(By.CLASS_NAME, "rgInfoPart")
                if info_elements:
                    info_text = info_elements[0].text
                    self.logger.debug(f"Pagination info: {info_text}")
                    
                    # Extract current page and total pages
                    match = re.search(r'(\d+)\s+items\s+in\s+(\d+)\s+pages', info_text)
                    if match:
                        total_items = int(match.group(1))
                        total_pages = int(match.group(2))
                        
                        # Find current page by looking for rgCurrentPage
                        current_page_elements = self.driver.find_elements(By.CLASS_NAME, "rgCurrentPage")
                        if current_page_elements:
                            current_page_text = current_page_elements[0].text.strip()
                            if current_page_text.isdigit():
                                current_page = int(current_page_text)
                                has_next = current_page < total_pages
                                
                                self.logger.info(f"Pagination: Page {current_page} of {total_pages} (has_next: {has_next})")
                                return has_next
            except Exception as e:
                self.logger.debug(f"Error checking pagination info: {str(e)}")
            
            self.logger.warning("No Next Page button found using any method")
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for next page: {str(e)}")
            return False
    
    def _click_next_page(self):
        """Click the next page button using flexible selectors"""
        try:
            next_button = None
            
            # Method 1: Find by title attribute
            next_buttons = self.driver.find_elements(
                By.XPATH, "//input[@title='Next Page' and contains(@name, 'RadGrid_OverallResults')]"
            )
            
            if next_buttons:
                next_button = next_buttons[0]
                self.logger.debug("Found Next button by title attribute")
            else:
                # Method 2: Find by class
                next_buttons_by_class = self.driver.find_elements(By.CLASS_NAME, "rgPageNext")
                if next_buttons_by_class:
                    next_button = next_buttons_by_class[0]
                    self.logger.debug("Found Next button by class")
            
            if next_button and next_button.is_enabled():
                # Log button details for debugging
                button_name = next_button.get_attribute("name")
                self.logger.debug(f"Clicking Next button: {button_name}")
                
                next_button.click()
                
                # Wait for page to load
                time.sleep(self.delay)
                
                # Wait for table to be updated with longer timeout
                wait = WebDriverWait(self.driver, 15)
                wait.until(EC.presence_of_element_located((By.ID, "RadGrid_OverallResults_ctl00")))
                
                # Additional wait for content to fully load
                time.sleep(2)
                
                self.logger.info("Successfully navigated to next page")
                return True
            else:
                if next_button:
                    self.logger.info("Next button found but is disabled")
                else:
                    self.logger.info("No Next button found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to click next page: {str(e)}")
            return False
    
    def _generate_filename(self, term):
        """
        Generate filename based on term
        
        Args:
            term (str): Term like '2025-26 Term 1'
            
        Returns:
            str: Filename like '2025-26_T1.xlsx'
        """
        # Convert term format
        term_map = {
            'Term 1': 'T1',
            'Term 2': 'T2', 
            'Term 3A': 'T3A',
            'Term 3B': 'T3B'
        }
        
        filename = term
        for full_term, short_term in term_map.items():
            if full_term in term:
                filename = term.replace(full_term, short_term)
                break
        
        return filename + '.xlsx'
    
    def _save_to_excel(self, data, filename):
        """
        Save data to Excel file, concatenating if file exists
        
        Args:
            data (list): List of dictionaries containing the data
            filename (str): Excel filename to save to
        """
        try:
            if not data:
                self.logger.warning("No data to save")
                return
            
            # Create DataFrame with desired column order
            new_df = pd.DataFrame(data)
            new_df = new_df[self.desired_columns]
            
            # Check if file exists
            if os.path.exists(filename):
                self.logger.info(f"File {filename} exists, concatenating data...")
                
                try:
                    # Read existing data
                    existing_df = pd.read_excel(filename, engine='openpyxl')
                    
                    # Concatenate and remove duplicates
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
                    
                    self.logger.info(f"Combined {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records")
                    
                except Exception as e:
                    self.logger.error(f"Error reading existing file, creating new: {str(e)}")
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Save to Excel
            combined_df.to_excel(filename, index=False, engine='openpyxl')
            
            self.logger.info(f"Data saved to {filename}")
            self.logger.info(f"Total records in file: {len(combined_df)}")
            
            return len(combined_df)
            
        except Exception as e:
            self.logger.error(f"Failed to save data to Excel: {str(e)}")
            raise
    
    def _get_current_page_info(self):
        """Get current page number and total pages"""
        try:
            # Method 1: From pagination info text
            info_elements = self.driver.find_elements(By.CLASS_NAME, "rgInfoPart")
            if info_elements:
                info_text = info_elements[0].text
                # Extract total pages from text like "1129 items in 23 pages"
                match = re.search(r'(\d+)\s+items\s+in\s+(\d+)\s+pages', info_text)
                if match:
                    total_items = int(match.group(1))
                    total_pages = int(match.group(2))
                    
                    # Get current page from rgCurrentPage element
                    current_page_elements = self.driver.find_elements(By.CLASS_NAME, "rgCurrentPage")
                    if current_page_elements:
                        current_page_text = current_page_elements[0].text.strip()
                        if current_page_text.isdigit():
                            current_page = int(current_page_text)
                            return current_page, total_pages, total_items
            
            return None, None, None
            
        except Exception as e:
            self.logger.debug(f"Error getting page info: {str(e)}")
            return None, None, None
    
    def scrape_term_data(self, term, bid_round=None, bid_window=None, output_dir="./"):
        """
        Scrape data for a specific term with improved pagination handling
        
        Args:
            term (str): Term to scrape (e.g., '2025-26 Term 1')
            bid_round (str): Specific bid round to filter by
            bid_window (str): Specific bid window to filter by
            output_dir (str): Directory to save the Excel file
        """
        try:
            # Setup driver if not already done
            if self.driver is None:
                self._setup_driver()
            
            # Navigate to page
            self._navigate_to_overall_results()
            
            # Fill form
            self._select_course_career("Undergraduate")
            self._select_term(term)
            self._select_bid_round(bid_round)
            self._select_bid_window(bid_window)
            
            # Submit search
            self._click_search()
            
            # Set page size to 50
            self._set_page_size_to_50()

            # Sort by bidding window to get Incoming Freshmen first
            self._sort_by_bidding_window()
            
            # Get initial page information
            current_page, total_pages, total_items = self._get_current_page_info()
            if total_pages:
                self.logger.info(f"Starting scrape: {total_items} total items across {total_pages} pages")
            
            # Collect all data from all pages
            all_data = []
            page_num = 1
            max_pages = 200  # Increased safety limit
            last_bidding_window = None
            should_stop_scraping = False

            while page_num <= max_pages and not should_stop_scraping:
                # Get current page info for verification
                current_page, total_pages, total_items = self._get_current_page_info()
                
                if current_page and total_pages:
                    self.logger.info(f"Scraping page {current_page} of {total_pages} (iteration {page_num})...")
                else:
                    self.logger.info(f"Scraping page {page_num}...")
                
                # Extract data from current page with early termination support
                page_data, should_stop, current_bidding_window = self._extract_table_data(
                    stop_on_bidding_window_change=True, 
                    last_bidding_window=last_bidding_window
                )
                
                # Update tracking variables
                if current_bidding_window:
                    last_bidding_window = current_bidding_window
                if should_stop:
                    should_stop_scraping = True
                
                if page_data:
                    all_data.extend(page_data)
                    self.logger.info(f"Page {page_num}: Found {len(page_data)} records")
                    if should_stop_scraping:
                        self.logger.info("Early termination triggered due to bidding window change")
                        break
                else:
                    self.logger.warning(f"Page {page_num}: No data found")
                    
                    # If first page has no data, something is wrong
                    if page_num == 1:
                        self.logger.error("No data on first page - check search criteria or page structure")
                        break
                
                # Check if we've reached the last page using multiple methods
                if current_page and total_pages and current_page >= total_pages:
                    self.logger.info(f"Reached last page ({current_page}/{total_pages})")
                    break
                
                # Check if there's a next page using our improved method
                if self._has_next_page():
                    # Store current page for verification
                    old_page = current_page
                    
                    if self._click_next_page():
                        # Verify we actually moved to next page
                        time.sleep(1)  # Brief wait
                        new_current_page, _, _ = self._get_current_page_info()
                        
                        if new_current_page and old_page and new_current_page <= old_page:
                            self.logger.warning(f"Page didn't advance (was {old_page}, now {new_current_page})")
                            break
                        
                        page_num += 1
                        time.sleep(self.delay)  # Rate limiting
                    else:
                        self.logger.info("Failed to navigate to next page, stopping")
                        break
                else:
                    self.logger.info("No more pages available")
                    break
            
            # Generate filename and save data
            if all_data:
                filename = self._generate_filename(term)
                filepath = os.path.join(output_dir, filename)
                
                total_records = self._save_to_excel(all_data, filepath)
                
                self.logger.info(f"Scraping completed for {term}")
                self.logger.info(f"Records collected this session: {len(all_data)}")
                self.logger.info(f"Total records in file: {total_records}")
                
                # Final verification
                if current_page and total_pages:
                    expected_total = total_items if total_items else "unknown"
                    self.logger.info(f"Expected ~{expected_total} total records from {total_pages} pages")
            else:
                self.logger.error("No data collected for any page")
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Failed to scrape term data: {str(e)}")
            raise
    
    def scrape_multiple_terms(self, terms_config, output_dir="./", automated_login=False):
        """
        Scrape data for multiple terms
        
        Args:
            terms_config (list): List of dictionaries with term configurations
                                Example: [
                                    {'term': '2025-26 Term 1', 'round': '1', 'window': '1'},
                                    {'term': '2024-25 Term 2', 'round': None, 'window': None}
                                ]
            output_dir (str): Directory to save Excel files
            automated_login (bool): If True, use automated login with TOTP. 
                                   If False, wait for manual login.
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup driver
            self._setup_driver()
            
            # Navigate to login page
            self.driver.get("https://boss.intranet.smu.edu.sg/")
            
            # Perform login (automated or manual)
            if automated_login:
                perform_automated_login(self.driver, logger=self.logger)
            else:
                wait_for_manual_login(self.driver, logger=self.logger)
            
            # Process each term configuration
            for i, config in enumerate(terms_config):
                try:
                    self.logger.info(f"Processing term {i+1}/{len(terms_config)}: {config['term']}")
                    
                    data = self.scrape_term_data(
                        term=config['term'],
                        bid_round=config.get('round'),
                        bid_window=config.get('window'),
                        output_dir=output_dir
                    )
                    
                    self.logger.info(f"Completed {config['term']}: {len(data)} records")
                    
                    # Delay between terms
                    if i < len(terms_config) - 1:
                        self.logger.info(f"Waiting {self.delay} seconds before next term...")
                        time.sleep(self.delay)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process term {config['term']}: {str(e)}")
                    continue
            
            self.logger.info("All terms processing completed")
            
        except Exception as e:
            self.logger.error(f"Failed to scrape multiple terms: {str(e)}")
            raise
        finally:
            self.close()

    def run(self, term, bid_round=None, bid_window=None, output_dir="./script_input/overallBossResults", auto_detect_phase=True, automated_login=False):
        """
        Runs the scraper for a single term, handling term format transformation internally.
        
        Args:
            term (str): Term to scrape in short format (e.g., '2025-26_T1').
            bid_round (str): Specific bid round to filter by.
            bid_window (str): Specific bid window to filter by.
            output_dir (str): Directory to save output files.
            auto_detect_phase (bool): Whether to auto-detect current bidding phase.
            automated_login (bool): If True, use automated login with TOTP. 
                                   If False, wait for manual login.
        """
        try:
            # First, transform the short-form term into the website-friendly format.
            website_term = transform_term_format(term)
            
            # Auto-detect current bidding phase if enabled and no explicit round/window provided
            if auto_detect_phase and (bid_round is None or bid_window is None):
                detected_round, detected_window = self._determine_current_bidding_phase()
                if detected_round and detected_window:
                    if bid_round is None: bid_round = detected_round
                    if bid_window is None: bid_window = detected_window
                    self.logger.info(f"Auto-detected bidding phase: Round {bid_round}, Window {bid_window}")
                else:
                    self.logger.info("Could not auto-detect a current bidding phase. Scraping with default filters.")
            
            round_str = f"Round {bid_round}" if bid_round else "All Rounds"
            window_str = f"Window {bid_window}" if bid_window else "All Windows"
            self.logger.info(f"Scraping {website_term} - {round_str}, {window_str}")
            
            self._setup_driver()
            self.driver.get("https://boss.intranet.smu.edu.sg/")
            
            if automated_login:
                perform_automated_login(self.driver, logger=self.logger)
            else:
                wait_for_manual_login(self.driver, logger=self.logger)
            
            # Scrape the term data using the correctly formatted term.
            data = self.scrape_term_data(
                term=website_term,
                bid_round=bid_round,
                bid_window=bid_window,
                output_dir=output_dir
            )
            
            print(f"Scraping completed! Collected {len(data)} records for {website_term}")
            return True  # Return True on success
            
        except Exception as e:
            print(f"An error occurred during the scraping process: {str(e)}")
            self.logger.error(f"Error during scraping: {str(e)}")
            return False  # Return False on failure
        finally:
            self.close()

    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.logger.info("WebDriver closed")


if __name__ == "__main__":
    scraper = ScrapeOverallResults(headless=False, delay=5)
    success = scraper.run(
        term=START_AY_TERM,
        bid_round=TARGET_ROUND,
        bid_window=TARGET_WINDOW,
        auto_detect_phase=True  # Ensure auto-detection is enabled.
    )
    if not success:
        sys.exit(1) # Exit with error
