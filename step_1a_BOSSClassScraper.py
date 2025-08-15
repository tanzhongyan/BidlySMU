# Import global configuration settings
from config import *

# Import dependencies
import os
import csv
import sys
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

class BOSSClassScraper:
    """
    A class to scrape class details from BOSS (SMU's online class registration system).
    It performs a full scan for the first bidding window of a term and targeted
    re-scrapes for subsequent windows based on previously found classes.
    """
    
    def __init__(self):
        """
        Initialize the BOSS Class Scraper with configuration parameters.
        """
        self.term_code_map = {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}
        self.all_terms = ['T1', 'T2', 'T3A', 'T3B']
        self.driver = None
        self.min_class_number = 1000
        self.max_class_number = 5000
        self.consecutive_empty_threshold = 300
        
        # Use the global bidding schedule
        self.bidding_schedule = BIDDING_SCHEDULES

    def _get_bidding_round_info_for_term(self, ay_term, now):
        """
        Determines the bidding round folder name for a given academic term based on the current time.
        """
        # Get the schedule for the specific academic term
        schedule = self.bidding_schedule.get(ay_term)
        if not schedule:
            return None

        # Find the correct window from the schedule
        for results_date, _, folder_suffix in schedule:
            if now < results_date:
                return f"{ay_term}_{folder_suffix}"
        return None

    def wait_for_manual_login(self):
        """Wait for manual login and Microsoft Authenticator process completion."""
        print("Please log in manually and complete the Microsoft Authenticator process.")
        print("Waiting for BOSS dashboard to load...")
        wait = WebDriverWait(self.driver, 120)
        try:
            wait.until(EC.presence_of_element_located((By.ID, "Label_UserName")))
            username = self.driver.find_element(By.ID, "Label_UserName").text
            print(f"Login successful! Logged in as {username}")
        except TimeoutException:
            raise Exception("Login failed or timed out.")
        time.sleep(1)

    def scrape_and_save_html(self, start_ay_term=START_AY_TERM, end_ay_term=END_AY_TERM, base_dir='script_input/classTimingsFull'):
        """
        Scrapes class details, always performing a full scan from 1000-5000.
        """
        now = datetime.now()
        start_year = int(start_ay_term[:4])
        end_year = int(end_ay_term[:4])
        all_academic_years = [f"{year}-{(year + 1) % 100:02d}" for year in range(start_year, end_year + 1)]
        all_ay_terms = [f"{ay}_{term}" for ay in all_academic_years for term in self.all_terms]
        
        try:
            start_idx = all_ay_terms.index(start_ay_term)
            end_idx = all_ay_terms.index(end_ay_term)
        except ValueError:
            print("Invalid start or end term provided.")
            return
            
        ay_terms_to_scrape = all_ay_terms[start_idx:end_idx+1]
        
        for ay_term in ay_terms_to_scrape:
            print(f"\nProcessing Academic Term: {ay_term}")
            
            round_window_folder_name = self._get_bidding_round_info_for_term(ay_term, now)
            if not round_window_folder_name:
                print(f"Not in a bidding window for {ay_term} at this time. Skipping.")
                continue

            current_round_path = os.path.join(base_dir, ay_term, round_window_folder_name)
            os.makedirs(current_round_path, exist_ok=True)
            
            ay, term = ay_term.split('_')
            ay_short, term_code = ay[2:4], self.term_code_map.get(term, '10')

            # Always perform full scan regardless of previous rounds
            print(f"Performing full scan for {ay_term}.")
            consecutive_empty = 0
            for class_num in range(self.min_class_number, self.max_class_number + 1):
                was_scraped = self._scrape_single_class(current_round_path, ay_short, term_code, class_num)
                if was_scraped is None: # Error occurred, stop this scan
                    break
                if not was_scraped: # Page had no record
                    consecutive_empty += 1
                    if consecutive_empty >= self.consecutive_empty_threshold:
                        print(f"Stopping scan after {consecutive_empty} consecutive empty records.")
                        break
                else: # Successful scrape
                    consecutive_empty = 0
        print("\nScraping process completed.")
    
    def _scrape_single_class(self, target_path, ay_short, term_code, class_num):
        """
        Scrapes a single class number and saves the HTML, always overwriting existing files.
        Returns True if data was found, False if "No record found", None on error.
        """
        filename = f"SelectedAcadTerm={ay_short}{term_code}&SelectedClassNumber={class_num:04}.html"
        filepath = os.path.join(target_path, filename)

        # Remove the existing file check - always scrape
        url = f"https://boss.intranet.smu.edu.sg/ClassDetails.aspx?SelectedClassNumber={class_num:04}&SelectedAcadTerm={ay_short}{term_code}&SelectedAcadCareer=UGRD"
        
        try:
            self.driver.get(url)
            # Robust wait for either content or an error message
            WebDriverWait(self.driver, 15).until(EC.any_of(
                EC.visibility_of_element_located((By.ID, "RadGrid_MeetingInfo_ctl00")),
                EC.presence_of_element_located((By.ID, "lblErrorDetails"))
            ))
            
            page_source = self.driver.page_source
            if "No record found" in page_source:
                print(f"No record for class {class_num}")
                return False
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(page_source)
            print(f"Saved {filepath}")
            time.sleep(1) # Small delay to be polite
            return True
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            time.sleep(5)
            return None # Indicate an error occurred

    def generate_scraped_filepaths_csv(self, base_dir='script_input/classTimingsFull', output_csv='script_input/scraped_filepaths.csv'):
        """Generates/appends to a CSV file with paths to all valid HTML files."""
        existing_filepaths = set()
        if os.path.exists(output_csv):
            try:
                with open(output_csv, 'r', encoding='utf-8', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        if row: existing_filepaths.add(row[0])
            except (IOError, StopIteration) as e:
                print(f"Could not read existing CSV, will overwrite: {e}")

        new_filepaths = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.html'):
                    filepath = os.path.join(root, file)
                    if filepath not in existing_filepaths:
                        new_filepaths.append(filepath)
        
        if not new_filepaths:
            print("No new valid HTML files found to add to the CSV.")
            return

        mode = 'a' if existing_filepaths else 'w'
        with open(output_csv, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if mode == 'w':
                writer.writerow(['Filepath'])
            for path in new_filepaths:
                writer.writerow([path])
        
        print(f"CSV updated. Total valid files now: {len(existing_filepaths) + len(new_filepaths)}")

    def run_full_scraping_process(self, start_ay_term=START_AY_TERM, end_ay_term=END_AY_TERM):
        """Run the complete scraping process for a specified term range."""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            self.driver.get("https://boss.intranet.smu.edu.sg/")
            self.wait_for_manual_login()
            
            self.scrape_and_save_html(start_ay_term, end_ay_term)
            self.generate_scraped_filepaths_csv()
            
            return True
        except Exception as e:
            print(f"Error during scraping process: {str(e)}")
            return False
        finally:
            if self.driver:
                self.driver.quit()
            print("Process completed!")

if __name__ == "__main__":
    scraper = BOSSClassScraper()
    success = scraper.run_full_scraping_process(START_AY_TERM, END_AY_TERM)
    if not success:
        sys.exit(1) # Exit with error