# Import global configuration settings
from config import *

# Import dependencies
import os
import re
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class HTMLDataExtractor:
    """
    Extract raw data from scraped HTML files and save to Excel format using Selenium
    """
    
    def __init__(self):
        self.standalone_data = []
        self.multiple_data = []
        self.errors = []
        self.driver = None
        
    def setup_selenium_driver(self):
        """Set up Selenium WebDriver for local file access"""
        try:
            options = Options()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--headless')  # Run in headless mode for efficiency
            options.add_argument('--disable-gpu')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            print("Selenium WebDriver initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Selenium WebDriver: {e}")
            raise
    
    def safe_find_element_text(self, by, value):
        """Safely find element and return its text with proper encoding handling"""
        try:
            element = self.driver.find_element(by, value)
            if element:
                raw_text = element.text.strip()
                return self.clean_text_encoding(raw_text)
            return None
        except Exception:
            return None
    
    def safe_find_element_attribute(self, by, value, attribute):
        """Safely find element and return its attribute with proper encoding handling"""
        try:
            element = self.driver.find_element(by, value)
            if element:
                raw_attr = element.get_attribute(attribute)
                return self.clean_text_encoding(raw_attr) if raw_attr else None
            return None
        except Exception:
            return None
    
    def convert_date_to_timestamp(self, date_str):
        """Convert DD-Mmm-YYYY to database timestamp format"""
        try:
            date_obj = datetime.strptime(date_str, '%d-%b-%Y')
            return date_obj.strftime('%Y-%m-%d 00:00:00.000 +0800')
        except Exception as e:
            return None
    
    def parse_acad_term(self, term_text, filepath=None):
        """Parse academic term text and return structured data with folder path fallback"""
        try:
            # Clean the term text first
            if term_text:
                term_text = self.clean_text_encoding(term_text)
            
            # Pattern like "2021-22 Term 2" or "2021-22 Session 1"
            pattern = r'(\d{4})-(\d{2})\s+(.*)'
            match = re.search(pattern, term_text) if term_text else None
            
            if not match:
                return None, None, None, None
            
            start_year = int(match.group(1))
            end_year_short = int(match.group(2))
            term_desc = match.group(3).lower()
            
            # Convert 2-digit year to 4-digit
            if end_year_short < 50:
                end_year = 2000 + end_year_short
            else:
                end_year = 1900 + end_year_short
            
            # Determine term code from text
            term_code = None
            if 'term 1' in term_desc or 'session 1' in term_desc or 'august term' in term_desc:
                term_code = 'T1'
            elif 'term 2' in term_desc or 'session 2' in term_desc or 'january term' in term_desc:
                term_code = 'T2'
            elif 'term 3a' in term_desc:
                term_code = 'T3A'
            elif 'term 3b' in term_desc:
                term_code = 'T3B'
            elif 'term 3' in term_desc:
                # Generic T3 - need to check folder path for A/B
                term_code = 'T3'
            
            # If term_code is incomplete or missing, use folder path as fallback
            if not term_code or term_code == 'T3':
                folder_term = self.extract_term_from_folder_path(filepath) if filepath else None
                if folder_term:
                    # If we have folder term, use it
                    if term_code == 'T3' and folder_term in ['T3A', 'T3B']:
                        term_code = folder_term
                    elif not term_code:
                        term_code = folder_term
            
            # If still no term code, return None
            if not term_code:
                return start_year, end_year, None, None
            
            acad_term_id = f"AY{start_year}{end_year_short:02d}{term_code}"
            
            return start_year, end_year, term_code, acad_term_id
        except Exception as e:
            return None, None, None, None
    
    def parse_course_and_section(self, header_text):
        """Parse course code and section from header text with encoding fixes"""
        try:
            if not header_text:
                return None, None
            
            # Clean the text first
            clean_text = self.clean_text_encoding(header_text)
            clean_text = re.sub(r'<[^>]+>', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text.strip())
            
            # Try multiple regex patterns
            patterns = [
                r'([A-Z0-9_-]+)\s+—\s+(.+)',  # Standard format with em-dash
                r'([A-Z0-9_-]+)\s+-\s+(.+)',  # Standard format with hyphen
                r'([A-Z]+)\s+(\d+[A-Z0-9_]*)\s+—\s+(.+)',  # Split format with em-dash
                r'([A-Z]+)\s+(\d+[A-Z0-9_]*)\s+-\s+(.+)',  # Split format with hyphen
                r'([A-Z0-9_\s-]+?)\s+—\s+([^—]+)',  # Flexible format with em-dash
                r'([A-Z0-9_\s-]+?)\s+-\s+([^-]+)',  # Flexible format with hyphen
            ]
            
            for pattern in patterns:
                match = re.match(pattern, clean_text)
                if match:
                    if len(match.groups()) == 2:
                        # Standard format: course_code - section
                        course_section = match.group(1).strip()
                        section_name = match.group(2).strip()
                        
                        # Extract section from the end of course_section if it's there
                        section_match = re.search(r'^(.+?)\s+(SG\d+|G\d+|\d+)$', course_section)
                        if section_match:
                            course_code = section_match.group(1)
                            section = section_match.group(2)
                        else:
                            course_code = course_section
                            # Try to extract section from section_name
                            section_extract = re.search(r'(SG\d+|G\d+|\d+)', section_name)
                            section = section_extract.group(1) if section_extract else None
                    else:
                        # Split format: course_prefix course_number - section_name
                        course_code = f"{match.group(1)}{match.group(2)}"
                        section_name = match.group(3).strip()
                        section_extract = re.search(r'(SG\d+|G\d+|\d+)', section_name)
                        section = section_extract.group(1) if section_extract else None
                    
                    return course_code.strip() if course_code else None, section
            
            return None, None
        except Exception as e:
            return None, None
    
    def parse_date_range(self, date_text):
        """Parse date range text and return start and end timestamps"""
        try:
            # Example: "10-Jan-2022 to 01-May-2022"
            pattern = r'(\d{1,2}-\w{3}-\d{4})\s+to\s+(\d{1,2}-\w{3}-\d{4})'
            match = re.search(pattern, date_text)
            
            if not match:
                return None, None
            
            start_date = self.convert_date_to_timestamp(match.group(1))
            end_date = self.convert_date_to_timestamp(match.group(2))
            
            return start_date, end_date
        except Exception as e:
            return None, None
    
    def extract_course_areas_list(self):
        """Extract course areas with encoding fixes"""
        try:
            course_areas_element = self.driver.find_element(By.ID, 'lblCourseAreas')
            if not course_areas_element:
                return None
            
            # Get innerHTML to handle HTML content
            course_areas_html = course_areas_element.get_attribute('innerHTML')
            if course_areas_html:
                # Clean encoding first
                course_areas_html = self.clean_text_encoding(course_areas_html)
                
                # Extract list items
                areas_list = re.findall(r'<li[^>]*>([^<]+)</li>', course_areas_html)
                if areas_list:
                    # Clean each area and join
                    cleaned_areas = [self.clean_text_encoding(area.strip()) for area in areas_list]
                    return ', '.join(cleaned_areas)
                else:
                    # Fallback to text content
                    text_content = course_areas_element.text.strip()
                    return self.clean_text_encoding(text_content)
            else:
                # Fallback to text content
                text_content = course_areas_element.text.strip()
                return self.clean_text_encoding(text_content)
        except Exception:
            return None
    
    def extract_course_outline_url(self):
        """Extract course outline URL from HTML using Selenium"""
        try:
            onclick_attr = self.safe_find_element_attribute(By.ID, 'imgCourseOutline', 'onclick')
            if onclick_attr:
                url_match = re.search(r"window\.open\('([^']+)'", onclick_attr)
                if url_match:
                    return url_match.group(1)
        except Exception:
            pass
        return None
    
    def extract_boss_ids_from_filepath(self, filepath):
        """Extract BOSS IDs from filepath"""
        try:
            filename = os.path.basename(filepath)
            acad_term_match = re.search(r'SelectedAcadTerm=(\d+)', filename)
            class_match = re.search(r'SelectedClassNumber=(\d+)', filename)
            
            acad_term_boss_id = int(acad_term_match.group(1)) if acad_term_match else None
            class_boss_id = int(class_match.group(1)) if class_match else None
            
            return acad_term_boss_id, class_boss_id
        except Exception:
            return None, None
    
    def extract_meeting_information(self, record_key):
        """Extract class timing and exam timing information using Selenium"""
        try:
            meeting_table = self.driver.find_element(By.ID, 'RadGrid_MeetingInfo_ctl00')
            tbody = meeting_table.find_element(By.TAG_NAME, 'tbody')
            rows = tbody.find_elements(By.TAG_NAME, 'tr')
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) < 7:
                    continue
                
                meeting_type = cells[0].text.strip()
                start_date_text = cells[1].text.strip()
                end_date_text = cells[2].text.strip()
                day_of_week = cells[3].text.strip()
                start_time = cells[4].text.strip()
                end_time = cells[5].text.strip()
                venue = cells[6].text.strip() if len(cells) > 6 else ""
                professor_name = cells[7].text.strip() if len(cells) > 7 else ""
                
                # Assume CLASS if meeting_type is empty
                if not meeting_type:
                    meeting_type = 'CLASS'
                
                if meeting_type == 'CLASS':
                    # Convert dates to timestamp format
                    start_date = self.convert_date_to_timestamp(start_date_text)
                    end_date = self.convert_date_to_timestamp(end_date_text)
                    
                    timing_record = {
                        'record_key': record_key,
                        'type': 'CLASS',
                        'start_date': start_date,
                        'end_date': end_date,
                        'day_of_week': day_of_week,
                        'start_time': start_time,
                        'end_time': end_time,
                        'venue': venue,
                        'professor_name': professor_name
                    }
                    self.multiple_data.append(timing_record)
                
                elif meeting_type == 'EXAM':
                    # For exams, use the second date (end_date_text) as the exam date
                    exam_date = self.convert_date_to_timestamp(end_date_text)
                    
                    exam_record = {
                        'record_key': record_key,
                        'type': 'EXAM',
                        'date': exam_date,
                        'day_of_week': day_of_week,
                        'start_time': start_time,
                        'end_time': end_time,
                        'venue': venue,
                        'professor_name': professor_name
                    }
                    self.multiple_data.append(exam_record)
        
        except Exception as e:
            self.errors.append({
                'record_key': record_key,
                'error': f'Error extracting meeting information: {str(e)}',
                'type': 'parse_error'
            })
    
    def process_html_file(self, filepath):
        """Process a single HTML file and extract all data using Selenium"""
        try:
            # Load HTML file
            html_file = Path(filepath).resolve()
            file_url = html_file.as_uri()
            self.driver.get(file_url)
            
            # Create unique record key
            record_key = f"{os.path.basename(filepath)}"
            
            # Extract basic information
            class_header_text = self.safe_find_element_text(By.ID, 'lblClassInfoHeader')
            if not class_header_text:
                self.errors.append({
                    'filepath': filepath,
                    'error': 'Missing class header',
                    'type': 'parse_error'
                })
                return False
            
            course_code, section = self.parse_course_and_section(class_header_text)
            
            # Extract academic term
            term_text = self.safe_find_element_text(By.ID, 'lblClassInfoSubHeader')
            acad_year_start, acad_year_end, term, acad_term_id = self.parse_acad_term(term_text, filepath) if term_text else (None, None, None, None)
            
            # Extract course information
            course_name = self.safe_find_element_text(By.ID, 'lblClassSection')
            course_description = self.safe_find_element_text(By.ID, 'lblCourseDescription')
            credit_units_text = self.safe_find_element_text(By.ID, 'lblUnits')
            course_areas = self.extract_course_areas_list()
            enrolment_requirements = self.safe_find_element_text(By.ID, 'lblEnrolmentRequirements')
            
            # Process credit units
            try:
                credit_units = float(credit_units_text) if credit_units_text else None
            except (ValueError, TypeError):
                credit_units = None
            
            # Extract grading basis
            grading_text = self.safe_find_element_text(By.ID, 'lblGradingBasis')
            grading_basis = None
            if grading_text:
                if grading_text.lower() == 'graded':
                    grading_basis = 'Graded'
                elif grading_text.lower() in ['pass/fail', 'pass fail']:
                    grading_basis = 'Pass/Fail'
                else:
                    grading_basis = 'NA'
            
            # Extract course outline URL
            course_outline_url = self.extract_course_outline_url()
            
            # Extract dates
            period_text = self.safe_find_element_text(By.ID, 'lblDates')
            start_dt, end_dt = self.parse_date_range(period_text) if period_text else (None, None)
            
            # Extract BOSS IDs
            acad_term_boss_id, class_boss_id = self.extract_boss_ids_from_filepath(filepath)
            
            # Extract bidding information
            total, current_enrolled, reserved, available = self.extract_bidding_info()
            
            # Get extraction date and determine bidding window from folder path
            extraction_date = datetime.now()
            bidding_window = self.determine_bidding_window_from_filepath(filepath)
            
            # Create standalone record
            standalone_record = {
                'record_key': record_key,
                'filepath': filepath,
                'course_code': course_code,
                'section': section,
                'course_name': course_name,
                'course_description': course_description,
                'credit_units': credit_units,
                'course_area': course_areas,
                'enrolment_requirements': enrolment_requirements,
                'acad_term_id': acad_term_id,
                'acad_year_start': acad_year_start,
                'acad_year_end': acad_year_end,
                'term': term,
                'start_dt': start_dt,
                'end_dt': end_dt,
                'grading_basis': grading_basis,
                'course_outline_url': course_outline_url,
                'acad_term_boss_id': acad_term_boss_id,
                'class_boss_id': class_boss_id,
                'term_text': term_text,
                'period_text': period_text,
                'total': total,
                'current_enrolled': current_enrolled,
                'reserved': reserved,
                'available': available,
                'date_extracted': extraction_date.strftime('%Y-%m-%d %H:%M:%S'),
                'bidding_window': bidding_window
            }
            
            self.standalone_data.append(standalone_record)
            
            # Extract meeting information
            self.extract_meeting_information(record_key)
            
            return True
            
        except Exception as e:
            self.errors.append({
                'filepath': filepath,
                'error': str(e),
                'type': 'processing_error'
            })
            return False

    def determine_bidding_window_from_filepath(self, filepath):
        """Determine bidding window from the file path structure"""
        try:
            # Extract folder name from path
            # e.g., script_input/classTimingsFull/2025-26_T1/2025-26_T1_R1W1/file.html
            folder_path = os.path.dirname(filepath)
            folder_name = os.path.basename(folder_path)
            
            return self.extract_bidding_window_from_folder(folder_name)
            
        except Exception as e:
            print(f"Error determining bidding window from filepath: {e}")
            return None

    def process_all_files(self, base_dir='script_input/classTimingsFull'):
        """Process only files from the latest round folder that haven't been processed yet"""
        try:
            # Find the current academic term (e.g., 2025-26_T1)
            current_term = self.get_current_academic_term()
            if not current_term:
                print("Could not determine current academic term")
                return
            
            print(f"Current academic term: {current_term}")
            print(f"Base directory: {base_dir}")
            
            term_path = os.path.join(base_dir, current_term)
            print(f"Term path: {term_path}")
            
            if not os.path.exists(term_path):
                print(f"Academic term folder not found: {term_path}")
                return
            
            # Find the latest round folder
            latest_round_folder = self.find_latest_round_folder(term_path)
            if not latest_round_folder:
                print(f"No round folders found in {term_path}")
                return
            
            latest_round_path = os.path.join(term_path, latest_round_folder)
            bidding_window = self.extract_bidding_window_from_folder(latest_round_folder)
            
            print(f"Processing latest round: {latest_round_folder}")
            print(f"Latest round path: {latest_round_path}")
            print(f"Bidding window: {bidding_window}")
            
            # Get all HTML files from the latest round folder
            html_files = []
            for filename in os.listdir(latest_round_path):
                if filename.endswith('.html'):
                    filepath = os.path.join(latest_round_path, filename)
                    html_files.append(filepath)
            
            if not html_files:
                print(f"No HTML files found in {latest_round_path}")
                return
            
            print(f"Found {len(html_files)} HTML files in latest round")
            
            # Load existing data to check what's already processed
            existing_standalone, _, _ = self.load_existing_data()
            
            # Filter files that haven't been processed for this bidding window
            files_to_process = []
            for filepath in html_files:
                record_key = os.path.basename(filepath)
                
                # Check if this file has already been processed for this bidding window
                if existing_standalone.empty:
                    files_to_process.append(filepath)
                else:
                    # Extract course info from filename to check against existing data
                    acad_term_boss_id, class_boss_id = self.extract_boss_ids_from_filepath(filepath)
                    
                    # Check if record exists for this bidding window
                    mask = (existing_standalone['acad_term_boss_id'] == acad_term_boss_id) & \
                        (existing_standalone['class_boss_id'] == class_boss_id) & \
                        (existing_standalone['bidding_window'] == bidding_window)
                    
                    if not mask.any():
                        files_to_process.append(filepath)
            
            if not files_to_process:
                print(f"All files from {latest_round_folder} have already been processed")
                return
            
            print(f"Processing {len(files_to_process)} new files from {latest_round_folder}")
            
            # Process only the new files
            processed_files = 0
            successful_files = 0
            
            for filepath in files_to_process:
                if os.path.exists(filepath):
                    # print(f"Processing: {os.path.basename(filepath)}")
                    if self.process_html_file(filepath):
                        successful_files += 1
                    processed_files += 1
                    
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{len(files_to_process)} files")
            
            print(f"Processing complete: {successful_files}/{processed_files} files successful")
            
        except Exception as e:
            print(f"Error in process_all_files: {e}")
            raise
    
    def save_to_excel(self, output_path='script_input/raw_data.xlsx'):
        """Save extracted data to Excel file, appending new records only"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load existing data
            existing_standalone, existing_multiple, existing_errors = self.load_existing_data(output_path)
            
            # Filter out duplicate standalone records
            new_standalone_records = []
            skipped_standalone = 0
            
            for record in self.standalone_data:
                if not self.check_record_exists(existing_standalone, record):
                    new_standalone_records.append(record)
                else:
                    skipped_standalone += 1
            
            # Handle multiple records - replace if changes detected per record_key
            records_to_update = set()
            updated_multiple = 0
            
            # Group new records by record_key
            new_records_by_key = {}
            for record in self.multiple_data:
                key = record['record_key']
                if key not in new_records_by_key:
                    new_records_by_key[key] = []
                new_records_by_key[key].append(record)
            
            # Check each record_key for changes
            for record_key, new_records in new_records_by_key.items():
                if existing_multiple.empty:
                    records_to_update.add(record_key)
                else:
                    # Get existing records for this record_key
                    existing_records = existing_multiple[existing_multiple['record_key'] == record_key]
                    
                    # Check if there are any changes
                    changes_detected = False
                    
                    # Compare counts first
                    if len(existing_records) != len(new_records):
                        changes_detected = True
                    else:
                        # Compare each record
                        for new_record in new_records:
                            # Find matching existing record
                            existing_match = existing_records[existing_records['type'] == new_record['type']]
                            
                            if existing_match.empty:
                                changes_detected = True
                                break
                            
                            # Compare relevant fields based on type
                            if new_record['type'] == 'CLASS':
                                compare_fields = ['start_date', 'end_date', 'day_of_week', 'start_time', 'end_time', 'venue', 'professor_name']
                            elif new_record['type'] == 'EXAM':
                                compare_fields = ['date', 'day_of_week', 'start_time', 'end_time', 'venue', 'professor_name']
                            else:
                                continue
                            
                            for field in compare_fields:
                                if new_record.get(field) != existing_match.iloc[0].get(field):
                                    changes_detected = True
                                    break
                            
                            if changes_detected:
                                break
                    
                    if changes_detected:
                        records_to_update.add(record_key)
                        updated_multiple += 1
            
            # Build final multiple records list
            if existing_multiple.empty:
                final_multiple_records = self.multiple_data
            else:
                # Keep existing records that are not being updated
                final_multiple_records = []
                for _, row in existing_multiple.iterrows():
                    if row['record_key'] not in records_to_update:
                        final_multiple_records.append(row.to_dict())
                
                # Add new records for updated record_keys
                for record_key in records_to_update:
                    final_multiple_records.extend(new_records_by_key[record_key])
            
            new_multiple_df = pd.DataFrame(final_multiple_records)
            
            # Create DataFrames for new records
            new_standalone_df = pd.DataFrame(new_standalone_records)
            new_errors_df = pd.DataFrame(self.errors)
            
            # Combine with existing data
            if not existing_standalone.empty and not new_standalone_df.empty:
                combined_standalone = pd.concat([existing_standalone, new_standalone_df], ignore_index=True)
            elif not new_standalone_df.empty:
                combined_standalone = new_standalone_df
            else:
                combined_standalone = existing_standalone
            
            # Multiple records are already handled above
            combined_multiple = new_multiple_df
            
            if not existing_errors.empty and not new_errors_df.empty:
                combined_errors = pd.concat([existing_errors, new_errors_df], ignore_index=True)
            elif not new_errors_df.empty:
                combined_errors = new_errors_df
            else:
                combined_errors = existing_errors
            
            # Save to Excel with multiple sheets
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                        combined_standalone.to_excel(writer, sheet_name='standalone', index=False)
                        combined_multiple.to_excel(writer, sheet_name='multiple', index=False)
                        
                        if not combined_errors.empty:
                            combined_errors.to_excel(writer, sheet_name='errors', index=False)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"Excel file is locked. Retrying in 2 seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        print(f"Failed to save Excel file after {max_retries} attempts. Please close the file and try again.")
                        raise
            
            print(f"Data saved to {output_path}")
            print(f"New standalone records added: {len(new_standalone_records)}")
            print(f"Skipped duplicate standalone records: {skipped_standalone}")
            print(f"Total standalone records: {len(combined_standalone)}")
            print(f"Updated multiple record keys: {updated_multiple}")
            print(f"Total multiple records: {len(combined_multiple)}")
            if self.errors:
                print(f"New errors: {len(self.errors)}")
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            raise
    
    def get_all_round_folders(self, term_path):
        """Get all round folders in chronological order"""
        try:
            # Get all subdirectories
            subdirs = [d for d in os.listdir(term_path) if os.path.isdir(os.path.join(term_path, d))]

            # Filter for round folders (should contain R and W)
            round_folders = [d for d in subdirs if 'R' in d and 'W' in d]

            if not round_folders:
                return []

            # Sort by folder name chronologically
            round_folders.sort(key=lambda x: (
                int(x.split('R')[1].split('W')[0].replace('A', '').replace('B', '').replace('C', '').replace('F', '')),
                x.count('A') + x.count('B') * 2 + x.count('C') * 3 + x.count('F') * 4,
                int(x.split('W')[1])
            ))

            return round_folders

        except Exception as e:
            print(f"Error getting all round folders: {e}")
            return []

    def fill_missing_window(self, term, missing_window_name, source_window_name):
        """
        Fill missing window data by carrying forward from the most recent prior window.
        Adds metadata flags to indicate carried-forward data.

        Args:
            term (str): Academic term (e.g., '2025-26_T1')
            missing_window_name (str): Window name to fill (e.g., 'Round 1A Window 2')
            source_window_name (str): Source window to copy from (e.g., 'Round 1A Window 1')

        Returns:
            int: Number of records created
        """
        try:
            print(f"\n  Filling missing window: {missing_window_name}")
            print(f"  Carrying forward from: {source_window_name}")

            # Load existing data
            existing_standalone, _, _ = self.load_existing_data()

            if existing_standalone.empty:
                print("  No existing data to carry forward")
                return 0

            # Find records from source window
            source_records = existing_standalone[
                (existing_standalone['bidding_window'] == source_window_name)
            ]

            if source_records.empty:
                print(f"  No records found in source window: {source_window_name}")
                return 0

            print(f"  Found {len(source_records)} records in source window")

            # Check if target window already has data
            existing_target = existing_standalone[
                (existing_standalone['bidding_window'] == missing_window_name)
            ]

            if not existing_target.empty:
                print(f"  Target window already has {len(existing_target)} records, skipping")
                return 0

            # Create new records with updated window and metadata
            records_created = 0
            for _, record in source_records.iterrows():
                # Create a copy of the record
                new_record = record.to_dict()

                # Update bidding window
                new_record['bidding_window'] = missing_window_name

                # Add metadata flags
                new_record['data_source'] = 'carried_forward'
                new_record['source_window'] = source_window_name

                # Add to standalone data
                self.standalone_data.append(new_record)
                records_created += 1

            print(f"  ✅ Created {records_created} carried-forward records")
            return records_created

        except Exception as e:
            print(f"  ❌ Error filling missing window: {e}")
            return 0

    def process_all_windows_with_gap_filling(self, base_dir='script_input/classTimingsFull'):
        """
        Process ALL round folders and fill gaps for missing windows.
        Uses BIDDING_SCHEDULES to detect missing windows and carry forward data.
        """
        try:
            print("="*70)
            print("PROCESSING ALL WINDOWS WITH GAP FILLING")
            print("="*70)

            # Find the current academic term
            current_term = self.get_current_academic_term()
            if not current_term:
                print("Could not determine current academic term")
                return

            print(f"\nCurrent academic term: {current_term}")

            # Get bidding schedule for this term
            if current_term not in BIDDING_SCHEDULES:
                print(f"No bidding schedule found for term: {current_term}")
                return

            schedule = BIDDING_SCHEDULES[current_term]
            current_time = datetime.now()

            # Get list of past windows that should exist
            expected_windows = []
            for schedule_time, window_name, folder_suffix in schedule:
                if schedule_time < current_time:
                    expected_windows.append((schedule_time, window_name, folder_suffix))

            print(f"Expected {len(expected_windows)} past windows based on schedule")

            term_path = os.path.join(base_dir, current_term)
            if not os.path.exists(term_path):
                print(f"Academic term folder not found: {term_path}")
                return

            # Get all existing round folders
            all_round_folders = self.get_all_round_folders(term_path)
            print(f"Found {len(all_round_folders)} existing round folders")

            # Create map of folder_suffix to folder name
            existing_folders_map = {}
            for folder in all_round_folders:
                suffix = folder.split('_')[-1]  # e.g., 'R1W1'
                existing_folders_map[suffix] = folder

            # Process each expected window
            total_processed = 0
            total_filled = 0
            most_recent_window = None

            for i, (schedule_time, window_name, folder_suffix) in enumerate(expected_windows):
                print(f"\n--- Window {i+1}/{len(expected_windows)}: {window_name} ---")

                folder_name = f"{current_term}_{folder_suffix}"
                folder_path = os.path.join(term_path, folder_name)

                if os.path.exists(folder_path):
                    # Process HTML files from this folder
                    print(f"  ✅ Folder exists: {folder_name}")

                    # Get HTML files
                    html_files = []
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.html'):
                            filepath = os.path.join(folder_path, filename)
                            html_files.append(filepath)

                    if html_files:
                        print(f"  Processing {len(html_files)} HTML files...")

                        # Load existing data to check what's already processed
                        existing_standalone, _, _ = self.load_existing_data()

                        # Filter files that haven't been processed for this bidding window
                        files_to_process = []
                        for filepath in html_files:
                            acad_term_boss_id, class_boss_id = self.extract_boss_ids_from_filepath(filepath)

                            # Check if record exists for this bidding window
                            if existing_standalone.empty:
                                files_to_process.append(filepath)
                            else:
                                mask = (existing_standalone['acad_term_boss_id'] == acad_term_boss_id) & \
                                    (existing_standalone['class_boss_id'] == class_boss_id) & \
                                    (existing_standalone['bidding_window'] == window_name)

                                if not mask.any():
                                    files_to_process.append(filepath)

                        if files_to_process:
                            print(f"  Processing {len(files_to_process)} new files...")
                            successful = 0
                            for filepath in files_to_process:
                                if self.process_html_file(filepath):
                                    successful += 1
                            print(f"  ✅ Processed {successful}/{len(files_to_process)} files successfully")
                            total_processed += successful
                        else:
                            print(f"  All files already processed")

                        # Update most recent window
                        most_recent_window = window_name
                    else:
                        print(f"  No HTML files found in folder")

                else:
                    # Folder doesn't exist - fill gap with carried-forward data
                    print(f"  ❌ Folder missing: {folder_name}")

                    if most_recent_window:
                        filled = self.fill_missing_window(current_term, window_name, most_recent_window)
                        total_filled += filled
                    else:
                        print(f"  Cannot fill - no prior window data available yet")

            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")
            print(f"Total HTML files processed: {total_processed}")
            print(f"Total records carried forward: {total_filled}")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"Error in process_all_windows_with_gap_filling: {e}")
            raise

    def run(self, output_path='script_input/raw_data.xlsx'):
        """Run the complete extraction process"""
        print("Starting HTML data extraction...")

        # Reset data containers
        self.standalone_data = []
        self.multiple_data = []
        self.errors = []

        # Set up Selenium driver
        self.setup_selenium_driver()

        try:
            # Process files from latest round folder only
            self.process_all_files()  # Use default base_dir parameter

            # Save to Excel
            self.save_to_excel(output_path)

            print("HTML data extraction completed!")
            return True # Return True on success

        except Exception as e:
            print(f"❌ An error occurred during HTML data extraction: {e}")
            return False # Return False on failure

        finally:
            if self.driver:
                self.driver.quit()
                print("Selenium driver closed")

    def run_with_catchup(self, output_path='script_input/raw_data.xlsx'):
        """
        Run extraction with catch-up mode: process ALL windows and fill gaps.
        This is the intelligent catch-up version that automatically handles missing windows.
        """
        print("="*70)
        print("HTML DATA EXTRACTION WITH CATCH-UP")
        print("="*70)

        # Reset data containers
        self.standalone_data = []
        self.multiple_data = []
        self.errors = []

        # Set up Selenium driver
        self.setup_selenium_driver()

        try:
            # Process all windows with gap filling
            self.process_all_windows_with_gap_filling()

            # Save to Excel
            self.save_to_excel(output_path)

            print("\n✅ HTML data extraction with catch-up completed!")
            return True

        except Exception as e:
            print(f"\n❌ An error occurred during HTML data extraction: {e}")
            return False

        finally:
            if self.driver:
                self.driver.quit()
                print("Selenium driver closed")

    def clean_text_encoding(self, text):
        """Clean text to fix encoding issues like â€" -> —"""
        if not text:
            return text
        
        # Common encoding fixes - ORDER MATTERS! Process longer patterns first
        encoding_fixes = [
            ('â€"', '—'),   # em-dash
            ('â€™', "'"),   # right single quotation mark
            ('â€œ', '"'),   # left double quotation mark
            ('â€¦', '…'),   # horizontal ellipsis
            ('â€¢', '•'),   # bullet
            ('â€‹', ''),    # zero-width space
            ('â€‚', ' '),   # en space
            ('â€ƒ', ' '),   # em space
            ('â€‰', ' '),   # thin space
            ('â€', '"'),    # right double quotation mark (shorter pattern, process last)
            ('Â', ''),      # non-breaking space artifacts
        ]
        
        cleaned_text = text
        # Process in order to avoid substring conflicts
        for bad, good in encoding_fixes:
            cleaned_text = cleaned_text.replace(bad, good)
        
        # Remove any remaining problematic characters
        cleaned_text = re.sub(r'â€[^\w]', '', cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_term_from_folder_path(self, filepath):
        """Extract term from folder path as fallback
        E.g., script_input\\classTimingsFull\\2023-24_T3A -> T3A"""
        try:
            # Get the folder path
            folder_path = os.path.dirname(filepath)
            folder_name = os.path.basename(folder_path)
            
            # Look for term pattern in folder name
            # Pattern: YYYY-YY_TXX or YYYY-YY_TXXA
            term_match = re.search(r'(\d{4}-\d{2})_T(\w+)', folder_name)
            if term_match:
                return f"T{term_match.group(2)}"
            
            # Fallback: look for any T followed by alphanumeric
            term_fallback = re.search(r'T(\w+)', folder_name)
            if term_fallback:
                return f"T{term_fallback.group(1)}"
            
            return None
        except Exception as e:
            return None

    def extract_bidding_info(self):
        """Extract current bidding information from HTML elements"""
        try:
            # Extract Total
            total = self.safe_find_element_text(By.ID, 'lblClassCapacity')
            total = int(total) if total and total.isdigit() else None
            
            # Extract Current Enrolled
            current_enrolled = self.safe_find_element_text(By.ID, 'lblEnrolmentTotal')
            current_enrolled = int(current_enrolled) if current_enrolled and current_enrolled.isdigit() else None
            
            # Extract Reserved (for incoming students)
            reserved = self.safe_find_element_text(By.ID, 'lblReserved')
            reserved = int(reserved) if reserved and reserved.isdigit() else None
            
            # Extract Available Seats
            available = self.safe_find_element_text(By.ID, 'lblAvailableSeats')
            available = int(available) if available and available.isdigit() else None
            
            return total, current_enrolled, reserved, available
            
        except Exception as e:
            return None, None, None, None

    def load_existing_data(self, output_path='script_input/raw_data.xlsx'):
        """Load existing data from Excel file if it exists"""
        try:
            if os.path.exists(output_path):
                existing_standalone = pd.read_excel(output_path, sheet_name='standalone')
                existing_multiple = pd.read_excel(output_path, sheet_name='multiple')
                
                # Handle case where errors sheet might not exist
                try:
                    existing_errors = pd.read_excel(output_path, sheet_name='errors')
                except:
                    existing_errors = pd.DataFrame()
                
                return existing_standalone, existing_multiple, existing_errors
            else:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    def check_record_exists(self, existing_df, new_record):
        """Check if a record already exists based on key fields"""
        if existing_df.empty:
            return False
        
        # Define key fields that make a record unique
        key_fields = ['acad_term_id', 'course_code', 'section', 'bidding_window']
        
        # Check if all key fields exist in both dataframes
        for field in key_fields:
            if field not in existing_df.columns:
                return False
            if new_record.get(field) is None:
                return False
        
        # Create a mask to check for matching records
        mask = True
        for field in key_fields:
            mask = mask & (existing_df[field] == new_record[field])
        
        return mask.any()
    
    def get_current_academic_term(self):
        """
        Determines the current academic term (e.g., 2025-26_T1) by mapping the
        current date to the academic calendar terms.

        This logic is based on the SMU Academic Calendar for 2025-26, where bidding
        for a term can start before the official term commencement date. The function
        approximates term boundaries based on the calendar.
        """
        now = datetime.now()
        month = now.month

        # Determine the starting year of the academic year.
        # The academic year 'YYYY-(YY+1)' is assumed to start with T1 bidding in July of 'YYYY'.
        # So, from July onwards, we are in the 'YYYY' to 'YYYY+1' academic year.
        # Before July, we are in the latter half of the 'YYYY-1' to 'YYYY' academic year.
        if month >= 7:
            acad_year_start = now.year
        else:
            acad_year_start = now.year - 1
        
        acad_year_end_short = (acad_year_start + 1) % 100
        
        # Define the approximate start dates for each term's activities (including bidding)
        # for the determined academic year. These dates are based on the 2025-26 calendar.
        # 
        t1_start = datetime(acad_year_start, 7, 1) # Bidding for T1 starts in July
        t2_start = datetime(acad_year_start + 1, 1, 1) # T2 starts in January
        t3a_start = datetime(acad_year_start + 1, 5, 11) # T3A starts May 11
        t3b_start = datetime(acad_year_start + 1, 6, 29) # T3B starts June 29
        
        # Determine the term by checking the current date against the start dates
        # in reverse chronological order.
        term_code = None
        if now >= t3b_start:
            term_code = 'T3B'
        elif now >= t3a_start:
            term_code = 'T3A'
        elif now >= t2_start:
            term_code = 'T2'
        elif now >= t1_start:
            term_code = 'T1'

        if term_code:
            return f"{acad_year_start}-{acad_year_end_short:02d}_{term_code}"
        else:
            # This is a fallback for dates that might fall outside the defined ranges,
            # though the logic should cover all dates within a valid academic year.
            print(f"Warning: Could not determine the academic term for the date {now}. Review term start dates.")
            return None

    def find_latest_round_folder(self, term_path):
        """Find the latest round folder in the academic term directory"""
        try:
            # Get all subdirectories
            subdirs = [d for d in os.listdir(term_path) if os.path.isdir(os.path.join(term_path, d))]
            
            # Filter for round folders (should contain R and W)
            round_folders = [d for d in subdirs if 'R' in d and 'W' in d]
            
            if not round_folders:
                return None
            
            # Sort by folder name (this should work for the naming convention)
            # R1W1, R1AW1, R1AW2, etc.
            round_folders.sort(key=lambda x: (
                int(x.split('R')[1].split('W')[0].replace('A', '').replace('B', '').replace('C', '').replace('F', '')),
                x.count('A') + x.count('B') * 2 + x.count('C') * 3 + x.count('F') * 4,
                int(x.split('W')[1])
            ))
            
            return round_folders[-1]  # Return the latest one
            
        except Exception as e:
            print(f"Error finding latest round folder: {e}")
            return None

    def extract_bidding_window_from_folder(self, folder_name):
        """Extract bidding window from folder name (e.g., 2025-26_T1_R1W1 -> Round 1 Window 1)"""
        try:
            # Extract the round and window part (e.g., R1W1, R1AW2, etc.)
            round_part = folder_name.split('_')[-1]  # Get the last part after underscore
            
            # Map folder codes to full names
            folder_to_window = {
                'R1W1': 'Round 1 Window 1',
                'R1AW1': 'Round 1A Window 1',
                'R1AW2': 'Round 1A Window 2',
                'R1AW3': 'Round 1A Window 3',
                'R1BW1': 'Round 1B Window 1',
                'R1BW2': 'Round 1B Window 2',
                'R1CW1': 'Incoming Exchange Rnd 1C Win 1',
                'R1CW2': 'Incoming Exchange Rnd 1C Win 2',
                'R1CW3': 'Incoming Exchange Rnd 1C Win 3',
                'R1FW1': 'Incoming Freshmen Rnd 1 Win 1',
                'R1FW2': 'Incoming Freshmen Rnd 1 Win 2',
                'R1FW3': 'Incoming Freshmen Rnd 1 Win 3',
                'R1FW4': 'Incoming Freshmen Rnd 1 Win 4',
                'R2W1': 'Round 2 Window 1',
                'R2W2': 'Round 2 Window 2',
                'R2W3': 'Round 2 Window 3',
                'R2AW1': 'Round 2A Window 1',
                'R2AW2': 'Round 2A Window 2',
                'R2AW3': 'Round 2A Window 3',
            }
            
            return folder_to_window.get(round_part, round_part)
            
        except Exception as e:
            print(f"Error extracting bidding window from folder: {e}")
            return folder_name
        
if __name__ == "__main__":
    extractor = HTMLDataExtractor()

    # Use catch-up mode by default - processes ALL windows and fills gaps
    # To process only the latest window, use extractor.run() instead
    success = extractor.run_with_catchup(output_path='script_input/raw_data.xlsx')

    if not success:
        sys.exit(1) # Exit with error