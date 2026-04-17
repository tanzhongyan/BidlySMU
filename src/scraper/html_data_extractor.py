"""
HTML Data Extractor - scrapes class details from saved HTML files.

Usage:
    extractor = HTMLDataExtractor()
    with extractor:
        result = extractor.run()
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import os
import re

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By

from src.base.base_scraper import BaseScraper
from src.driver.driver_factory import ChromeDriverFactory
from src.parser.excel_writer import ExcelWriter
from src.parser.encoding_handler import EncodingHandler
from src.logging.logger import get_logger


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""
    files_processed: int = 0
    files_successful: int = 0
    errors: List[dict] = field(default_factory=list)
    standalone_records: List[dict] = field(default_factory=list)
    multiple_records: List[dict] = field(default_factory=list)


class HTMLDataExtractor(BaseScraper):
    """
    Extracts raw data from scraped HTML files and saves to Excel format.

    Uses Selenium for parsing HTML content with proper encoding handling.

    Usage:
        extractor = HTMLDataExtractor()
        with extractor:
            result = extractor.run(output_path='script_input/raw_data.xlsx')
    """

    def __init__(
        self,
        driver: Optional[WebDriver] = None,
        logger=None,
    ):
        self._standalone_data: List[dict] = []
        self._multiple_data: List[dict] = []
        self._errors: List[dict] = []
        super().__init__(driver=driver, logger=logger)
        self._encoding_handler = EncodingHandler()

    # ==================== BaseScraper Implementation ====================

    def scrape(self, **kwargs) -> ExtractionResult:
        """Run the complete extraction process."""
        return self.run()

    # ==================== Public Methods ====================

    def run(self, output_path: str = 'script_input/raw_data.xlsx') -> bool:
        """
        Run the complete extraction process.

        Args:
            output_path: Path to output Excel file

        Returns:
            True on success, False on failure
        """
        self._logger.info("Starting HTML data extraction...")

        # Reset data containers
        self._standalone_data = []
        self._multiple_data = []
        self._errors = []

        # Set up Selenium driver if not already connected
        if self._driver is None:
            factory = ChromeDriverFactory(headless=True)
            self._driver = factory.create()

        try:
            self._process_all_files()
            self._save_to_excel(output_path)
            self._logger.info("HTML data extraction completed!")
            return True

        except Exception as e:
            self._logger.error(f"An error occurred during HTML data extraction: {e}")
            return False

        finally:
            if self._driver:
                self._driver.quit()
                self._driver = None

    def process_html_file(self, filepath: str) -> bool:
        """
        Process a single HTML file and extract all data.

        Args:
            filepath: Path to HTML file

        Returns:
            True if successful, False otherwise
        """
        try:
            html_file = Path(filepath).resolve()
            file_url = html_file.as_uri()
            self._driver.get(file_url)

            record_key = os.path.basename(filepath)

            # Extract basic information
            class_header_text = self._safe_find_element_text(By.ID, 'lblClassInfoHeader')
            if not class_header_text:
                self._errors.append({
                    'filepath': filepath,
                    'error': 'Missing class header',
                    'type': 'parse_error'
                })
                return False

            course_code, section = self._parse_course_and_section(class_header_text)

            # Extract academic term
            term_text = self._safe_find_element_text(By.ID, 'lblClassInfoSubHeader')
            acad_year_start, acad_year_end, term, acad_term_id = self._parse_acad_term(term_text, filepath)

            # Extract course information
            course_name = self._safe_find_element_text(By.ID, 'lblClassSection')
            course_description = self._safe_find_element_text(By.ID, 'lblCourseDescription')
            credit_units = self._parse_credit_units(self._safe_find_element_text(By.ID, 'lblUnits'))
            course_areas = self._extract_course_areas_list()
            enrolment_requirements = self._safe_find_element_text(By.ID, 'lblEnrolmentRequirements')
            grading_basis = self._parse_grading_basis(self._safe_find_element_text(By.ID, 'lblGradingBasis'))
            course_outline_url = self._extract_course_outline_url()

            # Extract dates
            period_text = self._safe_find_element_text(By.ID, 'lblDates')
            start_dt, end_dt = self._parse_date_range(period_text)

            # Extract BOSS IDs
            acad_term_boss_id, class_boss_id = self._extract_boss_ids_from_filepath(filepath)

            # Extract bidding information
            total, current_enrolled, reserved, available = self._extract_bidding_info()

            extraction_date = datetime.now()
            bidding_window = self._determine_bidding_window_from_filepath(filepath)

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

            self._standalone_data.append(standalone_record)

            # Extract meeting information
            self._extract_meeting_information(record_key)

            return True

        except Exception as e:
            self._errors.append({
                'filepath': filepath,
                'error': str(e),
                'type': 'processing_error'
            })
            return False

    # ==================== Internal Methods ====================

    def _process_all_files(self) -> None:
        """Process only files from the latest round folder that haven't been processed yet."""
        base_dir = 'script_input/classTimingsFull'

        current_term = self._get_current_academic_term()
        if not current_term:
            self._logger.warning("Could not determine current academic term")
            return

        term_path = os.path.join(base_dir, current_term)
        if not os.path.exists(term_path):
            self._logger.warning(f"Academic term folder not found: {term_path}")
            return

        latest_round_folder = self._find_latest_round_folder(term_path)
        if not latest_round_folder:
            self._logger.warning(f"No round folders found in {term_path}")
            return

        latest_round_path = os.path.join(term_path, latest_round_folder)
        bidding_window = self._extract_bidding_window_from_folder(latest_round_folder)

        self._logger.info(f"Processing latest round: {latest_round_folder}")

        # Get all HTML files
        html_files = [
            os.path.join(latest_round_path, f)
            for f in os.listdir(latest_round_path)
            if f.endswith('.html')
        ]

        if not html_files:
            self._logger.warning(f"No HTML files found in {latest_round_path}")
            return

        # Load existing data
        excel_writer = ExcelWriter()
        existing_standalone, _, _ = excel_writer.load_existing()

        # Filter files already processed
        files_to_process = self._filter_processed_files(
            html_files, existing_standalone, bidding_window
        )

        if not files_to_process:
            self._logger.info(f"All files from {latest_round_folder} have already been processed")
            return

        self._logger.info(f"Processing {len(files_to_process)} new files")

        # Process files
        processed = 0
        successful = 0

        for filepath in files_to_process:
            if os.path.exists(filepath):
                if self.process_html_file(filepath):
                    successful += 1
                processed += 1

                if processed % 100 == 0:
                    self._logger.info(f"Processed {processed}/{len(files_to_process)} files")

        self._logger.info(f"Processing complete: {successful}/{processed} files successful")

    def _filter_processed_files(
        self,
        html_files: List[str],
        existing: 'pd.DataFrame',
        bidding_window: str,
    ) -> List[str]:
        """Filter files that haven't been processed for this bidding window."""
        if existing.empty:
            return html_files

        files_to_process = []
        for filepath in html_files:
            acad_term_boss_id, class_boss_id = self._extract_boss_ids_from_filepath(filepath)

            mask = (
                (existing['acad_term_boss_id'] == acad_term_boss_id) &
                (existing['class_boss_id'] == class_boss_id) &
                (existing['bidding_window'] == bidding_window)
            )

            if not mask.any():
                files_to_process.append(filepath)

        return files_to_process

    def _save_to_excel(self, output_path: str) -> None:
        """Save extracted data to Excel file."""
        excel_writer = ExcelWriter(output_path)
        existing_standalone, existing_multiple, existing_errors = excel_writer.load_existing()

        excel_writer.save(
            self._standalone_data,
            self._multiple_data,
            self._errors,
            existing_standalone,
            existing_multiple,
            existing_errors,
        )

        self._logger.info(f"Data saved to {output_path}")
        self._logger.info(f"New standalone records: {len(self._standalone_data)}")

    def _safe_find_element_text(self, by: By, value: str) -> Optional[str]:
        """Safely find element and return its text with encoding handling."""
        try:
            element = self._driver.find_element(by, value)
            if element:
                raw_text = element.text.strip()
                return self._encoding_handler.clean(raw_text)
            return None
        except Exception:
            return None

    def _safe_find_element_attribute(
        self,
        by: By,
        value: str,
        attribute: str,
    ) -> Optional[str]:
        """Safely find element and return its attribute."""
        try:
            element = self._driver.find_element(by, value)
            if element:
                raw_attr = element.get_attribute(attribute)
                return self._encoding_handler.clean(raw_attr) if raw_attr else None
            return None
        except Exception:
            return None

    def _parse_course_and_section(self, header_text: str) -> tuple:
        """Parse course code and section from header text."""
        if not header_text:
            return None, None

        clean_text = self._encoding_handler.clean(header_text)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text.strip())

        patterns = [
            r'([A-Z0-9_-]+)\s+—\s+(.+)',
            r'([A-Z0-9_-]+)\s+-\s+(.+)',
            r'([A-Z]+)\s+(\d+[A-Z0-9_]*)\s+—\s+(.+)',
            r'([A-Z]+)\s+(\d+[A-Z0-9_]*)\s+-\s+(.+)',
            r'([A-Z0-9_\s-]+?)\s+—\s+([^—]+)',
            r'([A-Z0-9_\s-]+?)\s+-\s+([^-]+)',
        ]

        for pattern in patterns:
            match = re.match(pattern, clean_text)
            if match:
                if len(match.groups()) == 2:
                    course_section = match.group(1).strip()
                    section_name = match.group(2).strip()

                    section_match = re.search(r'^(.+?)\s+(SG\d+|G\d+|\d+)$', course_section)
                    if section_match:
                        course_code = section_match.group(1)
                        section = section_match.group(2)
                    else:
                        course_code = course_section
                        section_extract = re.search(r'(SG\d+|G\d+|\d+)', section_name)
                        section = section_extract.group(1) if section_extract else None
                else:
                    course_code = f"{match.group(1)}{match.group(2)}"
                    section_name = match.group(3).strip()
                    section_extract = re.search(r'(SG\d+|G\d+|\d+)', section_name)
                    section = section_extract.group(1) if section_extract else None

                return course_code.strip() if course_code else None, section

        return None, None

    def _parse_acad_term(self, term_text: str, filepath: str = None) -> tuple:
        """Parse academic term text and return structured data."""
        if term_text:
            term_text = self._encoding_handler.clean(term_text)

        pattern = r'(\d{4})-(\d{2})\s+(.*)'
        match = re.search(pattern, term_text) if term_text else None

        if not match:
            return None, None, None, None

        start_year = int(match.group(1))
        end_year_short = int(match.group(2))
        term_desc = match.group(3).lower()

        if end_year_short < 50:
            end_year = 2000 + end_year_short
        else:
            end_year = 1900 + end_year_short

        term_code = self._determine_term_code(term_desc)

        if not term_code or term_code == 'T3':
            folder_term = self._extract_term_from_folder_path(filepath) if filepath else None
            if folder_term:
                if term_code == 'T3' and folder_term in ['T3A', 'T3B']:
                    term_code = folder_term
                elif not term_code:
                    term_code = folder_term

        if not term_code:
            return start_year, end_year, None, None

        acad_term_id = f"AY{start_year}{end_year_short:02d}{term_code}"

        return start_year, end_year, term_code, acad_term_id

    def _determine_term_code(self, term_desc: str) -> Optional[str]:
        """Determine term code from description."""
        if 'term 1' in term_desc or 'session 1' in term_desc or 'august term' in term_desc:
            return 'T1'
        elif 'term 2' in term_desc or 'session 2' in term_desc or 'january term' in term_desc:
            return 'T2'
        elif 'term 3a' in term_desc:
            return 'T3A'
        elif 'term 3b' in term_desc:
            return 'T3B'
        elif 'term 3' in term_desc:
            return 'T3'
        return None

    def _extract_term_from_folder_path(self, filepath: str) -> Optional[str]:
        """Extract term from folder path as fallback."""
        try:
            folder_name = os.path.basename(os.path.dirname(filepath))
            term_match = re.search(r'(\d{4}-\d{2})_T(\w+)', folder_name)
            if term_match:
                return f"T{term_match.group(2)}"
            term_fallback = re.search(r'T(\w+)', folder_name)
            if term_fallback:
                return f"T{term_fallback.group(1)}"
            return None
        except Exception:
            return None

    def _parse_date_range(self, date_text: str) -> tuple:
        """Parse date range text and return start and end timestamps."""
        if not date_text:
            return None, None

        pattern = r'(\d{1,2}-\w{3}-\d{4})\s+to\s+(\d{1,2}-\w{3}-\d{4})'
        match = re.search(pattern, date_text)

        if not match:
            return None, None

        start_date = self._convert_date_to_timestamp(match.group(1))
        end_date = self._convert_date_to_timestamp(match.group(2))

        return start_date, end_date

    def _convert_date_to_timestamp(self, date_str: str) -> Optional[str]:
        """Convert DD-Mmm-YYYY to database timestamp format."""
        try:
            date_obj = datetime.strptime(date_str, '%d-%b-%Y')
            return date_obj.strftime('%Y-%m-%d 00:00:00.000 +0800')
        except Exception:
            return None

    def _parse_credit_units(self, text: str) -> Optional[float]:
        """Parse credit units from text."""
        try:
            return float(text) if text else None
        except (ValueError, TypeError):
            return None

    def _parse_grading_basis(self, text: str) -> Optional[str]:
        """Parse grading basis from text."""
        if not text:
            return None
        text_lower = text.lower()
        if text_lower == 'graded':
            return 'Graded'
        elif text_lower in ['pass/fail', 'pass fail']:
            return 'Pass/Fail'
        return 'NA'

    def _extract_course_areas_list(self) -> Optional[str]:
        """Extract course areas with encoding fixes."""
        try:
            course_areas_element = self._driver.find_element(By.ID, 'lblCourseAreas')
            if not course_areas_element:
                return None

            course_areas_html = course_areas_element.get_attribute('innerHTML')
            if course_areas_html:
                course_areas_html = self._encoding_handler.clean(course_areas_html)
                areas_list = re.findall(r'<li[^>]*>([^<]+)</li>', course_areas_html)
                if areas_list:
                    cleaned_areas = [self._encoding_handler.clean(area.strip()) for area in areas_list]
                    return ', '.join(cleaned_areas)

            return self._encoding_handler.clean(course_areas_element.text.strip())
        except Exception:
            return None

    def _extract_course_outline_url(self) -> Optional[str]:
        """Extract course outline URL from HTML."""
        try:
            onclick_attr = self._safe_find_element_attribute(By.ID, 'imgCourseOutline', 'onclick')
            if onclick_attr:
                url_match = re.search(r"window\.open\('([^']+)'", onclick_attr)
                if url_match:
                    return url_match.group(1)
        except Exception:
            pass
        return None

    def _extract_boss_ids_from_filepath(self, filepath: str) -> tuple:
        """Extract BOSS IDs from filepath."""
        try:
            filename = os.path.basename(filepath)
            acad_term_match = re.search(r'SelectedAcadTerm=(\d+)', filename)
            class_match = re.search(r'SelectedClassNumber=(\d+)', filename)

            acad_term_boss_id = int(acad_term_match.group(1)) if acad_term_match else None
            class_boss_id = int(class_match.group(1)) if class_match else None

            return acad_term_boss_id, class_boss_id
        except Exception:
            return None, None

    def _extract_bidding_info(self) -> tuple:
        """Extract current bidding information from HTML elements."""
        try:
            total = self._safe_find_element_text(By.ID, 'lblClassCapacity')
            total = int(total) if total and total.isdigit() else None

            current_enrolled = self._safe_find_element_text(By.ID, 'lblEnrolmentTotal')
            current_enrolled = int(current_enrolled) if current_enrolled and current_enrolled.isdigit() else None

            reserved = self._safe_find_element_text(By.ID, 'lblReserved')
            reserved = int(reserved) if reserved and reserved.isdigit() else None

            available = self._safe_find_element_text(By.ID, 'lblAvailableSeats')
            available = int(available) if available and available.isdigit() else None

            return total, current_enrolled, reserved, available
        except Exception:
            return None, None, None, None

    def _extract_meeting_information(self, record_key: str) -> None:
        """Extract class timing and exam timing information."""
        try:
            meeting_table = self._driver.find_element(By.ID, 'RadGrid_MeetingInfo_ctl00')
            tbody = meeting_table.find_element(By.TAG_NAME, 'tbody')
            rows = tbody.find_elements(By.TAG_NAME, 'tr')

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) < 7:
                    continue

                meeting_type = cells[0].text.strip() or 'CLASS'
                start_date_text = cells[1].text.strip()
                end_date_text = cells[2].text.strip()
                day_of_week = cells[3].text.strip()
                start_time = cells[4].text.strip()
                end_time = cells[5].text.strip()
                venue = cells[6].text.strip() if len(cells) > 6 else ""
                professor_name = cells[7].text.strip() if len(cells) > 7 else ""

                if meeting_type == 'CLASS':
                    timing_record = {
                        'record_key': record_key,
                        'type': 'CLASS',
                        'start_date': self._convert_date_to_timestamp(start_date_text),
                        'end_date': self._convert_date_to_timestamp(end_date_text),
                        'day_of_week': day_of_week,
                        'start_time': start_time,
                        'end_time': end_time,
                        'venue': venue,
                        'professor_name': professor_name
                    }
                    self._multiple_data.append(timing_record)

                elif meeting_type == 'EXAM':
                    exam_record = {
                        'record_key': record_key,
                        'type': 'EXAM',
                        'date': self._convert_date_to_timestamp(end_date_text),
                        'day_of_week': day_of_week,
                        'start_time': start_time,
                        'end_time': end_time,
                        'venue': venue,
                        'professor_name': professor_name
                    }
                    self._multiple_data.append(exam_record)

        except Exception as e:
            self._errors.append({
                'record_key': record_key,
                'error': f'Error extracting meeting information: {str(e)}',
                'type': 'parse_error'
            })

    def _determine_bidding_window_from_filepath(self, filepath: str) -> Optional[str]:
        """Determine bidding window from the file path structure."""
        try:
            folder_name = os.path.basename(os.path.dirname(filepath))
            return self._extract_bidding_window_from_folder(folder_name)
        except Exception:
            return None

    def _extract_bidding_window_from_folder(self, folder_name: str) -> str:
        """Extract bidding window from folder name."""
        round_part = folder_name.split('_')[-1]

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
            'R2W4': 'Round 2 Window 4',
            'R2AW1': 'Round 2A Window 1',
            'R2AW2': 'Round 2A Window 2',
            'R2AW3': 'Round 2A Window 3',
        }

        return folder_to_window.get(round_part, round_part)

    def _get_current_academic_term(self) -> Optional[str]:
        """
        Determine which academic term folder to process.

        Currently hardcoded to use config.START_AY_TERM for consistency with
        the scraper (class_scraper.py) which also uses config to determine target term.

        TODO: Automate this logic based on BIDDING_SCHEDULES:
        - When in an active bidding window for term X, process term X
        - When between terms (prep phase), process the NEXT upcoming term
        - This allows scraping/processing to happen before a term starts

        For now, hardcoding to config ensures scraper and extractor stay in sync.
        """
        # TODO: Replace with automated logic:
        # from src.config import BIDDING_SCHEDULES
        # now = datetime.now()
        # for schedule_item in BIDDING_SCHEDULES:
        #     results_date, window_name, folder_suffix = schedule_item
        #     if now < results_date:
        #         # Extract term from folder_suffix (e.g., "2510" -> "2025-26_T1")
        #         return extract_term_from_suffix(folder_suffix)
        # return START_AY_TERM  # Fallback to config

        from src.config import START_AY_TERM
        return START_AY_TERM

    def _find_latest_round_folder(self, term_path: str) -> Optional[str]:
        """Find the latest round folder in the academic term directory."""
        try:
            subdirs = [d for d in os.listdir(term_path) if os.path.isdir(os.path.join(term_path, d))]
            round_folders = [d for d in subdirs if 'R' in d and 'W' in d]

            if not round_folders:
                return None

            round_folders.sort(key=lambda x: (
                int(x.split('R')[1].split('W')[0].replace('A', '').replace('B', '').replace('C', '').replace('F', '')),
                x.count('A') + x.count('B') * 2 + x.count('C') * 3 + x.count('F') * 4,
                int(x.split('W')[1])
            ))

            return round_folders[-1]
        except Exception:
            return None


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    extractor = HTMLDataExtractor()
    success = extractor.run(output_path='script_input/raw_data.xlsx')
    sys.exit(0 if success else 1)
