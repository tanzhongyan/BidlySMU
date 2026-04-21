"""
PipelineCoordinator - orchestrates pipeline execution.
Sequentially calls processors and collects results as DTOs.
"""
import os
import csv
import pandas as pd
import pickle
from typing import Dict, List

from src.logging.logger import get_logger
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.processors.professor_processor import ProfessorProcessor
from src.pipeline.processors.bid_window_processor import BidWindowProcessor
from src.pipeline.dtos.course_dto import CourseDTO
from src.pipeline.dtos.professor_dto import ProfessorDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO

# Sheet name constants
SHEET_STANDALONE = 'standalone'
SHEET_MULTIPLE = 'multiple'


class PipelineCoordinator:
    """Coordinates pipeline execution with pure function processors."""

    def __init__(self, config):
        self.config = config
        self._logger = get_logger(__name__)
        self.results = {}

        os.makedirs(self.config.output_base, exist_ok=True)
        os.makedirs(self.config.verify_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)

        self.raw_data = None
        self.db_cache = {}
        self._load_caches()

    def _load_caches(self):
        """Load caches from db_cache directory."""
        cache_files = {
            'acad_term': 'acad_term_cache.pkl',
            'courses': 'course_cache.pkl',
            'professors': 'professor_cache.pkl',
            'faculties': 'faculty_cache.pkl',
            'bid_window': 'bid_window_cache.pkl',
        }

        for cache_name, filename in cache_files.items():
            filepath = os.path.join(self.config.cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.db_cache[cache_name] = pickle.load(f)
                self._logger.info(f"Loaded {cache_name} cache: {len(self.db_cache[cache_name])} entries")
            else:
                self.db_cache[cache_name] = {}
                self._logger.info(f"Initialized empty {cache_name} cache")

    def load_raw_data(self):
        """Load raw data from Excel file."""
        input_file = self.config.input_file
        self._logger.info(f"📂 Loading raw data from {input_file}")

        standalone_df = pd.read_excel(input_file, sheet_name=SHEET_STANDALONE)
        multiple_df = pd.read_excel(input_file, sheet_name=SHEET_MULTIPLE)

        self.raw_data = {
            SHEET_STANDALONE: standalone_df,
            SHEET_MULTIPLE: multiple_df
        }
        self._logger.info(f"✅ Loaded {len(standalone_df)} standalone and {len(multiple_df)} multiple records")

    def run(self):
        """Run the pipeline."""
        self._logger.info("🚀 Starting PipelineCoordinator")

        # Load raw data
        self.load_raw_data()

        # Process academic terms
        acad_term_processor = AcadTermProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            acad_term_cache=self.db_cache.get('acad_term', {}),
            logger=self._logger
        )
        acad_terms_new, acad_terms_updated = acad_term_processor.process()
        self.results['acad_terms'] = {'new': acad_terms_new, 'updated': acad_terms_updated}
        self.results['acad_term_lookup'] = self._build_lookup('acad_terms', 'id')
        self._logger.info(f"✅ Processed {len(acad_terms_new)} academic terms")

        # Process courses
        course_processor = CourseProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            courses_cache=self.db_cache.get('courses', {}),
            faculties_cache=self.db_cache.get('faculties', {}),
            logger=self._logger
        )
        courses_new, courses_updated = course_processor.process()
        self.results['courses'] = {'new': courses_new, 'updated': courses_updated}
        self.results['course_lookup'] = self._build_lookup('courses', 'code')
        self._logger.info(f"✅ Processed courses: {len(courses_new)} new, {len(courses_updated)} updated")

        # Process professors
        professor_processor = ProfessorProcessor(
            raw_data=self.raw_data[SHEET_MULTIPLE],  # Professor names from multiple sheet
            professors_cache=self.db_cache.get('professors', {}),
            logger=self._logger
        )
        professors_new, professors_updated = professor_processor.process()
        self.results['professors'] = {'new': professors_new, 'updated': professors_updated}

        # Build professor lookup for fact tables
        self.results['professor_lookup'] = self._build_lookup('professors', 'id')

        self._logger.info(f"✅ Processed professors: {len(professors_new)} new, {len(professors_updated)} updated")

        # Process bid windows
        bid_window_processor = BidWindowProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            bid_window_cache=self.db_cache.get('bid_window', {}),
            logger=self._logger
        )
        bid_windows_new, bid_windows_updated = bid_window_processor.process()
        self.results['bid_windows'] = {'new': bid_windows_new, 'updated': bid_windows_updated}

        # Build composite lookup for fact tables: (acad_term_id, round, window) -> BidWindowDTO
        self.results['bid_window_lookup'] = self._build_composite_lookup(
            'bid_windows',
            ['acad_term_id', 'round', 'window']
        )
        self._logger.info(f"✅ Processed bid_windows: {len(bid_windows_new)} new, {len(bid_windows_updated)} updated")

        self._logger.info("🚀 Pipeline completed")
        return self.results

    def _build_lookup(self, dimension: str, key_field: str) -> dict:
        """Build {key: DTO} lookup from stored dimension table results.

        Generic function to build lookups for any dimension table.
        Combines 'new' and 'updated' DTOs into single lookup dict.

        Args:
            dimension: The dimension name in self.results (e.g., 'courses', 'acad_terms')
            key_field: The DTO attribute to use as key (e.g., 'code', 'id')

        Returns:
            Dict mapping key_field value to DTO
        """
        lookup = {}
        data = self.results.get(dimension, {})
        for dto in data.get('new', []):
            lookup[getattr(dto, key_field)] = dto
        for dto in data.get('updated', []):
            lookup[getattr(dto, key_field)] = dto
        return lookup

    def _build_composite_lookup(self, dimension: str, key_fields: List[str]) -> dict:
        """Build lookup using multiple fields as key.

        Args:
            dimension: The dimension name in self.results
            key_fields: List of DTO attribute names to use as composite key

        Returns:
            Dict mapping tuple of key_field values to DTO
        """
        lookup = {}
        data = self.results.get(dimension, {})
        for dto in data.get('new', []) + data.get('updated', []):
            key = tuple(getattr(dto, f) for f in key_fields)
            lookup[key] = dto
        return lookup

    def _write_csv(self, filename: str, dtos: list, log_message: str):
        """Helper method to write DTOs to CSV file."""
        if not dtos:
            return
        output_file = os.path.join(self.config.output_base, filename)
        headers = list(dtos[0].COLUMNS.values())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for dto in dtos:
                writer.writerow(dto.to_csv_row())
        self._logger.info(log_message.format(filename=filename, count=len(dtos)))

    def save_csv(self):
        """Save results to CSV files."""
        # Save academic terms
        if 'acad_terms' in self.results and self.results['acad_terms']:
            terms = self.results['acad_terms']
            self._write_csv('new_acad_terms.csv', terms, f"✅ Saved acad_terms to {{filename}}")

        # Save courses
        if 'courses' in self.results:
            courses = self.results['courses']
            self._write_csv('new_courses.csv', courses['new'],
                            f"✅ Saved {{count}} new courses to {{filename}}")
            self._write_csv('update_courses.csv', courses['updated'],
                            f"✅ Saved {{count}} updated courses to {{filename}}")

        # Save professors
        if 'professors' in self.results:
            professors = self.results['professors']
            self._write_csv('new_professors.csv', professors['new'],
                            f"✅ Saved {{count}} new professors to {{filename}}")
            self._write_csv('update_professors.csv', professors['updated'],
                            f"✅ Saved {{count}} updated professors to {{filename}}")