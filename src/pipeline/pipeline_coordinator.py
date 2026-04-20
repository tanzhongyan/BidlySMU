"""
PipelineCoordinator - orchestrates pipeline execution.
Sequentially calls processors and collects results as DTOs.
"""
import os
import csv
import pandas as pd
import pickle

from src.logging.logger import get_logger
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.dtos.course_dto import CourseDTO


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

        standalone_df = pd.read_excel(input_file, sheet_name='standalone')
        multiple_df = pd.read_excel(input_file, sheet_name='multiple')

        self.raw_data = {
            'standalone': standalone_df,
            'multiple': multiple_df
        }
        self._logger.info(f"✅ Loaded {len(standalone_df)} standalone and {len(multiple_df)} multiple records")

    def run(self):
        """Run the pipeline."""
        self._logger.info("🚀 Starting PipelineCoordinator")

        # Load raw data
        self.load_raw_data()

        # Process academic terms
        acad_term_processor = AcadTermProcessor(
            raw_data=self.raw_data['standalone'],
            acad_term_cache=self.db_cache.get('acad_term', {})
        )
        acad_terms_new, acad_terms_updated = acad_term_processor.process()
        self.results['acad_terms'] = {'new': acad_terms_new, 'updated': acad_terms_updated}
        self._logger.info(f"✅ Processed {len(acad_terms_new)} academic terms")

        # Process courses
        course_processor = CourseProcessor(
            raw_data=self.raw_data['standalone'],
            courses_cache=self.db_cache.get('courses', {}),
            faculties_cache=self.db_cache.get('faculties', {})
        )
        courses_new, courses_updated = course_processor.process()
        self.results['courses'] = {'new': courses_new, 'updated': courses_updated}

        # Build lookups for fact tables (in-memory, no pickle!)
        self.results['acad_term_lookup'] = self._build_lookup('acad_terms', 'id')
        self.results['course_lookup'] = self._build_lookup('courses', 'code')

        self._logger.info(f"✅ Processed courses: {len(courses_new)} new, {len(courses_updated)} updated")
        self._logger.info(f"✅ Built course_lookup with {len(self.results['course_lookup'])} entries")

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

    def save_csv(self):
        """Save results to CSV files."""
        # Save academic terms
        if 'acad_terms' in self.results and self.results['acad_terms']:
            output_file = os.path.join(self.config.output_base, 'new_acad_terms.csv')
            terms = self.results['acad_terms']
            if terms:
                headers = list(terms[0].COLUMNS.values())
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for term in terms:
                        writer.writerow(term.to_csv_row())
                self._logger.info(f"✅ Saved acad_terms to {output_file}")

        # Save new courses to script_output/ (NOT script_output/verify/)
        if 'courses' in self.results:
            new_courses = self.results['courses']['new']
            if new_courses:
                output_file = os.path.join(self.config.output_base, 'new_courses.csv')
                headers = list(new_courses[0].COLUMNS.values())
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for dto in new_courses:
                        writer.writerow(dto.to_csv_row())
                self._logger.info(f"✅ Saved {len(new_courses)} new courses to {output_file}")

            updated_courses = self.results['courses']['updated']
            if updated_courses:
                output_file = os.path.join(self.config.output_base, 'update_courses.csv')
                headers = list(updated_courses[0].COLUMNS.values())
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for dto in updated_courses:
                        writer.writerow(dto.to_csv_row())
                self._logger.info(f"✅ Saved {len(updated_courses)} updated courses to {output_file}")