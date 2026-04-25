"""
PipelineCoordinator - orchestrates pipeline execution.
Sequentially calls processors and collects results as DTOs.
"""
import os
import csv
import pandas as pd
import pickle
from typing import List

from src.logging.logger import get_logger
from src.db.database_helper import DatabaseHelper
from src.db.adapters import Psycopg2Adapter
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processors.bid_prediction_processor import BidPredictionProcessor
from src.pipeline.processors.bid_result_processor import BidResultProcessor
from src.pipeline.processors.bid_window_processor import BidWindowProcessor
from src.pipeline.processors.class_availability_processor import ClassAvailabilityProcessor
from src.pipeline.processors.class_exam_timing_processor import ClassExamTimingProcessor
from src.pipeline.processors.class_processor import ClassProcessor
from src.pipeline.processors.class_timing_processor import ClassTimingProcessor
from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.processors.professor_processor import ProfessorProcessor
from src.pipeline.processors.safety_factor_processor import SafetyFactorProcessor

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
        self._db_connection = None
        self._init_database()
        self._load_caches()

    def _init_database(self):
        """Initialize database connection."""
        db_adapter = Psycopg2Adapter(self.config.db_config, self._logger)
        self._db_connection = DatabaseHelper.create_connection(db_adapter, self._logger)

    def _load_caches(self):
        """Load caches from db_cache directory. Downloads from DB if cache doesn't exist."""
        cache_files = {
            'acad_term': 'acad_term_cache.pkl',
            'courses': 'courses_cache.pkl',
            'professors': 'professors_cache.pkl',
            'faculties': 'faculties_cache.pkl',
            'bid_window': 'bid_window_cache.pkl',
            'classes': 'classes_cache.pkl',
            'class_timing': 'class_timing_cache.pkl',
            'class_exam_timing': 'class_exam_timing_cache.pkl',
        }

        tables_to_download = []
        for cache_name, filename in cache_files.items():
            filepath = os.path.join(self.config.cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    df = pickle.load(f)
                    self.db_cache[cache_name] = df
                self._logger.info(f"Loaded {cache_name} cache: {len(self.db_cache[cache_name])} entries")
            else:
                tables_to_download.append(cache_name)

        if tables_to_download:
            if not self._db_connection:
                raise RuntimeError(
                    f"Cache miss for {tables_to_download} but no database connection available. "
                    "Please ensure the database is accessible or restore db_cache/ from a previous run."
                )
            self._logger.info(f"Cache miss for {tables_to_download} - downloading from database...")
            DatabaseHelper.download_cache(
                self._db_connection,
                self.config.cache_dir,
                tables_to_download,
                self._logger
            )
            for cache_name in tables_to_download:
                filename = cache_files[cache_name]
                filepath = os.path.join(self.config.cache_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        df = pickle.load(f)
                        self.db_cache[cache_name] = df
                    self._logger.info(f"Loaded {cache_name} cache: {len(self.db_cache[cache_name])} entries")
                else:
                    raise RuntimeError(f"Failed to download {cache_name} cache from database")

        # Convert DataFrames to dicts for processors that expect dict-like access
        self._convert_caches_to_dicts()

    def _convert_caches_to_dicts(self):
        """Convert DataFrame caches to dict-of-dict format for processor compatibility.

        Processors iterate over caches with `for _, item in cache.items()` expecting
        (key, value) pairs, not (index, Series) pairs from DataFrame iteration.
        """
        # acad_term: {id: row_dict}
        if isinstance(self.db_cache.get('acad_term'), pd.DataFrame):
            df = self.db_cache['acad_term']
            if not df.empty and 'id' in df.columns:
                self.db_cache['acad_term'] = dict(zip(df['id'], df.to_dict('records')))
            else:
                self.db_cache['acad_term'] = {}
            self._logger.info(f"Converted acad_term cache to dict with {len(self.db_cache['acad_term'])} entries")

        # courses: {code: row_dict}
        if isinstance(self.db_cache.get('courses'), pd.DataFrame):
            df = self.db_cache['courses']
            if not df.empty and 'code' in df.columns:
                self.db_cache['courses'] = dict(zip(df['code'], df.to_dict('records')))
            else:
                self.db_cache['courses'] = {}
            self._logger.info(f"Converted courses cache to dict with {len(self.db_cache['courses'])} entries")

        # professors: {name_upper: row_dict}
        if isinstance(self.db_cache.get('professors'), pd.DataFrame):
            df = self.db_cache['professors']
            if not df.empty:
                # Build name -> row lookup (use 'name' column if available, else 'id')
                if 'name' in df.columns:
                    self.db_cache['professors'] = dict(zip(df['name'].str.upper(), df.to_dict('records')))
                else:
                    self.db_cache['professors'] = {}
            else:
                self.db_cache['professors'] = {}
            self._logger.info(f"Converted professors cache to dict with {len(self.db_cache['professors'])} entries")

        # faculties: {id: row_dict}
        if isinstance(self.db_cache.get('faculties'), pd.DataFrame):
            df = self.db_cache['faculties']
            if not df.empty and 'id' in df.columns:
                self.db_cache['faculties'] = dict(zip(df['id'], df.to_dict('records')))
            else:
                self.db_cache['faculties'] = {}
            self._logger.info(f"Converted faculties cache to dict with {len(self.db_cache['faculties'])} entries")

        # bid_window: {(acad_term_id, round, window): row_dict}
        if isinstance(self.db_cache.get('bid_window'), pd.DataFrame):
            df = self.db_cache['bid_window']
            if not df.empty and all(col in df.columns for col in ['acad_term_id', 'round', 'window']):
                self.db_cache['bid_window'] = {}
                for _, row in df.iterrows():
                    key = (row['acad_term_id'], str(row['round']), int(row['window']))
                    self.db_cache['bid_window'][key] = row.to_dict()
            else:
                self.db_cache['bid_window'] = {}
            self._logger.info(f"Converted bid_window cache to dict with {len(self.db_cache['bid_window'])} entries")

        # classes: {(acad_term_id, boss_id, professor_id): row_dict}
        if isinstance(self.db_cache.get('classes'), pd.DataFrame):
            df = self.db_cache['classes']
            if not df.empty and all(col in df.columns for col in ['acad_term_id', 'boss_id', 'professor_id']):
                self.db_cache['classes'] = {}
                for _, row in df.iterrows():
                    key = (row['acad_term_id'], row['boss_id'], row.get('professor_id'))
                    self.db_cache['classes'][key] = row.to_dict()
            else:
                self.db_cache['classes'] = {}
            self._logger.info(f"Converted classes cache to dict with {len(self.db_cache['classes'])} entries")

        # class_timing: set of (class_id, day_of_week, start_time, end_time, venue) keys
        if isinstance(self.db_cache.get('class_timing'), pd.DataFrame):
            df = self.db_cache['class_timing']
            if not df.empty and 'class_id' in df.columns:
                existing_timing_keys = set()
                for _, row in df.iterrows():
                    key = (
                        str(row['class_id']),
                        '' if pd.isna(row.get('day_of_week')) else str(row.get('day_of_week')),
                        '' if pd.isna(row.get('start_time')) else str(row.get('start_time')),
                        '' if pd.isna(row.get('end_time')) else str(row.get('end_time')),
                        '' if pd.isna(row.get('venue')) else str(row.get('venue'))
                    )
                    existing_timing_keys.add(key)
                self.db_cache['class_timing'] = existing_timing_keys
            else:
                self.db_cache['class_timing'] = set()
            self._logger.info(f"Converted class_timing cache to set with {len(self.db_cache['class_timing'])} entries")

        # class_exam_timing: set of class_ids that already have exam timings
        if isinstance(self.db_cache.get('class_exam_timing'), pd.DataFrame):
            df = self.db_cache['class_exam_timing']
            if not df.empty and 'class_id' in df.columns:
                existing_exam_class_ids = set(df['class_id'].astype(str).unique())
                self.db_cache['class_exam_timing'] = existing_exam_class_ids
            else:
                self.db_cache['class_exam_timing'] = set()
            self._logger.info(f"Converted class_exam_timing cache to set with {len(self.db_cache['class_exam_timing'])} entries")

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

        # ============================================================
        # Phase 1: Dim Tables
        # ============================================================

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
        self.results['professor_lookup'] = self._build_professor_name_lookup()
        self._logger.info(f"✅ Processed professors: {len(professors_new)} new, {len(professors_updated)} updated")

        # Process bid windows
        bid_window_processor = BidWindowProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            bid_window_cache=self.db_cache.get('bid_window', {}),
            logger=self._logger
        )
        bid_windows_new, bid_windows_updated = bid_window_processor.process()
        self.results['bid_windows'] = {'new': bid_windows_new, 'updated': bid_windows_updated}
        self.results['bid_window_lookup'] = self._build_composite_lookup(
            'bid_windows',
            ['acad_term_id', 'round', 'window']
        )
        self._logger.info(f"✅ Processed bid_windows: {len(bid_windows_new)} new, {len(bid_windows_updated)} updated")

        # ============================================================
        # Phase 2: Fact Tables
        # ============================================================

        # Load classes cache from db_cache (already converted to dict in _load_caches)
        existing_classes_cache = []
        if 'classes' in self.db_cache and isinstance(self.db_cache['classes'], dict):
            existing_classes_cache = list(self.db_cache['classes'].values())
        self._logger.info(f"Loaded {len(existing_classes_cache)} existing class records from cache")

        # Build multiple_lookup from raw_data multiple sheet
        self._multiple_lookup = self._build_multiple_lookup()
        multiple_lookup = self._multiple_lookup

        # Process classes
        # course_lookup must contain CourseDTO objects with .id attribute
        # _build_lookup combines new + updated courses from self.results
        course_lookup = self._build_lookup('courses', 'code')
        class_processor = ClassProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            multiple_lookup=multiple_lookup,
            course_lookup=course_lookup,
            professor_lookup=self.results['professor_lookup'],
            existing_classes_cache=existing_classes_cache,
            logger=self._logger
        )
        classes_new, classes_updated = class_processor.process()
        self.results['classes'] = {'new': classes_new, 'updated': classes_updated}
        self.results['class_lookup'] = self._build_class_lookup()
        self._logger.info(f"✅ Processed classes: {len(classes_new)} new, {len(classes_updated)} updated")

        # Get record_key -> [class_ids] mapping for timing processing
        record_key_to_class_ids = class_processor.get_record_key_to_class_ids_mapping()
        self._logger.info(f"📊 Built record_key -> class_ids mapping with {len(record_key_to_class_ids)} entries")

        # Process class timings
        existing_timing_keys = self.db_cache.get('class_timing', set())
        class_timing_processor = ClassTimingProcessor(
            raw_data=self.raw_data[SHEET_MULTIPLE],
            class_lookup=self.results['class_lookup'],
            record_key_to_class_ids=record_key_to_class_ids,
            existing_class_timing_keys=existing_timing_keys,
            logger=self._logger
        )
        new_class_timings = class_timing_processor.process()
        self.results['class_timings'] = new_class_timings
        self._logger.info(f"✅ Processed {len(new_class_timings)} class timings")

        # Process exam timings
        existing_exam_class_ids = self.db_cache.get('class_exam_timing', set())
        class_exam_timing_processor = ClassExamTimingProcessor(
            raw_data=self.raw_data[SHEET_MULTIPLE],
            class_lookup=self.results['class_lookup'],
            record_key_to_class_ids=record_key_to_class_ids,
            processed_exam_class_ids=existing_exam_class_ids,
            logger=self._logger
        )
        new_exam_timings = class_exam_timing_processor.process()
        self.results['class_exam_timings'] = new_exam_timings
        self._logger.info(f"✅ Processed {len(new_exam_timings)} exam timings")

        # Get bidding schedule for availability and bid results
        # Note: self.config.start_ay_term is already in dash format (START_AY_TERM from config)
        bidding_schedule = self.config.bidding_schedules.get(self.config.start_ay_term, [])

        # Process class availability (current window only)
        class_avail_processor = ClassAvailabilityProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            class_lookup=self.results['class_lookup'],
            bid_window_lookup=self.results['bid_window_lookup'],
            bidding_schedule=bidding_schedule,
            expected_acad_term_id=self.config.start_ay_term,
            logger=self._logger
        )
        new_class_avail = class_avail_processor.process()
        self.results['class_availabilities'] = new_class_avail
        self._logger.info(f"✅ Processed {len(new_class_avail)} class availability records")

        # Process bid results (previous + current window)
        overall_results_path = self._get_overall_results_path()
        bid_result_processor = BidResultProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            overall_results_path=overall_results_path,
            class_lookup=self.results['class_lookup'],
            bid_window_lookup=self.results['bid_window_lookup'],
            course_lookup=self.results['course_lookup'],
            bidding_schedule=bidding_schedule,
            expected_acad_term_id=self.config.start_ay_term,
            logger=self._logger
        )
        bid_results_new, bid_results_updated = bid_result_processor.process()
        self.results['bid_results'] = {'new': bid_results_new, 'updated': bid_results_updated}
        self._logger.info(f"✅ Processed bid_results: {len(bid_results_new)} new, {len(bid_results_updated)} updated")

        # ============================================================
        # Phase 3: Bid Predictions
        # ============================================================

        bid_prediction_processor = BidPredictionProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            class_lookup=self.results['class_lookup'],
            bid_window_lookup=self.results['bid_window_lookup'],
            multiple_lookup=self._multiple_lookup,
            bidding_schedule=bidding_schedule,
            expected_acad_term_id=self.config.start_ay_term,
            model_dir='models',
            logger=self._logger
        )
        predictions = bid_prediction_processor.process()
        self.results['bid_predictions'] = predictions
        self._logger.info(f"✅ Generated {len(predictions)} bid predictions")

        safety_factor_processor = SafetyFactorProcessor(
            expected_acad_term_id=self.config.start_ay_term,
            cache_dir=self.config.cache_dir,
            logger=self._logger
        )
        safety_factors = safety_factor_processor.process()
        if safety_factors:
            self.results['safety_factors'] = safety_factors
            self._logger.info(f"✅ Generated {len(safety_factors)} safety factor entries")

        # Save results to CSV and database
        self._logger.info("💾 Saving results...")
        self.save_csv()
        self.save_to_database()

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

    def _build_professor_name_lookup(self) -> dict:
        """Build {boss_name_upper: professor_id} lookup from professor DTOs AND DB cache.

        This is different from _build_lookup because ClassProcessor needs
        boss_name -> professor_id mapping, not id -> ProfessorDTO.

        IMPORTANT: Must include professors from DB cache because ClassProcessor
        only receives professor names from the raw data and needs to resolve them
        to IDs. The lookup must contain ALL known professors, not just new/updated ones.
        """
        lookup = {}

        # First, add all professors from DB cache (they are already in the database)
        # Note: professors_cache is a DataFrame where each ROW is a professor
        professors_cache = self.db_cache.get('professors', {})
        if isinstance(professors_cache, pd.DataFrame):
            # Iterate over DataFrame rows, not columns
            for _, row in professors_cache.iterrows():
                if row.get('name'):
                    lookup[row['name'].upper()] = row['id']
                    # Also add boss_aliases
                    boss_aliases = row.get('boss_aliases')
                    if boss_aliases:
                        # boss_aliases can be a Python list or a JSON string from DB
                        if isinstance(boss_aliases, list):
                            for alias in boss_aliases:
                                lookup[alias.upper()] = row['id']
                        elif isinstance(boss_aliases, str):
                            import json
                            try:
                                aliases_list = json.loads(boss_aliases)
                                for alias in aliases_list:
                                    lookup[alias.upper()] = row['id']
                            except:
                                pass
        elif isinstance(professors_cache, dict):
            # Legacy dict format
            for name_upper, prof_data in professors_cache.items():
                if isinstance(prof_data, dict) and 'id' in prof_data:
                    lookup[name_upper] = prof_data['id']
                    # Also add boss_aliases if present
                    boss_aliases = prof_data.get('boss_aliases')
                    if boss_aliases and isinstance(boss_aliases, list):
                        for alias in boss_aliases:
                            lookup[alias.upper()] = prof_data['id']

        # Then add/update with professors processed in this run (they may be new or updated)
        for dto in self.results.get('professors', {}).get('new', []) + \
                  self.results.get('professors', {}).get('updated', []):
            if dto.name:
                lookup[dto.name.upper()] = dto.id
            for alias in dto.boss_aliases:
                lookup[alias.upper()] = dto.id

        return lookup

    def _build_multiple_lookup(self) -> dict:
        """Build record_key -> rows lookup from multiple sheet."""
        multiple_df = self.raw_data[SHEET_MULTIPLE]
        multiple_lookup = {}
        for _, row in multiple_df.iterrows():
            record_key = row.get('record_key')
            if record_key:
                if record_key not in multiple_lookup:
                    multiple_lookup[record_key] = []
                multiple_lookup[record_key].append(row.to_dict())
        return multiple_lookup

    def _build_class_lookup(self) -> dict:
        """Build composite class lookup: (acad_term_id, boss_id, professor_id) -> ClassDTO."""
        lookup = {}
        for dto in self.results.get('classes', {}).get('new', []) + \
                  self.results.get('classes', {}).get('updated', []):
            key = (dto.acad_term_id, dto.boss_id, dto.professor_id)
            lookup[key] = dto
        return lookup

    def _get_overall_results_path(self) -> str:
        """Get the path to overallBossResults.xlsx based on scraped data location."""
        # Note: self.config.start_ay_term is already in dash format (START_AY_TERM from config)
        return os.path.join(
            self.config.verify_dir,
            'overallBossResults',
            self.config.start_ay_term + '.xlsx'
        )

    def _write_csv(self, filename: str, dtos: list, log_message: str):
        """Helper method to write DTOs to CSV file."""
        if not dtos:
            return
        output_file = os.path.join(self.config.output_base, filename)
        # Use COLUMNS keys (snake_case) not values (camelCase) to match to_csv_row() output
        headers = list(dtos[0].COLUMNS.keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
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
            self._write_csv('new_acad_terms.csv', terms['new'], f"✅ Saved acad_terms to {{filename}}")

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

        # Save bid windows
        if 'bid_windows' in self.results:
            bid_windows = self.results['bid_windows']
            self._write_csv('new_bid_windows.csv', bid_windows['new'],
                            f"✅ Saved {{count}} new bid windows to {{filename}}")
            self._write_csv('updated_bid_windows.csv', bid_windows['updated'],
                            f"✅ Saved {{count}} updated bid windows to {{filename}}")

        # Save classes
        if 'classes' in self.results:
            classes = self.results['classes']
            self._write_csv('new_classes.csv', classes['new'],
                            f"✅ Saved {{count}} new classes to {{filename}}")
            self._write_csv('updated_classes.csv', classes['updated'],
                            f"✅ Saved {{count}} updated classes to {{filename}}")

        # Save class timings
        if 'class_timings' in self.results:
            self._write_csv('new_class_timings.csv', self.results['class_timings'],
                            f"✅ Saved {{count}} new class timings to {{filename}}")

        # Save exam timings
        if 'class_exam_timings' in self.results:
            self._write_csv('new_class_exam_timings.csv', self.results['class_exam_timings'],
                            f"✅ Saved {{count}} new exam timings to {{filename}}")

        # Save class availabilities
        if 'class_availabilities' in self.results:
            self._write_csv('new_class_availabilities.csv', self.results['class_availabilities'],
                            f"✅ Saved {{count}} new class availabilities to {{filename}}")

        # Save bid results
        if 'bid_results' in self.results:
            bid_results = self.results['bid_results']
            self._write_csv('new_bid_results.csv', bid_results['new'],
                            f"✅ Saved {{count}} new bid results to {{filename}}")
            self._write_csv('updated_bid_results.csv', bid_results['updated'],
                            f"✅ Saved {{count}} updated bid results to {{filename}}")

        # Save bid predictions
        if 'bid_predictions' in self.results and self.results['bid_predictions']:
            self._write_csv('new_bid_predictions.csv', self.results['bid_predictions'],
                            f"✅ Saved {{count}} bid predictions to {{filename}}")

        # Save safety factors
        if 'safety_factors' in self.results and self.results['safety_factors']:
            self._write_csv('new_safety_factors.csv', self.results['safety_factors'],
                            f"✅ Saved {{count}} safety factors to {{filename}}")

    def save_to_database(self):
        """Persist results to PostgreSQL database."""
        if self._db_connection is None:
            self._logger.warning("No database connection - skipping database save")
            return

        tables = [
            ('acad_terms', 'acad_term'),
            ('courses', 'courses'),
            ('professors', 'professors'),
            ('bid_windows', 'bid_window'),
            ('classes', 'classes'),
            ('class_timings', 'class_timing'),
            ('class_exam_timings', 'class_exam_timing'),
            ('class_availabilities', 'class_availability'),
            ('bid_results', 'bid_result'),
            ('bid_predictions', 'bid_prediction'),
            ('safety_factors', 'safety_factor'),
        ]

        for result_key, table_name in tables:
            if result_key not in self.results:
                continue

            data = self.results[result_key]
            if isinstance(data, dict):
                # Has 'new' and 'updated' keys
                if data.get('new'):
                    df = pd.DataFrame([d.to_db_row() for d in data['new']])
                    DatabaseHelper.insert_df(self._db_connection, df, table_name, self._logger)
                if data.get('updated'):
                    df = pd.DataFrame([d.to_db_row() for d in data['updated']])
                    DatabaseHelper.update_df(self._db_connection, df, table_name, ['id'], self._logger)
            else:
                # List of DTOs (INSERT only)
                if data:
                    df = pd.DataFrame([d.to_db_row() for d in data])
                    DatabaseHelper.insert_df(self._db_connection, df, table_name, self._logger)

        try:
            self._db_connection.commit()
            self._logger.info("✅ Committed all results to database")
        except Exception as e:
            self._logger.error(f"Failed to commit to database: {e}")
            self._db_connection.rollback()