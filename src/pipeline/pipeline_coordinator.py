"""
PipelineCoordinator - orchestrates pipeline execution.
Sequentially calls processors and collects results as DTOs.
"""
import os
import csv
import pandas as pd
import pickle
from datetime import datetime
from typing import List

from src.config import ACAD_TERM_ID, RESULTS_DATETIME, dash_format_to_acad_term_id, encode_subterm_for_boss_id
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
from src.pipeline.dtos.bid_window_dto import BidWindowDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.course_dto import CourseDTO
from src.pipeline.dtos.acad_term_dto import AcadTermDTO
from src.pipeline.dtos.professor_dto import ProfessorDTO

# Sheet name constants
SHEET_STANDALONE = 'standalone'
SHEET_MULTIPLE = 'multiple'

# Mapping from plural result/cache keys to actual singular DB table names.
# Derived from Prisma schema (@@map directives).
# Cache keys (plural) are used internally; DB tables use singular names.
RESULT_KEY_TO_TABLE = {
    'acad_terms': 'acad_term',
    'courses': 'courses',
    'professors': 'professors',
    'faculties': 'faculties',
    'bid_windows': 'bid_window',
    'classes': 'classes',
    'class_timings': 'class_timing',
    'class_exam_timings': 'class_exam_timing',
    'class_availabilities': 'class_availability',
    'bid_results': 'bid_result',
    'bid_predictions': 'bid_prediction',
    'safety_factors': 'safety_factor',
}


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
        # Cache keys are plural — they match self.results keys so that
        # _build_lookup / _build_composite_lookup can use the same name
        # for both cache and results lookups.
        cache_files = {
            'acad_terms': 'acad_term_cache.pkl',
            'courses': 'courses_cache.pkl',
            'professors': 'professors_cache.pkl',
            'faculties': 'faculties_cache.pkl',
            'bid_windows': 'bid_window_cache.pkl',
            'classes': 'classes_cache.pkl',
            'class_timings': 'class_timing_cache.pkl',
            'class_exam_timings': 'class_exam_timing_cache.pkl',
            'bid_results': 'bid_result_cache.pkl',
            'class_availabilities': 'class_availability_cache.pkl',
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
            actual_tables = [RESULT_KEY_TO_TABLE[t] for t in tables_to_download]
            DatabaseHelper.download_cache(
                self._db_connection,
                self.config.cache_dir,
                actual_tables,
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
        # acad_terms: {id: row_dict}
        if isinstance(self.db_cache.get('acad_terms'), pd.DataFrame):
            df = self.db_cache['acad_terms']
            if not df.empty and 'id' in df.columns:
                self.db_cache['acad_terms'] = dict(zip(df['id'], df.to_dict('records')))
            else:
                self.db_cache['acad_terms'] = {}
            self._logger.info(f"Converted acad_terms cache to dict with {len(self.db_cache['acad_terms'])} entries")

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

        # bid_windows: {(acad_term_id, round, window): row_dict}
        if isinstance(self.db_cache.get('bid_windows'), pd.DataFrame):
            df = self.db_cache['bid_windows']
            if not df.empty and all(col in df.columns for col in ['acad_term_id', 'round', 'window']):
                self.db_cache['bid_windows'] = {}
                for _, row in df.iterrows():
                    key = (row['acad_term_id'], str(row['round']), int(row['window']))
                    self.db_cache['bid_windows'][key] = row.to_dict()
            else:
                self.db_cache['bid_windows'] = {}
            self._logger.info(f"Converted bid_windows cache to dict with {len(self.db_cache['bid_windows'])} entries")

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

        # class_timings: set of (class_id, day_of_week, start_time, end_time, venue) keys
        if isinstance(self.db_cache.get('class_timings'), pd.DataFrame):
            df = self.db_cache['class_timings']
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
                self.db_cache['class_timings'] = existing_timing_keys
            else:
                self.db_cache['class_timings'] = set()
            self._logger.info(f"Converted class_timings cache to set with {len(self.db_cache['class_timings'])} entries")

        # class_exam_timings: set of class_ids that already have exam timings
        if isinstance(self.db_cache.get('class_exam_timings'), pd.DataFrame):
            df = self.db_cache['class_exam_timings']
            if not df.empty and 'class_id' in df.columns:
                existing_exam_class_ids = set(df['class_id'].astype(str).unique())
                self.db_cache['class_exam_timings'] = existing_exam_class_ids
            else:
                self.db_cache['class_exam_timings'] = set()
            self._logger.info(f"Converted class_exam_timings cache to set with {len(self.db_cache['class_exam_timings'])} entries")

        # bid_results: set of (bid_window_id, class_id) tuples for dedup
        if isinstance(self.db_cache.get('bid_results'), pd.DataFrame):
            df = self.db_cache['bid_results']
            if not df.empty and 'bid_window_id' in df.columns and 'class_id' in df.columns:
                existing_bid_result_keys = set()
                for _, row in df.iterrows():
                    existing_bid_result_keys.add((int(row['bid_window_id']), str(row['class_id'])))
                self.db_cache['bid_results'] = existing_bid_result_keys
            else:
                self.db_cache['bid_results'] = set()
            self._logger.info(f"Converted bid_results cache to set with {len(self.db_cache['bid_results'])} entries")

        # class_availabilities: set of (class_id, bid_window_id) tuples for dedup
        if isinstance(self.db_cache.get('class_availabilities'), pd.DataFrame):
            df = self.db_cache['class_availabilities']
            if not df.empty and 'class_id' in df.columns and 'bid_window_id' in df.columns:
                existing_availability_keys = set()
                for _, row in df.iterrows():
                    existing_availability_keys.add((str(row['class_id']), int(row['bid_window_id'])))
                self.db_cache['class_availabilities'] = existing_availability_keys
            else:
                self.db_cache['class_availabilities'] = set()
            self._logger.info(f"Converted class_availabilities cache to set with {len(self.db_cache['class_availabilities'])} entries")

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
            acad_term_cache=self.db_cache.get('acad_terms', {}),
            logger=self._logger
        )
        acad_terms_new, acad_terms_updated = acad_term_processor.process()

        # Fallback: ensure current term exists even if not in scraped data
        expected_acad_term_id = dash_format_to_acad_term_id(self.config.start_ay_term)
        existing_ids = {t.id for t in acad_terms_new}
        acad_cache = self.db_cache.get('acad_terms', {})
        db_ids = set(acad_cache.keys()) if isinstance(acad_cache, dict) else set()

        if expected_acad_term_id not in existing_ids and expected_acad_term_id not in db_ids:
            # Parse from start_ay_term format "2026-27_T1" -> year_start=2026, year_end=27, term_num=1
            import re
            m = re.match(r'(\d{4})-(\d{2})_T(\d+)([AB]?)', self.config.start_ay_term)
            if m:
                y_start = int(m.group(1))
                y_end = int(m.group(2))
                t_num = m.group(3) + m.group(4)
                # boss_id uses start year prefix (BOSS convention), with sub-term encoding
                sub = encode_subterm_for_boss_id(t_num)
                boss_id = int(f"{y_start}{m.group(3)}{sub}")
                # Note: start_dt/end_dt are set to None — bidding schedule dates are
                # results release timestamps, not academic term boundaries.
                fallback_term = AcadTermDTO(
                    id=expected_acad_term_id,
                    acad_year_start=y_start,
                    acad_year_end=2000 + y_end,
                    term=t_num,
                    boss_id=boss_id,
                    start_dt=None,
                    end_dt=None
                )
                acad_terms_new.append(fallback_term)
                self._logger.info(f"Created fallback acad_term: {expected_acad_term_id}")

        self.results['acad_terms'] = {'new': acad_terms_new, 'updated': acad_terms_updated}
        self.results['acad_term_lookup'] = self._build_lookup('acad_terms', 'id')
        self._logger.info(f"Processed {len(acad_terms_new)} academic terms")

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
        professor_resolution_service = professor_processor.resolution_service
        self._logger.info(f"✅ Processed professors: {len(professors_new)} new, {len(professors_updated)} updated")

        # Process bid windows
        bid_window_processor = BidWindowProcessor(
            raw_data=self.raw_data[SHEET_STANDALONE],
            bid_window_cache=self.db_cache.get('bid_windows', {}),
            expected_acad_term_id=dash_format_to_acad_term_id(self.config.start_ay_term),
            bidding_schedules=self.config.bidding_schedules,
            results_datetime=RESULTS_DATETIME,
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
            professor_resolution_service=professor_resolution_service,
            existing_classes_cache=existing_classes_cache,
            logger=self._logger
        )
        classes_new, classes_updated = class_processor.process()
        self.results['classes'] = {'new': classes_new, 'updated': classes_updated}

        # Process class deactivations (excess classes → set professor_id=None)
        classes_to_deactivate = class_processor.get_classes_to_deactivate()
        if classes_to_deactivate:
            self._logger.info(f"Processing {len(classes_to_deactivate)} class deactivations")
            for c in classes_to_deactivate:
                if c.get('professor_id') is not None:
                    deactivated_dto = ClassDTO(
                        id=c['id'],
                        section=c.get('section', ''),
                        course_id=c.get('course_id', ''),
                        professor_id=None,
                        acad_term_id=c.get('acad_term_id', ''),
                        grading_basis=c.get('grading_basis'),
                        course_outline_url=c.get('course_outline_url'),
                        boss_id=int(c['boss_id']) if c.get('boss_id') is not None else None,
                        warn_inaccuracy=c.get('warn_inaccuracy', False),
                        created_at=c.get('created_at'),
                        updated_at=datetime.now()
                    )
                    classes_updated.append(deactivated_dto)
            self._logger.info(f"✅ Deactivated {len(classes_to_deactivate)} excess class records")

        self.results['class_lookup'] = self._build_class_lookup()
        self._logger.info(f"✅ Processed classes: {len(classes_new)} new, {len(classes_updated)} updated")

        # Get record_key -> [class_ids] mapping for timing processing
        record_key_to_class_ids = class_processor.get_record_key_to_class_ids_mapping()
        self._logger.info(f"📊 Built record_key -> class_ids mapping with {len(record_key_to_class_ids)} entries")

        # Process class timings
        existing_timing_keys = self.db_cache.get('class_timings', set())
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
        existing_exam_class_ids = self.db_cache.get('class_exam_timings', set())
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
            expected_acad_term_id=dash_format_to_acad_term_id(self.config.start_ay_term),
            existing_availability_keys=self.db_cache.get('class_availabilities', set()),
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
            existing_bid_result_keys=self.db_cache.get('bid_results', set()),
            bidding_schedule=bidding_schedule,
            expected_acad_term_id=dash_format_to_acad_term_id(self.config.start_ay_term),
            logger=self._logger
        )
        bid_results_new, bid_results_updated = bid_result_processor.process()
        self.results['bid_results'] = {'new': bid_results_new, 'updated': bid_results_updated}
        self._logger.info(f"✅ Processed bid_results: {len(bid_results_new)} new, {len(bid_results_updated)} updated")

        # ============================================================
        # Phase 3: Bid Predictions
        # ============================================================

        try:
            bid_prediction_processor = BidPredictionProcessor(
                raw_data=self.raw_data[SHEET_STANDALONE],
                class_lookup=self.results['class_lookup'],
                bid_window_lookup=self.results['bid_window_lookup'],
                multiple_lookup=self._multiple_lookup,
                bidding_schedule=bidding_schedule,
                expected_acad_term_id=dash_format_to_acad_term_id(self.config.start_ay_term),
                model_dir='models',
                logger=self._logger
            )
            predictions = bid_prediction_processor.process()
            self.results['bid_predictions'] = predictions
            self._logger.info(f"✅ Generated {len(predictions)} bid predictions")

            safety_factor_processor = SafetyFactorProcessor(
                expected_acad_term_id=ACAD_TERM_ID,
                cache_dir=self.config.cache_dir,
                logger=self._logger,
                db_connection=self._db_connection,
            )
            safety_factors = safety_factor_processor.process()
            if safety_factors:
                self.results['safety_factors'] = safety_factors
                self._logger.info(f"✅ Generated {len(safety_factors)} safety factor entries")
        except Exception as e:
            self._logger.error(f"Phase 3 (predictions) failed — saving Phase 1-2 results anyway: {e}")

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
        Also includes existing DTOs from the database cache.

        Args:
            dimension: The dimension name in self.results (e.g., 'courses', 'acad_terms')
            key_field: The DTO attribute to use as key (e.g., 'code', 'id')

        Returns:
            Dict mapping key_field value to DTO
        """
        lookup = {}

        # First, add existing DTOs from database cache (if available)
        # This ensures lookup contains ALL existing records, not just new/updated
        cache = self.db_cache.get(dimension, {})
        if isinstance(cache, list):
            # Cache is a list of dicts, convert to DTOs based on dimension type
            for item in cache:
                dto = self._dict_to_dto(dimension, item)
                if dto and hasattr(dto, key_field):
                    lookup[getattr(dto, key_field)] = dto
        elif isinstance(cache, dict):
            # Cache is already a dict, may need conversion
            for key, item in cache.items():
                if hasattr(item, key_field):
                    lookup[getattr(item, key_field)] = item
                elif isinstance(item, dict):
                    dto = self._dict_to_dto(dimension, item)
                    if dto and hasattr(dto, key_field):
                        lookup[getattr(dto, key_field)] = dto

        # Then add new/updated DTOs (these override cache entries if same key)
        data = self.results.get(dimension, {})
        for dto in data.get('new', []):
            lookup[getattr(dto, key_field)] = dto
        for dto in data.get('updated', []):
            lookup[getattr(dto, key_field)] = dto

        return lookup

    def _dict_to_dto(self, dimension: str, item: dict):
        """Convert a dict from database cache to a DTO.

        Accepts both singular (DB table name) and plural (results key)
        dimension names so that both _build_lookup and _build_composite_lookup
        work regardless of which convention the caller uses.
        """
        if dimension in ('courses',):
            return CourseDTO.from_dict(item) if hasattr(CourseDTO, 'from_dict') else self._create_course_dto(item)
        elif dimension in ('acad_term', 'acad_terms'):
            return AcadTermDTO.from_dict(item) if hasattr(AcadTermDTO, 'from_dict') else None
        elif dimension in ('professors',):
            return ProfessorDTO.from_dict(item) if hasattr(ProfessorDTO, 'from_dict') else None
        elif dimension in ('bid_window', 'bid_windows'):
            return BidWindowDTO.from_dict(item)
        return None

    def _create_course_dto(self, item: dict):
        """Create a CourseDTO from a dict."""
        return CourseDTO(
            id=item.get('id'),
            code=item.get('code'),
            name=item.get('name'),
            description=item.get('description'),
            credit_units=item.get('credit_units'),
            belong_to_university=item.get('belong_to_university'),
            belong_to_faculty=item.get('belong_to_faculty'),
            course_area=item.get('course_area'),
            enrolment_requirements=item.get('enrolment_requirements'),
            updated_at=item.get('updated_at', datetime.now())
        )

    def _build_composite_lookup(self, dimension: str, key_fields: List[str]) -> dict:
        """Build lookup using multiple fields as key.

        Includes existing entries from DB cache first, then overlays
        new/updated DTOs (which override cache entries with the same key).

        Args:
            dimension: The dimension name in self.results and self.db_cache
            key_fields: List of DTO attribute names to use as composite key

        Returns:
            Dict mapping tuple of key_field values to DTO
        """
        lookup = {}

        # First, add existing entries from database cache (if available)
        cache = self.db_cache.get(dimension, {})
        if isinstance(cache, dict):
            for cache_key, item in cache.items():
                if isinstance(item, dict):
                    dto = self._dict_to_dto(dimension, item)
                    if dto is not None:
                        lookup_key = tuple(getattr(dto, f) for f in key_fields)
                        lookup[lookup_key] = dto
                elif hasattr(item, key_fields[0]):
                    lookup_key = tuple(getattr(item, f) for f in key_fields)
                    lookup[lookup_key] = item

        # Then add new/updated DTOs (these override cache entries if same key)
        data = self.results.get(dimension, {})
        for dto in data.get('new', []) + data.get('updated', []):
            key = tuple(getattr(dto, f) for f in key_fields)
            lookup[key] = dto
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

    def _dict_to_class_dto(self, item: dict):
        """Create a ClassDTO from a dict (e.g., from database cache)."""
        return ClassDTO(
            id=item.get('id'),
            section=item.get('section', ''),
            course_id=item.get('course_id', ''),
            professor_id=item.get('professor_id'),
            acad_term_id=item.get('acad_term_id', ''),
            grading_basis=item.get('grading_basis'),
            course_outline_url=item.get('course_outline_url'),
            boss_id=int(item.get('boss_id')) if item.get('boss_id') is not None else None,
            warn_inaccuracy=item.get('warn_inaccuracy', False),
            created_at=item.get('created_at'),
            updated_at=item.get('updated_at')
        )

    def _build_class_lookup(self) -> dict:
        """Build composite class lookup: (acad_term_id, boss_id, professor_id) -> ClassDTO.

        Includes existing classes from DB cache first, then new/updated DTOs (which override
        cache entries with the same key). This ensures downstream processors can find ALL
        classes, not just those created or updated this run.
        """
        lookup = {}
        # Add existing classes from DB cache first
        classes_cache = self.db_cache.get('classes', {})
        if isinstance(classes_cache, dict):
            for key, item in classes_cache.items():
                dto = self._dict_to_class_dto(item) if isinstance(item, dict) else item
                if dto and hasattr(dto, 'acad_term_id'):
                    lookup_key = (dto.acad_term_id, dto.boss_id, dto.professor_id)
                    lookup[lookup_key] = dto
        # Then add new/updated DTOs (override cache entries if same key)
        for dto in self.results.get('classes', {}).get('new', []) + \
                  self.results.get('classes', {}).get('updated', []):
            key = (dto.acad_term_id, dto.boss_id, dto.professor_id)
            lookup[key] = dto
        return lookup

    def _get_overall_results_path(self) -> str:
        """Get the path to overallBossResults.xlsx.

        Uses dedicated overall_results_dir config instead of deriving
        from input_file, so the path is stable regardless of input_file location.
        """
        return os.path.join(
            self.config.overall_results_dir,
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

        tables = list(RESULT_KEY_TO_TABLE.items())

        # Tables whose primary key is not a single 'id' column.
        # update_df uses these columns to build WHERE clauses.
        _UPDATE_KEY_MAP = {
            'bid_results': ['bid_window_id', 'class_id'],
            'class_availabilities': ['class_id', 'bid_window_id'],
        }

        for result_key, table_name in tables:
            if result_key not in self.results:
                continue

            data = self.results[result_key]
            if isinstance(data, dict):
                # Has 'new' and 'updated' keys
                if data.get('new'):
                    df = pd.DataFrame([d.to_db_row() for d in data['new']])
                    on_conflict = (result_key in ('bid_results', 'safety_factors'))
                    DatabaseHelper.insert_df(self._db_connection, df, table_name, self._logger,
                                             on_conflict_do_nothing=on_conflict)
                if data.get('updated'):
                    df = pd.DataFrame([d.to_db_row() for d in data['updated']])
                    index_elements = _UPDATE_KEY_MAP.get(result_key, ['id'])
                    DatabaseHelper.update_df(self._db_connection, df, table_name, index_elements, self._logger)
            else:
                # List of DTOs (INSERT only)
                if data:
                    df = pd.DataFrame([d.to_db_row() for d in data])
                    # safety_factors are generated once per semester — skip if already present
                    on_conflict = (result_key in ('safety_factors', 'class_availabilities', 'bid_predictions'))
                    DatabaseHelper.insert_df(self._db_connection, df, table_name, self._logger,
                                             on_conflict_do_nothing=on_conflict)

        try:
            self._db_connection.commit()
            self._logger.info("✅ Committed all results to database")
        except Exception as e:
            self._logger.error(f"Failed to commit to database: {e}")
            self._db_connection.rollback()