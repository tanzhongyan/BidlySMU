from dataclasses import dataclass
from typing import Optional, Dict
import os
from dotenv import load_dotenv

from src.db.adapters import Psycopg2Adapter
from src.db.database_helper import DatabaseHelper
from src.utils.term_resolver import convert_target_term_format
from src.utils.name_data import (
    ASIAN_SURNAMES, ALL_ASIAN_SURNAMES, WESTERN_GIVEN_NAMES,
    PATRONYMIC_KEYWORDS, SURNAME_PARTICLES
)
from src.utils.professor_normalizer import ProfessorNormalizer
from src.logging.logger import get_logger
from src.pipeline.processor_context import ProcessorContext
from src.pipeline.processors.professor_processor import ProfessorProcessor
from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processors.class_processor import ClassProcessor
from src.pipeline.processors.timing_processor import TimingProcessor
from src.pipeline.processors.boss_processor import BOSSProcessor

# Import dependencieS
import json
import logging
import pandas as pd
from google import genai
from psycopg2.extras import execute_values
import traceback
import math


@dataclass(frozen=True)
class TableBuilderConfig:
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int
    gemini_api_key: Optional[str]
    input_file: str = 'script_input/raw_data.xlsx'
    output_base: str = 'script_output'
    verify_dir: str = 'script_output/verify'
    cache_dir: str = 'db_cache'
    bidding_schedules: Optional[Dict] = None
    start_ay_term: Optional[str] = None
    llm_model_name: str = 'gemini-2.5-flash'
    llm_batch_size: int = 50

    @classmethod
    def from_env(cls, bidding_schedules: dict = None, start_ay_term: str = None) -> 'TableBuilderConfig':
        """Create config from environment variables."""
        load_dotenv()
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')

        if not all([db_host, db_name, db_user, db_password]):
            raise ValueError("Missing database configuration in environment variables.")

        return cls(
            db_host=db_host,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
            db_port=int(os.getenv('DB_PORT', 5432)),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            bidding_schedules=bidding_schedules or {},
            start_ay_term=start_ay_term
        )


class TableBuilder:
    """Comprehensive table builder for university class management system"""
    
    def __init__(self, config=None, logger=None, db_connection=None):
        """Initialize TableBuilder with configuration and logger."""
        if config is None:
            raise ValueError("config is required")
        self.config = config
        self._logger = logger or get_logger(__name__)
        self.logger = self._logger  # Alias for DatabaseHelper compatibility
        self.db_connection = db_connection
        
        self.input_file = self.config.input_file
        self.output_base = self.config.output_base
        self.verify_dir = self.config.verify_dir
        self.cache_dir = self.config.cache_dir
        
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.verify_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.db_config = {
            'host': self.config.db_host,
            'database': self.config.db_name,
            'user': self.config.db_user,
            'password': self.config.db_password,
            'port': self.config.db_port
        }
        
        self.standalone_data = None
        self.multiple_data = None
        self.professors_cache = {}
        self.courses_cache = {}
        self.acad_term_cache = {}
        self.faculties_cache = {}
        self.faculty_acronym_to_id = {}
        self.professor_lookup = {}
        self.processed_timing_keys = set()
        self.processed_exam_class_ids = set()
        self.new_professors = []
        self.new_courses = []
        self.update_courses = []
        self.update_classes = []
        self.new_acad_terms = []
        self.new_classes = []
        self.new_class_timings = []
        self.new_class_exam_timings = []
        self.update_professors = []
        self.update_bid_result = []
        self.class_id_mapping = {}
        self.existing_classes_cache = []
        self.existing_class_lookup = {}
        self.new_bid_windows = []
        self.new_class_availability = []
        self.failed_mappings = []
        self.bid_window_cache = {}
        self.bid_window_id_counter = 1
        self.stats = {
            'professors_created': 0, 'professors_updated': 0, 'courses_created': 0,
            'courses_updated': 0, 'classes_created': 0, 'timings_created': 0,
            'exams_created': 0
        }
        
        bidding_schedules = self.config.bidding_schedules or {}
        self.bidding_schedule = bidding_schedules.get(self.config.start_ay_term, [])
        self.expected_acad_term_id = convert_target_term_format(self.config.start_ay_term)
        
        # Surnames setup... (unchanged from original but kept inside init)
        
        # Name data (lifted to src/utils/name_data.py)
        self.asian_surnames = ASIAN_SURNAMES
        self.all_asian_surnames = ALL_ASIAN_SURNAMES.copy()
        self.western_given_names = WESTERN_GIVEN_NAMES
        self.patronymic_keywords = PATRONYMIC_KEYWORDS
        self.surname_particles = SURNAME_PARTICLES

        # Bid results data collectors
        self.boss_log_file = os.path.join(self.output_base, 'boss_result_log.txt')
        self.new_bid_windows = []
        self.new_class_availability = []
        self.new_bid_result = []
        self.failed_mappings = []
        self.bid_window_cache = {}
        self.bid_window_id_counter = 1

        # Professor lookup from CSV
        self.professor_lookup = {}
        
        # LLM Configuration
        self._logger.info("🔧 Initializing LLM configuration...")
        self.llm_model_name = "gemini-2.5-flash"
        self.llm_batch_size = 50
        self.llm_prompt = """
        You are an expert in academic name structures from around the world.
        You will be given a JSON list of professor names.
        Your task is to identify the primary surname for each name.
        You MUST return a single JSON array of strings, where each string is the identified surname.
        The order of surnames in your response must exactly match the order of the full names in the input list.
        Provide ONLY the JSON array in your response.
        """
        
        # Pre-configure the model if the API key exists
        self.llm_client = None
        if self.config.gemini_api_key:
            try:
                # Create client with API key
                self.llm_client = genai.Client(api_key=self.config.gemini_api_key)
                
                # Test the client to make sure it works
                # You can remove this test call if you want
                test_response = self.llm_client.models.generate_content(
                    model=self.llm_model_name,
                    contents="Test"
                )
                
                self._logger.info(f"✅ Gemini client '{self.llm_model_name}' configured successfully.")
            except Exception as e:
                self._logger.warning(f"⚠️ Could not pre-configure Gemini client: {e}")
                self.llm_client = None
        else:
            self._logger.warning("⚠️ GEMINI_API_KEY not found. LLM normalization will be skipped.")

        # Initialize database connection
        self.db_connection = None
        self.db_transaction = None
        self._db_adapter = Psycopg2Adapter(self.db_config, logger=self._logger)

        # Initialize ProfessorNormalizer
        self.professor_normalizer = ProfessorNormalizer(
            logger=self._logger,
            professor_lookup=self.professor_lookup,
            professors_cache=self.professors_cache,
            llm_client=self.llm_client,
            llm_model_name=self.llm_model_name,
            llm_batch_size=self.llm_batch_size,
            llm_prompt=self.llm_prompt
        )

    def connect_database(self) -> bool:
        """Connect to the database using the configured adapter."""
        try:
            self._logger.info("Connecting to database...")
            self.db_connection = self._db_adapter.connect()
            if self.db_connection:
                self.db_transaction = self.db_connection
                self._logger.info("✅ Database connection established")
                return True
            else:
                self._logger.error("❌ Database connection failed")
                return False
        except Exception as e:
            self._logger.error(f"❌ Database connection error: {e}")
            return False

    def _create_processor_context(self):
        """Create ProcessorContext with current TableBuilder state."""
        return ProcessorContext(
            config=self.config,
            logger=self._logger,
            db_connection=self.db_connection,
            professors_cache=self.professors_cache,
            courses_cache=self.courses_cache,
            acad_term_cache=self.acad_term_cache,
            faculties_cache=self.faculties_cache,
            bid_window_cache=self.bid_window_cache,
            professor_lookup=self.professor_lookup,
            existing_classes_cache=getattr(self, 'existing_classes_cache', {}),
            standalone_data=self.standalone_data,
            multiple_data=self.multiple_data,
            boss_data=getattr(self, 'boss_data', None),
            multiple_lookup=getattr(self, 'multiple_lookup', {}),
            faculty_acronym_to_id=self.faculty_acronym_to_id,
            class_id_mapping=self.class_id_mapping,
            new_professors=self.new_professors,
            update_professors=self.update_professors,
            new_courses=self.new_courses,
            update_courses=self.update_courses,
            new_acad_terms=self.new_acad_terms,
            new_classes=self.new_classes,
            new_class_timings=self.new_class_timings,
            new_class_exam_timings=self.new_class_exam_timings,
            update_classes=self.update_classes,
            new_bid_windows=self.new_bid_windows,
            new_class_availability=self.new_class_availability,
            new_bid_result=self.new_bid_result,
            update_bid_result=self.update_bid_result,
            new_faculties=getattr(self, 'new_faculties', []),
            stats=self.stats,
            processed_timing_keys=self.processed_timing_keys,
            processed_exam_class_ids=self.processed_exam_class_ids,
            failed_mappings=self.failed_mappings,
            llm_client=self.llm_client,
            llm_model_name=self.llm_model_name,
            llm_batch_size=self.llm_batch_size,
            llm_prompt=self.llm_prompt,
            bid_window_id_counter=self.bid_window_id_counter,
            expected_acad_term_id=getattr(self, 'expected_acad_term_id', None),
            boss_stats=getattr(self, 'boss_stats', {}),
        )


    def _execute_db_operations(self):
        """Executes all collected INSERT, UPDATE, and UPSERT operations within the transaction."""
        self._logger.info("Executing database operations...")
        conn = self.db_connection
        cursor = conn.cursor()

        try:
            # 1. Simple INSERTS using execute_values
            insert_map = {
                'professors': self.new_professors,
                'courses': self.new_courses,
                'acad_term': self.new_acad_terms,
                'classes': self.new_classes,
                'class_timing': self.new_class_timings,
                'class_exam_timing': self.new_class_exam_timings,
                'bid_window': self.new_bid_windows
            }
            for table_name, data_list in insert_map.items():
                if data_list:
                    df = pd.DataFrame(data_list)
                    if df.empty:
                        continue
                    self._logger.info(f"Inserting {len(df)} records into {table_name}...")
                    cols = df.columns.tolist()
                    # Convert numpy types to native Python to avoid psycopg2 issues
                    import numpy as np
                    from datetime import datetime, timezone
                    def clean_value(v):
                        if v is None:
                            return None
                        # Handle numpy types that cause issues with math.isnan
                        if isinstance(v, np.floating):
                            if np.isnan(v):
                                return None
                            return float(v)
                        if isinstance(v, float):
                            try:
                                if math.isnan(v):
                                    return None
                            except TypeError:
                                pass
                            return v
                        if isinstance(v, str) and v.strip() == '':
                            return None
                        return v

                    # Tables with created_at/updated_at timestamps
                    tables_with_timestamps = {'courses', 'professors', 'classes'}
                    needs_timestamps = table_name in tables_with_timestamps

                    # Pre-compute whether we need to add timestamps (before the loop)
                    add_created = needs_timestamps and 'created_at' not in df.columns
                    add_updated = needs_timestamps and 'updated_at' not in df.columns

                    # Add missing timestamp column names to cols
                    if add_created:
                        cols.append('created_at')
                    if add_updated:
                        cols.append('updated_at')

                    values = []
                    for row in df.to_numpy():
                        row_values = [clean_value(v) for v in row]
                        # Add default timestamps if missing
                        if add_created:
                            row_values.append(datetime.now(timezone.utc))
                        if add_updated:
                            row_values.append(datetime.now(timezone.utc))
                        values.append(tuple(row_values))
                    sql = f'INSERT INTO "{table_name}" ({", ".join(f'"{c}"' for c in cols)}) VALUES %s'
                    execute_values(cursor, sql, values, page_size=1000)

            # 2. UPDATES using cursor.execute
            def execute_update(table, records):
                self._logger.info(f"Updating {len(records)} records in {table}...")
                for record in records:
                    record_id = record.pop('id')
                    # Use raw SQL with parameterized query for safety
                    set_clause = ', '.join([f'"{k}" = %s' for k in record.keys()])
                    sql = f'UPDATE "{table}" SET {set_clause} WHERE id = %s'
                    cursor.execute(sql, list(record.values()) + [record_id])

            if self.update_courses:
                execute_update('courses', self.update_courses)
            if hasattr(self, 'update_classes') and self.update_classes:
                execute_update('classes', self.update_classes)
            if self.update_professors:
                self._logger.info(f"Updating {len(self.update_professors)} records in professors...")
                for record in self.update_professors:
                    record_id = record.pop('id')
                    # boss_aliases is a text[] in postgres, format correctly
                    aliases = json.loads(record['boss_aliases'])
                    sql = f'UPDATE "professors" SET "boss_aliases" = %s WHERE id = %s'
                    cursor.execute(sql, [aliases, record_id])

            # 3. UPSERTS using psycopg2_upsert_df
            def execute_upsert(table, records, index_elements):
                if not records:
                    return
                self._logger.info(f"Upserting {len(records)} records into {table}...")
                df = pd.DataFrame(records)
                DatabaseHelper.upsert_df(conn, df, table, index_elements, logger=self._logger)

            if self.new_class_availability:
                execute_upsert('class_availability', self.new_class_availability, ['class_id', 'bid_window_id'])

            all_bid_results = self.new_bid_result + getattr(self, 'update_bid_result', [])
            if all_bid_results:
                execute_upsert('bid_result', all_bid_results, ['bid_window_id', 'class_id'])

            self._logger.info("✅ Database operations executed successfully within transaction.")
            cursor.close()
            return True

        except Exception as e:
            self._logger.error(f"❌ Error during database operations: {e}")
            traceback.print_exc()
            cursor.close()
            raise  # Re-raise the exception to trigger a rollback

    def save_outputs(self):
        """Save all generated CSV files, only creating files that have data."""
        self._logger.info("💾 Saving output files...")

        def to_csv_if_not_empty(data_list, filename):
            if data_list:
                df = pd.DataFrame(data_list)
                if not df.empty:
                    path = os.path.join(self.output_base, filename)
                    df.to_csv(path, index=False)
                    self._logger.info(f"✅ Saved {len(df)} records to {filename}")
                    
                    # DEBUG: For update_bid_result, show what we're saving
                    if filename == 'update_bid_result.csv':
                        self._logger.info(f"🔍 DEBUG: update_bid_result.csv columns: {list(df.columns)}")
                        # Show first few rows with median/min values
                        rows_with_bid_data = df[(df['median'].notna()) | (df['min'].notna())]
                        if not rows_with_bid_data.empty:
                            self._logger.info(f"🔍 DEBUG: {len(rows_with_bid_data)} rows have median/min data")
                            self._logger.info(f"🔍 DEBUG: Sample data:")
                            for _, row in rows_with_bid_data.head(3).iterrows():
                                self._logger.info(f"  bid_window_id={row['bid_window_id']}, class_id={row['class_id']}, median={row.get('median')}, min={row.get('min')}")
                        else:
                            self._logger.warning("⚠️ DEBUG: No rows in update_bid_result have median/min values!")

        # The following line is the only change. It has been removed.
        # to_csv_if_not_empty(self.new_professors, 'new_professors.csv') 
        
        to_csv_if_not_empty(getattr(self, 'update_professors', []), 'update_professor.csv')
        # We also no longer need to save new_courses here, as it's handled in Phase 1.
        # to_csv_if_not_empty(self.new_courses, os.path.join('verify', 'new_courses.csv'))
        to_csv_if_not_empty(self.update_courses, 'update_courses.csv')
        to_csv_if_not_empty(getattr(self, 'update_classes', []), 'update_classes.csv')
        to_csv_if_not_empty(self.new_acad_terms, 'new_acad_term.csv')
        to_csv_if_not_empty(self.new_classes, 'new_classes.csv')
        to_csv_if_not_empty(self.new_class_timings, 'new_class_timing.csv')
        to_csv_if_not_empty(self.new_class_exam_timings, 'new_class_exam_timing.csv')
        update_records = getattr(self, 'update_bid_result', [])
        if update_records:
            self._logger.info(f"📝 Preparing to save {len(update_records)} bid result updates")
            # Show sample of records with actual bid data
            records_with_bids = [r for r in update_records if r.get('median') is not None or r.get('min') is not None]
            self._logger.info(f"   - {len(records_with_bids)} records have median/min bid data")
            if records_with_bids:
                sample = records_with_bids[0]
                self._logger.info(f"   - Sample: bid_window_id={sample.get('bid_window_id')}, median={sample.get('median')}, min={sample.get('min')}")
        to_csv_if_not_empty(update_records, 'update_bid_result.csv')

    def process_remaining_tables(self):
        """Process classes and timings after professor lookup is updated"""
        self._logger.info("🏫 Processing remaining tables (classes, timings)...")
        
        try:
            # Clear any existing data from Phase 1 to avoid duplicates
            self.new_classes = []
            self.new_class_timings = []
            self.new_class_exam_timings = []
            self.class_id_mapping = {}
            self.stats['classes_created'] = 0
            self.stats['timings_created'] = 0
            self.stats['exams_created'] = 0
            
            # Process classes (depends on updated professor lookup)
            class_processor = ClassProcessor(self._create_processor_context())
            class_processor.process()

            # Process timings (depends on classes)
            timing_processor = TimingProcessor(self._create_processor_context())
            timing_processor.process()
            
            self._logger.info("✅ Remaining tables processed successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"❌ Failed to process remaining tables: {e}")
            return False

    def print_summary(self):
        """Print processing summary"""
        self._logger.info("\n" + "="*70)
        self._logger.info("📊 PROCESSING SUMMARY")
        self._logger.info("="*70)
        self._logger.info(f"✅ Professors created: {self.stats['professors_created']}")
        self._logger.info(f"✅ Courses created: {self.stats['courses_created']}")
        self._logger.info(f"✅ Courses updated: {self.stats['courses_updated']}")
        self._logger.info(f"✅ Classes created: {self.stats['classes_created']}")
        self._logger.info(f"✅ Class timings created: {self.stats['timings_created']}")
        self._logger.info(f"✅ Exam timings created: {self.stats['exams_created']}")
        self._logger.info("="*70)
        
        self._logger.info("\n📁 OUTPUT FILES:")
        self._logger.info(f"   Verify folder: {self.verify_dir}/")
        self._logger.info(f"   - new_professors.csv ({self.stats['professors_created']} records)")
        self._logger.info(f"   - new_courses.csv ({self.stats['courses_created']} records)")
        self._logger.info(f"   Output folder: {self.output_base}/")
        self._logger.info(f"   - update_courses.csv ({self.stats['courses_updated']} records)")
        self._logger.info(f"   - new_acad_term.csv ({len(self.new_acad_terms)} records)")
        self._logger.info(f"   - new_classes.csv ({self.stats['classes_created']} records)")
        self._logger.info(f"   - new_class_timing.csv ({self.stats['timings_created']} records)")
        self._logger.info(f"   - new_class_exam_timing.csv ({self.stats['exams_created']} records)")
        self._logger.info(f"   - professor_lookup.csv (updated)")
        self._logger.info("="*70)

    def run_phase1_professors_and_courses(self):
        """Phase 1: Process professors and courses with automated faculty mapping and cache checking"""
        try:
            self._logger.info("🚀 Starting Phase 1: Professors and Courses with Cache Checking")
            self._logger.info("="*60)
            
            # Step 1: Load or cache database data.
            # The new _load_from_cache now handles all professor lookup validation and synchronization internally.
            if not DatabaseHelper.load_or_cache_data_with_freshness_check(self):
                self._logger.error("❌ Failed to load or validate database data")
                return False
            
            # Step 2: Load the raw input data from Excel
            if not DatabaseHelper.load_raw_data(self):
                self._logger.error("❌ Failed to load raw data")
                return False
            
            # Step 3: Process the data using the now-validated caches
            self._logger.info("\n🎓 Running automated faculty mapping...")
            professor_processor = ProfessorProcessor(self._create_processor_context())
            professor_processor.process()

            course_processor = CourseProcessor(self._create_processor_context())
            course_processor.process()

            acad_term_processor = AcadTermProcessor(self._create_processor_context())
            acad_term_processor.process()
            
            # Step 4: Save phase 1 outputs
            self._save_phase1_outputs()

            self._logger.info("✅ Phase 1 completed - Review files in verify/ folder")
            return True
            
        except Exception as e:
            self._logger.error(f"❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase2_remaining_tables(self):
        """Phase 2: Process classes and timings after professor correction with cache checking"""
        try:
            self._logger.info("🚀 Starting Phase 2: Classes and Timings with Cache Checking")
            self._logger.info("="*60)
            
            # Set phase 2 mode to prevent overwriting corrected professors
            self._phase2_mode = True

            # Ensure cache is fresh
            if not DatabaseHelper.load_or_cache_data_with_freshness_check(self):
                self._logger.error("❌ Failed to load fresh database data")
                return False

            processor = ProfessorProcessor(self._create_processor_context())
            
            # Update professor lookup from corrected CSV
            if not processor.update_professor_lookup_from_corrected_csv():
                self._logger.error("❌ Failed to update professor lookup")
                return False
            
            # Update professors with missing boss_names
            processor.update_professors_with_boss_names()
            
            # Process remaining tables with cache checking
            if not self.process_remaining_tables():
                self._logger.error("❌ Failed to process remaining tables")
                return False
            
            # Save all outputs
            self.save_outputs()
            
            # Print summary
            self.print_summary()
            
            self._logger.info("✅ Phase 2 completed successfully!")
            return True

        except Exception as e:
            self._logger.error(f"❌ Phase 2 failed: {e}")
            return False

    def _save_phase1_outputs(self):
        """Save Phase 1 outputs (professors, courses, acad_terms)"""
        # Save new professors (to verify folder for manual correction)
        # Always create the file, even if empty
        df = pd.DataFrame(self.new_professors) if self.new_professors else pd.DataFrame(columns=['id', 'name', 'boss_name', 'afterclass_name', 'original_scraped_name'])
        df.to_csv(os.path.join(self.verify_dir, 'new_professors.csv'), index=False)
        if self.new_professors:
            self._logger.info(f"✅ Saved {len(self.new_professors)} new professors for review")
        else:
            self._logger.info(f"✅ Created empty new_professors.csv (all professors already exist)")
        
        # Save new courses (to verify folder)
        if self.new_courses:
            df = pd.DataFrame(self.new_courses)
            df.to_csv(os.path.join(self.verify_dir, 'new_courses.csv'), index=False)
            self._logger.info(f"✅ Saved {len(self.new_courses)} new courses")
        
        # Save course updates
        if self.update_courses:
            df = pd.DataFrame(self.update_courses)
            df.to_csv(os.path.join(self.output_base, 'update_courses.csv'), index=False)
            self._logger.info(f"✅ Saved {len(self.update_courses)} course updates")
        
        # Save academic terms
        if self.new_acad_terms:
            df = pd.DataFrame(self.new_acad_terms)
            df.to_csv(os.path.join(self.output_base, 'new_acad_term.csv'), index=False)
            self._logger.info(f"✅ Saved {len(self.new_acad_terms)} academic terms")

class TableBuilderCoordinator:
    def __init__(self, config, logger=None, db_connection=None):
        self.config = config
        self._logger = logger or get_logger(__name__)
        self.db_connection = db_connection
        self.builder = TableBuilder(config=self.config, logger=self._logger, db_connection=self.db_connection)

    def run(self):
        import sys
        self.builder.db_connection = DatabaseHelper.create_connection(self.builder._db_adapter, self._logger)
        if not self.builder.db_connection:
            sys.exit(1)

        try:
            self._logger.info("--- Running Phase 1: Professors and Courses ---")
            if not self.builder.run_phase1_professors_and_courses():
                raise Exception("Phase 1 failed.")
            self._logger.info("Phase 1 completed successfully.")

            self._logger.info("--- Running Phase 2: Classes and Timings ---")
            if not self.builder.run_phase2_remaining_tables():
                raise Exception("Phase 2 failed.")
            self._logger.info("Phase 2 completed successfully.")

            self._logger.info("--- Running Phase 3: Bidding Data ---")
            boss_processor = BOSSProcessor(self.builder._create_processor_context())
            if not boss_processor.process():
                raise Exception("Phase 3 failed.")
            self._logger.info("Phase 3 completed successfully.")

            self._logger.info("--- Executing database writes ---")
            self.builder._execute_db_operations()

            self.builder.db_connection.commit()
            self._logger.info("All phases completed successfully and transaction committed!")

        except Exception as e:
            self._logger.error(f"An error occurred: {e}. Rolling back transaction.")
            self.builder.db_connection.rollback()
            sys.exit(1)

        finally:
            if self.builder.db_connection:
                self.builder.db_connection.close()
                self._logger.info("Database connection closed.")