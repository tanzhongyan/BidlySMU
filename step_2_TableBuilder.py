# Import global configuration settings
from config import *

# Import dependencies
import os
import re
import sys
import time
import json
import uuid
import logging
import pandas as pd
import win32com.client as win32
from collections import defaultdict
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from thefuzz import fuzz
from typing import List, Optional, Tuple
from collections import Counter, defaultdict
from dotenv import load_dotenv
from google import genai 
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError

# Import database modules
from util import get_logger
from db_manager import DatabaseManager

# Set up logging
import traceback

# Use centralized logging utility
logger = get_logger("bidlysmu.tablebuilder")

class TableBuilder:
    """Comprehensive table builder for university class management system"""
    
    def __init__(self, input_file: str = 'script_input/raw_data.xlsx'):
        """Initialize TableBuilder with database configuration and caching"""
        self.input_file = input_file
        self.output_base = 'script_output'
        self.verify_dir = os.path.join(self.output_base, 'verify')
        self.cache_dir = 'db_cache'
        
        # Create output directories
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.verify_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        
        # Database connection
        self.connection = None
        
        # Data storage
        self.standalone_data = None
        self.multiple_data = None
        
        # Caches
        self.professors_cache = {}  # name -> professor data
        self.courses_cache = {}     # code -> course data
        self.acad_term_cache = {}   # id -> acad_term data
        self.faculties_cache = {}   # id -> faculty data
        self.faculty_acronym_to_id = {}  # acronym -> faculty_id mapping
        self.professor_lookup = {}  # scraped_name -> database mapping
        
        self.processed_timing_keys = set()
        self.processed_exam_class_ids = set()

        # Output data collectors
        self.new_professors = []
        self.new_courses = []
        self.update_courses = []
        self.new_acad_terms = []
        self.new_classes = []
        self.new_class_timings = []
        self.new_class_exam_timings = []
        self.update_professors = []  # For boss_name updates
        self.update_bid_result = []  # For bid result updates
        
        # Class ID mapping for timing tables
        self.class_id_mapping = {}  # record_key -> class_id
        
        # Courses requiring faculty assignment
        self.courses_needing_faculty = []
        
        # Statistics
        self.stats = {
            'professors_created': 0,
            'professors_updated': 0,
            'courses_created': 0,
            'courses_updated': 0,
            'classes_created': 0,
            'timings_created': 0,
            'exams_created': 0,
            'courses_needing_faculty': 0
        }
        
        # Use the global bidding schedule, assuming the start term is the target
        self.bidding_schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
        
        # Asian surnames database for name normalization
        self.asian_surnames = {
            # Top 100+ common surnames covering Mainland China, Taiwan, HK, and Singapore/Malaysia
            'chinese': [
                'WANG', 'LI', 'ZHANG', 'LIU', 'CHEN', 'YANG', 'HUANG', 'ZHAO', 'WU', 'ZHOU', 'XU', 'SUN', 'MA', 'ZHU', 'HU', 'GUO', 'HE', 'LIN', 'GAO', 'LUO', 
                'CHENG', 'LIANG', 'XIE', 'SONG', 'TANG', 'HAN', 'FENG', 'DENG', 'CAO', 'PENG', 'YUAN', 'SU', 'JIANG', 'JIA', 'LU', 'WEI', 'XIAO', 'YU', 'QIAN', 
                'PAN', 'YAO', 'TAN', 'DU', 'YE', 'TIAN', 'SHI', 'BAI', 'QIN', 'XUE', 'YAN', 'DAI', 'MO', 'CHANG', 'WAN', 'GU', 'ZENG', 'LUO', 'FAN', 'JIN',
                'ONG', 'LIM', 'LEE', 'TEO', 'NG', 'GOH', 'CHUA', 'CHAN', 'KOH', 'ANG', 'YEO', 'SIM', 'CHIA', 'CHONG', 'LAM', 'CHEW', 'TOH', 'LOW', 'SEAH',
                'PEK', 'KWEK', 'QUEK', 'LOH', 'AW', 'CHYE', 'LOK'
            ],
            # Top ~30 Korean surnames
            'korean': [
                'KIM', 'LEE', 'PARK', 'CHOI', 'JEONG', 'KANG', 'CHO', 'YOON', 'JANG', 'LIM', 'HAN', 'OH', 'SEO', 'KWON', 'HWANG', 'SONG', 'JUNG', 'HONG', 
                'AHN', 'GO', 'MOON', 'SON', 'BAE', 'BAEK', 'HEO', 'NAM'
            ],
            # Top ~20 Vietnamese surnames
            'vietnamese': [
                'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG', 'PHAN', 'VU', 'VO', 'DANG', 'BUI', 'DO', 'HO', 'NGO', 'DUONG', 'LY'
            ],
            # Top ~30 Indian surnames from various regions
            'indian': [
                'SHARMA', 'SINGH', 'KUMAR', 'GUPTA', 'PATEL', 'KHAN', 'REDDY', 'YADAV', 'DAS', 'JAIN', 'RAO', 'MEHTA', 'CHOPRA', 'KAPOOR', 'MALHOTRA',
                'AGGARWAL', 'JOSHI', 'MISHRA', 'TRIPATHI', 'PANDEY', 'NAIR', 'MENON', 'PILLAI', 'IYER', 'MUKHERJEE', 'BANERJEE', 'CHATTERJEE'
            ],
            # Top ~20 Japanese surnames
            'japanese': [
                'SATO', 'SUZUKI', 'TAKAHASHI', 'TANAKA', 'WATANABE', 'ITO', 'YAMAMOTO', 'NAKAMURA', 'KOBAYASHI', 'SAITO', 'KATO', 'YOSHIDA', 'YAMADA'
            ]
        }
        self.all_asian_surnames = set()
        for surnames in self.asian_surnames.values():
            self.all_asian_surnames.update(surnames)
        
        # Western given names
        self.western_given_names = {
            'AARON', 'ADAM', 'ADRIAN', 'ALAN', 'ALBERT', 'ALEX', 'ALEXANDER', 'ALFRED', 'ALVIN', 'AMANDA', 'AMY', 'ANDREA', 'ANDREW', 'ANGELA', 'ANNA', 'ANTHONY', 'ARTHUR', 'AUDREY',
            'BEN', 'BENJAMIN', 'BERNARD', 'BETTY', 'BILLY', 'BOB', 'BOWEN', 'BRANDON', 'BRENDA', 'BRIAN', 'BRYAN', 'BRUCE',
            'CARL', 'CAROL', 'CATHERINE', 'CHARLES', 'CHRIS', 'CHRISTIAN', 'CHRISTINA', 'CHRISTINE', 'CHRISTOPHER', 'COLIN', 'CRAIG', 'CRYS',
            'DANIEL', 'DANNY', 'DARREN', 'DAVID', 'DEBORAH', 'DENISE', 'DENNIS', 'DEREK', 'DIANA', 'DONALD', 'DOUGLAS',
            'EDWARD', 'EDWIN', 'ELAINE', 'ELIZABETH', 'EMILY', 'ERIC', 'EUGENE', 'EVELYN',
            'FELIX', 'FRANCIS', 'FRANK',
            'GABRIEL', 'GARY', 'GEOFFREY', 'GEORGE', 'GERALD', 'GLORIA', 'GORDON', 'GRACE', 'GRAHAM', 'GREGORY',
            'HANNAH', 'HARRY', 'HELEN', 'HENRY', 'HOWARD',
            'IAN', 'IVAN',
            'JACK', 'JACOB', 'JAMES', 'JANE', 'JANET', 'JASON', 'JEAN', 'JEFFREY', 'JENNIFER', 'JEREMY', 'JERRY', 'JESSICA', 'JIM', 'JOAN', 'JOE', 'JOHN', 'JONATHAN', 'JOSEPH', 'JOSHUA', 'JOYCE', 'JUDY', 'JULIA', 'JULIE', 'JUSTIN',
            'KAREN', 'KATHERINE', 'KATHY', 'KEITH', 'KELLY', 'KELVIN', 'KENNETH', 'KEVIN', 'KIMBERLY',
            'LARRY', 'LAURA', 'LAWRENCE', 'LEO', 'LEONARD', 'LINDA', 'LISA',
            'MARGARET', 'MARIA', 'MARK', 'MARTIN', 'MARY', 'MATTHEW', 'MEGAN', 'MELISSA', 'MICHAEL', 'MICHELLE', 'MIKE',
            'NANCY', 'NATHAN', 'NEHA', 'NICHOLAS', 'NICOLE',
            'OLIVER', 'OLIVIA',
            'PAMELA', 'PATRICIA', 'PATRICK', 'PAUL', 'PETER', 'PHILIP',
            'RACHEL', 'RAYMOND', 'REBECCA', 'RICHARD', 'ROBERT', 'ROGER', 'RONALD', 'ROY', 'RUSSELL', 'RYAN',
            'SAM', 'SAMUEL', 'SANDRA', 'SARAH', 'SCOTT', 'SEAN', 'SHARON', 'SOPHIA', 'STANLEY', 'STEPHANIE', 'STEPHEN', 'STEVEN', 'SUSAN',
            'TERENCE', 'TERRY', 'THERESA', 'THOMAS', 'TIMOTHY', 'TONY',
            'VALERIE', 'VICTOR', 'VINCENT', 'VIRGINIA',
            'WALTER', 'WAYNE', 'WENDY', 'WILLIAM', 'WILLIE'
        }

        # Keywords for patronymic names (Malay, Indian, etc.)
        self.patronymic_keywords = {'BIN', 'BINTE', 'S/O', 'D/O'}

        # European surname particles
        self.surname_particles = {'DE', 'DI', 'DA', 'VAN', 'VON', 'LA', 'LE', 'DEL', 'DELLA'}       

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
        
        # Load professor lookup if available
        self.load_professor_lookup_csv()

        # LLM Configuration
        logger.info("🔧 Initializing LLM configuration...")
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
        if os.getenv('GEMINI_API_KEY'):
            try:
                # Create client with API key
                self.llm_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
                
                # Test the client to make sure it works
                # You can remove this test call if you want
                test_response = self.llm_client.models.generate_content(
                    model=self.llm_model_name,
                    contents="Test"
                )
                
                logger.info(f"✅ Gemini client '{self.llm_model_name}' configured successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Could not pre-configure Gemini client: {e}")
                self.llm_client = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY not found. LLM normalization will be skipped.")

        # Initialize database connection with pooling
        self.engine = None
        self.db_connection = None
        self.db_manager = None  # DatabaseManager instance for connection pooling

    def connect_database(self):
        """Connect to PostgreSQL database using DatabaseManager with connection pooling."""
        try:
            # Create database URL
            db_url = (
                f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            
            # Initialize DatabaseManager with connection pooling
            self.db_manager = DatabaseManager(db_url, logger_name="bidlysmu.db")
            
            # Get engine and connection with retry logic
            self.engine = self.db_manager.get_engine()
            self.db_connection = self.db_manager.get_connection()
            
            logger.info("✅ Database connection with pooling established")
            
            # Test connection
            if self.db_manager.test_connection():
                logger.info("✅ Database connection test successful")
                return True
            else:
                logger.error("❌ Database connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}", exc_info=True)
            return False

    def load_or_cache_data(self):
        """Load data from cache or database"""
        # Try loading from cache first
        if self._load_from_cache():
            logger.info("✅ Loaded data from cache")
            return True
        
        # Connect to database and download
        if not self.connect_database():
            return False
        
        try:
            self._download_and_cache_data()
            logger.info("✅ Downloaded and cached data from database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download data: {e}")
            return False

    def _download_and_cache_data(self):
        """Download data from database tables and cache them locally."""
        try:
            tables_to_cache = [
                "professors", "courses", "acad_term", "faculties", 
                "classes", "class_timing", "class_exam_timing", 
                "class_availability", "bid_window", "bid_result", "bid_prediction"
            ]
            
            for table_name in tables_to_cache:
                logger.info(f"⬇️ Caching table: {table_name}")
                query = f"SELECT * FROM {table_name}"
                # Use the new SQLAlchemy engine to run the query
                df = pd.read_sql_query(query, self.engine)
                df.to_pickle(os.path.join(self.cache_dir, f'{table_name}_cache.pkl'))
                
            logger.info("✅ Downloaded all tables from database and cached locally")
            self._load_from_cache()
            
        except Exception as e:
            logger.error(f"❌ Failed to download and cache data for table '{table_name}': {e}")
            raise

    def _load_from_cache(self) -> bool:
        """
        Load cached data from files, with robust validation of the professor lookup against the database cache.
        Professor validation only runs during Phase 1.
        """
        try:
            cache_files = {
                'professors': os.path.join(self.cache_dir, 'professors_cache.pkl'),
                'courses': os.path.join(self.cache_dir, 'courses_cache.pkl'),
                'acad_terms': os.path.join(self.cache_dir, 'acad_term_cache.pkl'),
                'faculties': os.path.join(self.cache_dir, 'faculties_cache.pkl'),
                'bid_result': os.path.join(self.cache_dir, 'bid_result_cache.pkl'),
                'bid_window': os.path.join(self.cache_dir, 'bid_window_cache.pkl'),
                'class_availability': os.path.join(self.cache_dir, 'class_availability_cache.pkl'),
                'class_exam_timing': os.path.join(self.cache_dir, 'class_exam_timing_cache.pkl'),
                'class_timing': os.path.join(self.cache_dir, 'class_timing_cache.pkl'),
                'classes': os.path.join(self.cache_dir, 'classes_cache.pkl')
            }
            
            if not all(os.path.exists(f) for f in cache_files.values()):
                logger.warning("⚠️ Not all cache files found. Need to download from database.")
                return False

            # Load professors data first
            professors_df = pd.read_pickle(cache_files['professors'])
            
            # Check if this is Phase 1 by looking at the call stack or phase indicator
            is_phase1 = (hasattr(self, '_phase2_mode') and not self._phase2_mode) or not hasattr(self, '_phase2_mode')
            
            if is_phase1:
                # --- Professor Lookup Synchronization (Phase 1 only) ---
                logger.info("🔄 Phase 1: Synchronizing professor lookup with database cache...")
                
                database_professors = {}
                all_database_aliases = {}

                for _, row in professors_df.iterrows():
                    professor_data = row.to_dict()
                    professor_id = str(row.get('id'))
                    professor_name = str(row.get('name', '')).strip()
                    
                    database_professors[professor_id] = professor_data
                    
                    # Add the professor's actual name as an alias
                    if professor_name:
                        all_database_aliases[professor_name.upper()] = professor_id

                    # Handle boss_aliases - robust parsing
                    aliases_list = self._parse_boss_aliases(row.get('boss_aliases'))
                    for alias in aliases_list:
                        if alias and str(alias).strip():
                            all_database_aliases[str(alias).upper()] = professor_id

                logger.info(f"📚 Loaded {len(database_professors)} professors from cache")
                logger.info(f"📚 Found {len(all_database_aliases)} total aliases (including names)")

                # Load and validate professor_lookup.csv
                lookup_file = 'script_input/professor_lookup.csv'
                validated_professor_lookup = {}
                csv_entries_removed = 0
                csv_entries_corrected = 0
                csv_entries_added = 0

                if os.path.exists(lookup_file):
                    try:
                        lookup_df = pd.read_csv(lookup_file)
                        for _, row in lookup_df.iterrows():
                            boss_name = str(row.get('boss_name', '')).strip()
                            afterclass_name = str(row.get('afterclass_name', '')).strip()
                            database_id = str(row.get('database_id', '')).strip()
                            
                            if not boss_name or not database_id: continue
                                
                            boss_name_key = boss_name.upper()

                            # CRITICAL: Validate database_id exists in database
                            if database_id not in database_professors:
                                logger.warning(f"❌ Invalid database_id in lookup: '{boss_name}' references non-existent ID {database_id}. Removing.")
                                csv_entries_removed += 1
                                continue

                            db_professor = database_professors[database_id]
                            db_name = str(db_professor.get('name', '')).strip()
                            
                            # Correct afterclass_name if it differs from database
                            if afterclass_name != db_name:
                                logger.warning(f"✏️ Correcting lookup entry for '{boss_name}': Name mismatch (CSV: '{afterclass_name}' vs DB: '{db_name}'). Using DB name.")
                                afterclass_name = db_name
                                csv_entries_corrected += 1
                            
                            validated_professor_lookup[boss_name_key] = {
                                'database_id': database_id,
                                'boss_name': boss_name,
                                'afterclass_name': afterclass_name
                            }
                    except Exception as e:
                        logger.error(f"❌ Error reading professor_lookup.csv: {e}")
                else:
                    logger.info("📋 professor_lookup.csv not found. Creating from database.")

                # Add missing database aliases to lookup (bidirectional sync)
                for alias_key, professor_id in all_database_aliases.items():
                    if alias_key not in validated_professor_lookup:
                        db_professor = database_professors[professor_id]
                        db_name = str(db_professor.get('name', '')).strip()
                        
                        validated_professor_lookup[alias_key] = {
                            'database_id': str(professor_id),
                            'boss_name': alias_key,
                            'afterclass_name': db_name
                        }
                        csv_entries_added += 1
                        # Only log for non-name aliases to reduce noise
                        if alias_key != db_name.upper():
                            logger.info(f"➕ Added missing DB alias to lookup: '{alias_key}' -> '{db_name}' (ID: {professor_id})")

                self.professor_lookup = validated_professor_lookup
                
                logger.info("✅ Phase 1 Professor lookup synchronization complete:")
                logger.info(f"  - Entries validated: {len(validated_professor_lookup)}")
                logger.info(f"  - Corrected entries: {csv_entries_corrected}")
                logger.info(f"  - Added DB entries: {csv_entries_added}")
                logger.info(f"  - Removed invalid entries: {csv_entries_removed}")

                # Save corrected lookup back to file
                corrected_lookup_data = sorted(list(self.professor_lookup.values()), key=lambda x: x['boss_name'])
                for item in corrected_lookup_data:
                    item['method'] = 'validated'
                
                corrected_df = pd.DataFrame(corrected_lookup_data)
                corrected_df.to_csv(lookup_file, index=False, columns=['boss_name', 'afterclass_name', 'database_id', 'method'])
                logger.info(f"💾 Updated '{lookup_file}' with synchronized data.")

                # Build professors_cache for lookups
                self.professors_cache = {}
                for lookup_data in self.professor_lookup.values():
                    db_id = lookup_data['database_id']
                    boss_name_key = lookup_data['boss_name'].upper()
                    if db_id in database_professors:
                        self.professors_cache[boss_name_key] = database_professors[db_id]
            else:
                # Phase 2/3: Simple loading without validation
                logger.info("🔄 Phase 2/3: Loading professor data without validation...")
                self.professors_cache = {}
                for _, row in professors_df.iterrows(): 
                    # Simple loading for non-Phase 1
                    professor_data = row.to_dict()
                    professor_name = str(row.get('name', '')).strip().upper()
                    if professor_name:
                        self.professors_cache[professor_name] = professor_data

            # --- Load Remaining Caches (all phases) ---
            courses_df = pd.read_pickle(cache_files['courses'])
            for _, row in courses_df.iterrows(): self.courses_cache[row['code']] = row.to_dict()
            
            acad_terms_df = pd.read_pickle(cache_files['acad_terms'])
            for _, row in acad_terms_df.iterrows(): self.acad_term_cache[row['id']] = row.to_dict()
                
            faculties_df = pd.read_pickle(cache_files['faculties'])
            for _, row in faculties_df.iterrows():
                self.faculties_cache[row['id']] = row.to_dict()
                self.faculty_acronym_to_id[row['acronym'].upper()] = row['id']
                
            bid_window_df = pd.read_pickle(cache_files['bid_window'])
            if not bid_window_df.empty:
                self.bid_window_id_counter = bid_window_df['id'].max() + 1
                for _, row in bid_window_df.iterrows():
                    self.bid_window_cache[(row['acad_term_id'], row['round'], row['window'])] = row['id']
            else:
                self.bid_window_id_counter = 1
                self.bid_window_cache = {}

            logger.info("✅ All cache files loaded successfully.")
            if is_phase1:
                logger.info(f"  - Professor lookup entries: {len(self.professor_lookup)} entries")
            logger.info(f"  - Professors cache: {len(self.professors_cache)} entries")
            return True

        except Exception as e:
            logger.error(f"❌ Cache loading error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_raw_data(self):
        """
        Load raw data WITHOUT applying global filtering.
        Each processing function will apply its own appropriate filtering.
        """
        try:
            logger.info(f"📂 Loading raw data from {self.input_file}")
            
            full_standalone_df = pd.read_excel(self.input_file, sheet_name='standalone')
            full_multiple_df = pd.read_excel(self.input_file, sheet_name='multiple')
            
            logger.info(f"✅ Loaded {len(full_standalone_df)} total standalone and {len(full_multiple_df)} total multiple records.")
            
            # === REMOVED GLOBAL FILTERING ===
            # Store the full data without filtering - each processing function will filter as needed
            self.standalone_data = full_standalone_df
            self.multiple_data = full_multiple_df
            
            # Log available bidding windows for debugging
            if 'bidding_window' in full_standalone_df.columns:
                available_windows = full_standalone_df['bidding_window'].dropna().unique()
                logger.info(f"📊 Available bidding windows in data: {sorted(available_windows)}")
            
            # Log available academic terms for debugging  
            if 'acad_term_id' in full_standalone_df.columns:
                available_terms = full_standalone_df['acad_term_id'].dropna().unique()
                logger.info(f"📊 Available academic terms in data: {sorted(available_terms)}")

            # Create optimized lookup for the multiple_data
            self.multiple_lookup = defaultdict(list)
            for _, row in self.multiple_data.iterrows():
                key = row.get('record_key')
                if pd.notna(key):
                    self.multiple_lookup[key].append(row)
            
            logger.info(f"✅ Created optimized lookup for {len(self.multiple_lookup)} record keys from unfiltered data.")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load raw data: {e}")
            return False

    def upload_csvs_to_database(self):
        """
        Upload CSV files to database within a transaction.
        
        Reads CSV files from script_output directory and uploads them to the database.
        All operations are wrapped in a transaction - if any upload fails, all changes
        are rolled back. This allows retrying the upload without regenerating CSVs.
        
        Returns:
            bool: True if all uploads successful, False otherwise
        """
        logger.info("📤 Uploading CSV files to database...")
        
        # Define CSV to table mapping
        # Format: 'csv_filename': ('table_name', 'operation', {'option': 'value'})
        csv_operations = [
            # INSERT operations (simple bulk inserts)
            ('new_professors.csv', 'insert', {'table': 'professors'}),
            ('new_courses.csv', 'insert', {'table': 'courses'}),
            ('new_acad_term.csv', 'insert', {'table': 'acad_term'}),
            ('new_classes.csv', 'insert', {'table': 'classes'}),
            ('new_class_timing.csv', 'insert', {'table': 'class_timing'}),
            ('new_class_exam_timing.csv', 'insert', {'table': 'class_exam_timing'}),
            ('new_bid_windows.csv', 'insert', {'table': 'bid_window'}),
            
            # UPDATE operations (update existing records by id)
            ('update_professor.csv', 'update', {'table': 'professors', 'id_column': 'id', 'array_columns': ['boss_aliases']}),
            ('update_courses.csv', 'update', {'table': 'courses', 'id_column': 'id'}),
            ('update_classes.csv', 'update', {'table': 'classes', 'id_column': 'id'}),
            
            # UPSERT operations (INSERT ON CONFLICT)
            ('class_availability.csv', 'upsert', {
                'table': 'class_availability', 
                'index_elements': ['class_id', 'bid_window_id']
            }),
            ('bid_result.csv', 'upsert', {
                'table': 'bid_result',
                'index_elements': ['bid_window_id', 'class_id']
            }),
        ]
        
        try:
            # Use SQLAlchemy transaction context manager for automatic commit/rollback
            with self.db_connection.begin():
                logger.info("🔄 Starting database transaction...")
                
                for csv_filename, operation, options in csv_operations:
                    csv_path = os.path.join(self.output_base, csv_filename)
                    
                    # Skip if CSV doesn't exist
                    if not os.path.exists(csv_path):
                        logger.debug(f"⏭️ Skipping {csv_filename} - file not found")
                        continue
                    
                    # Read CSV
                    df = pd.read_csv(csv_path)
                    if df.empty:
                        logger.debug(f"⏭️ Skipping {csv_filename} - empty file")
                        continue
                    
                    table_name = options['table']
                    logger.info(f"📥 {operation.upper()}: {len(df)} records from {csv_filename} to {table_name}")
                    
                    if operation == 'insert':
                        # Simple bulk insert
                        df.to_sql(
                            table_name,
                            self.db_connection,
                            if_exists='append',
                            index=False,
                            chunksize=1000,
                            method='multi'  # Multi-row insert for efficiency
                        )
                        
                    elif operation == 'update':
                        # Row-by-row update (for smaller update files)
                        id_column = options.get('id_column', 'id')
                        array_columns = options.get('array_columns', [])
                        
                        records = df.where(pd.notna(df), None).to_dict('records')
                        for record in records:
                            record_id = record.pop(id_column)
                            
                            # Handle array columns (like boss_aliases)
                            for col in array_columns:
                                if col in record and isinstance(record[col], str):
                                    record[col] = json.loads(record[col])
                            
                            # Build SET clause
                            set_clause = ', '.join([f"{k} = :{k}" for k in record.keys()])
                            stmt = text(f"UPDATE {table_name} SET {set_clause} WHERE {id_column} = :{id_column}_value")
                            
                            # Add id to params with unique key
                            params = {**record, f"{id_column}_value": record_id}
                            self.db_connection.execute(stmt, params)
                            
                    elif operation == 'upsert':
                        # INSERT ON CONFLICT
                        records = df.where(pd.notna(df), None).to_dict('records')
                        index_elements = options['index_elements']
                        
                        # Build insert statement
                        stmt = pg_insert(table_name).values(records)
                        
                        # Determine which columns to update on conflict (all except index elements)
                        update_cols = {
                            col.name: col 
                            for col in stmt.excluded 
                            if col.name not in index_elements
                        }
                        
                        # Add ON CONFLICT clause
                        stmt = stmt.on_conflict_do_update(
                            index_elements=index_elements,
                            set_=update_cols
                        )
                        
                        self.db_connection.execute(stmt)
                
                logger.info("✅ All CSV files uploaded successfully - committing transaction")
                # Transaction commits automatically when exiting context
                
            logger.info("🎉 Database upload complete!")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database upload failed: {e}", exc_info=True)
            # Transaction automatically rolls back on exception
            logger.info("🔄 Transaction rolled back - CSV files preserved for retry")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during upload: {e}", exc_info=True)
            return False

    def _execute_db_operations(self):
        """
        Legacy method for backward compatibility.
        Now calls upload_csvs_to_database() which reads from CSVs.
        """
        return self.upload_csvs_to_database()

    def _normalize_professor_name_fallback(self, name: str) -> Tuple[str, str]:
        """
        (Fallback Method) Normalizes professor names using a definitive, rule-based system.
        """
        if name is None or pd.isna(name) or not str(name).strip():
            return "UNKNOWN", "Unknown"

        # --- Step 1: Aggressive Preprocessing ---
        name_str = str(name).strip().replace("’", "'")
        name_str = re.sub(r'\s*\(.*\)\s*', ' ', name_str).strip()
        # Remove all middle initials (e.g., "S.", "H.", "H H", "S") to standardize names
        # This looks for standalone single letters, with or without a dot.
        words = name_str.split()
        words_no_initials = [word for word in words if not (len(word) == 1 and word.isalpha()) and not (len(word) == 2 and word.endswith('.'))]
        name_str = ' '.join(words_no_initials)

        boss_name = name_str.upper()

        # --- Step 2: Handle High-Certainty Delimiters ---
        if ',' in name_str:
            parts = [p.strip() for p in name_str.split(',')]
            words = ' '.join(parts).split()
            surname_to_check = words[0].upper()
            if len(parts) == 2:
                words_after_comma = parts[1].split()
                if words_after_comma and words_after_comma[0].upper() in self.all_asian_surnames:
                    surname_to_check = words_after_comma[0].upper()
                else:
                    words_before_comma = parts[0].split()
                    if words_before_comma and words_before_comma[0].upper() in self.all_asian_surnames:
                        surname_to_check = words_before_comma[0].upper()
            afterclass_parts = [word.capitalize() for word in words]
            for i, word in enumerate(words):
                if word.upper() == surname_to_check:
                    afterclass_parts[i] = word.upper()
            return boss_name, ' '.join(afterclass_parts)

        words = name_str.split()
        if not words: return boss_name, "Unknown"
        if len(words) == 1: return boss_name, words[0].capitalize()

        for i, word in enumerate(words):
            if word.upper() in self.patronymic_keywords and i < len(words) - 1:
                surname_index = i + 1
                afterclass_parts = [w.capitalize() for w in words]
                afterclass_parts[i] = word.lower()
                afterclass_parts[surname_index] = words[surname_index].upper()
                return boss_name, ' '.join(afterclass_parts)

        # --- Step 3: Definitive Rule-Based Surname Identification ---
        surname_index = -1
        
        # Rule 1 (Fixes "Middle Surname"): If the name starts with a Western/Indian given name,
        # actively search for the first known Asian/Indian surname that follows.
        if words[0].upper() in self.western_given_names:
            for i in range(1, len(words)):
                if words[i].upper() in self.all_asian_surnames:
                    surname_index = i
                    break
        
        # Rule 2 (Fixes "Surname-First Western"): If a name contains a Western given name but
        # does NOT start with it, and the first word is not an Asian surname, assume the first word is the surname.
        elif any(w.upper() in self.western_given_names for w in words) and words[0].upper() not in self.all_asian_surnames:
            surname_index = 0

        # Rule 3: If neither of the complex cases above apply, check if the name starts with a known Asian surname.
        # This handles the most common SURNAME-first pattern.
        elif words[0].upper() in self.all_asian_surnames:
            surname_index = 0

        # Rule 4 (Fallback): If no specific pattern has been matched, default to the last word.
        if surname_index == -1:
            surname_index = len(words) - 1
            
        # Post-processing for European particles
        afterclass_parts = [word.capitalize() for word in words]
        if surname_index > 0 and words[surname_index-1].upper() in self.surname_particles:
             afterclass_parts[surname_index-1] = words[surname_index-1].upper()

        afterclass_parts[surname_index] = words[surname_index].upper()
        
        return boss_name, ' '.join(afterclass_parts)
    def resolve_professor_email(self, professor_name):
        """Resolve professor email using Outlook contacts"""
        try:
            # Initialize Outlook
            outlook = win32.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            
            # Try exact resolver first
            recipient = namespace.CreateRecipient(professor_name)
            if recipient.Resolve():
                # Try to get SMTP address
                address_entry = recipient.AddressEntry
                
                # Try Exchange user
                try:
                    exchange_user = address_entry.GetExchangeUser()
                    if exchange_user and exchange_user.PrimarySmtpAddress:
                        return exchange_user.PrimarySmtpAddress.lower()
                except:
                    pass
                
                # Try Exchange distribution list
                try:
                    exchange_dl = address_entry.GetExchangeDistributionList()
                    if exchange_dl and exchange_dl.PrimarySmtpAddress:
                        return exchange_dl.PrimarySmtpAddress.lower()
                except:
                    pass
                
                # Try PR_SMTP_ADDRESS property
                try:
                    property_accessor = address_entry.PropertyAccessor
                    smtp_addr = property_accessor.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x39FE001E")
                    if smtp_addr:
                        return smtp_addr.lower()
                except:
                    pass
                
                # Fallback: regex search in Address field
                try:
                    address = getattr(address_entry, "Address", "") or ""
                    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", address)
                    if match:
                        return match.group(0).lower()
                except:
                    pass
            
            # If exact resolve fails, try contacts search
            contacts_folder = namespace.GetDefaultFolder(10)  # olFolderContacts
            tokens = [t.lower() for t in professor_name.split() if t]
            
            for item in contacts_folder.Items:
                try:
                    full_name = (item.FullName or "").lower()
                    if all(token in full_name for token in tokens):
                        # Try the three standard email slots
                        for field in ("Email1Address", "Email2Address", "Email3Address"):
                            addr = getattr(item, field, "") or ""
                            if addr and "@" in addr:
                                return addr.lower()
                except:
                    continue
            
            # If no email found, return default
            return 'enquiry@smu.edu.sg'
            
        except Exception as e:
            logger.warning(f"Email resolution failed for {professor_name}: {e}")
            return 'enquiry@smu.edu.sg'
        
    def process_professors(self):
        """
        Orchestrates the processing of professors: extraction, normalization, and creation.
        """
        logger.info("👥 Processing professors...")
        
        # Step 1: Extract unique names and their variations from the data source.
        unique_professors, professor_variations = self._extract_unique_professors()

        # Step 2: Filter out existing professors to find only new names.
        new_professors_to_normalize = []
        for prof_name in unique_professors:
            # A professor is considered "new" if they are not in the primary lookup.
            if prof_name.upper() not in self.professor_lookup:
                new_professors_to_normalize.append(prof_name)

        logger.info(f"Found {len(unique_professors)} unique names. "
                    f"Identified {len(new_professors_to_normalize)} as new and requiring normalization.")
        
        # Step 2b: Normalize only the new names using the LLM-first, fallback-second approach.
        normalized_map = self._normalize_professors_batch(new_professors_to_normalize)
        
        # Add a fallback for existing professors to ensure they are still processed later
        for prof_name in unique_professors:
            if prof_name not in normalized_map:
                # If an existing professor wasn't normalized, add them to the map using the fallback
                # to ensure they are processed correctly in the steps that follow.
                normalized_map[prof_name] = self._normalize_professor_name_fallback(prof_name)

        if not normalized_map:
            logger.info("No professor names were normalized. Aborting professor processing.")
            return

        # Step 3: Check cache, fuzzy match, and create new professor records.
        email_to_professor = {}
        for boss_name_key, prof_data in self.professors_cache.items():
            if 'email' in prof_data and prof_data['email'] and prof_data['email'].lower() != 'enquiry@smu.edu.sg':
                email_to_professor[prof_data['email'].lower()] = prof_data

        fuzzy_matched_professors = []
        
        for prof_name in unique_professors:
            try:
                boss_name, afterclass_name = normalized_map[prof_name]
                
                # --- The logic below is IDENTICAL to your original script's Step 3 ---
                if hasattr(self, 'professor_lookup') and prof_name.upper() in self.professor_lookup:
                    continue
                if hasattr(self, 'professor_lookup') and boss_name.upper() in self.professor_lookup:
                    self.professor_lookup[prof_name.upper()] = self.professor_lookup[boss_name.upper()]
                    continue
                
                if hasattr(self, 'professor_lookup'):
                    found_partial_match = False
                    for lookup_boss_name, lookup_data in self.professor_lookup.items():
                        prof_words = set(prof_name.upper().split())
                        lookup_words = set(lookup_boss_name.split())
                        
                        if prof_words.issubset(lookup_words) and len(prof_words) >= 2:
                            self.professor_lookup[prof_name.upper()] = lookup_data
                            found_partial_match = True
                            break
                    if found_partial_match:
                        continue
                
                if boss_name in self.professors_cache:
                    if not hasattr(self, 'professor_lookup'):
                        self.professor_lookup = {}
                    self.professor_lookup[prof_name.upper()] = {
                        'database_id': self.professors_cache[boss_name]['id'],
                        'boss_name': boss_name,
                        'afterclass_name': self.professors_cache[boss_name].get('name', afterclass_name)
                    }
                    continue
                
                fuzzy_match_found = False
                normalized_prof = ' '.join(str(prof_name).replace(',', ' ').split()).upper()
                
                for cached_name, cached_prof in self.professors_cache.items():
                    if cached_name is None:
                        continue
                    cached_normalized = ' '.join(str(cached_name).replace(',', ' ').split()).upper()
                    if normalized_prof == cached_normalized:
                        if not hasattr(self, 'professor_lookup'):
                            self.professor_lookup = {}
                        self.professor_lookup[prof_name.upper()] = {
                            'database_id': cached_prof['id'],
                            'boss_name': cached_prof.get('boss_name', cached_prof['name'].upper()),
                            'afterclass_name': cached_prof.get('name', afterclass_name)
                        }
                        fuzzy_match_found = True
                        break
                if fuzzy_match_found:
                    continue
                
                # This is the block that was previously a placeholder
                for new_prof in self.new_professors:
                    if 'boss_aliases' in new_prof:
                        try:
                            boss_aliases = json.loads(new_prof['boss_aliases'])
                            if isinstance(boss_aliases, list) and boss_aliases:
                                new_normalized = ' '.join(boss_aliases[0].replace(',', ' ').split()).upper()
                            else:
                                new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()
                        except (json.JSONDecodeError, TypeError):
                            new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()
                    else:
                        new_normalized = ' '.join(new_prof.get('afterclass_name', '').replace(',', ' ').split()).upper()

                    if normalized_prof == new_normalized:
                        if not hasattr(self, 'professor_lookup'):
                            self.professor_lookup = {}
                        self.professor_lookup[prof_name.upper()] = {
                            'database_id': new_prof['id'],
                            'boss_name': boss_name,
                            'afterclass_name': new_prof['afterclass_name']
                        }
                        fuzzy_match_found = True
                        break
                
                if fuzzy_match_found:
                    continue
                
                if hasattr(self, 'professor_lookup'):
                    best_fuzzy_match = None
                    best_fuzzy_score = 0
                    FUZZY_MATCH_THRESHOLD = 90
                    for lookup_boss_name, lookup_data in self.professor_lookup.items():                        
                        score = self._calculate_fuzzy_score(prof_name, lookup_boss_name)
                        if score > best_fuzzy_score:
                            best_fuzzy_match = lookup_data
                            best_fuzzy_score = score
                    
                    if best_fuzzy_match and best_fuzzy_score >= FUZZY_MATCH_THRESHOLD:
                        fuzzy_matched_professors.append({
                            'boss_aliases': f'["{prof_name.upper()}"]',
                            'afterclass_name': best_fuzzy_match.get('afterclass_name', prof_name),
                            'database_id': best_fuzzy_match['database_id'],
                            'method': 'fuzzy_match',
                            'confidence_score': f"{best_fuzzy_score:.2f}"
                        })
                        if not hasattr(self, 'professor_lookup'):
                            self.professor_lookup = {}
                        self.professor_lookup[prof_name.upper()] = best_fuzzy_match
                        continue
                
                resolved_email = self.resolve_professor_email(afterclass_name)
                
                if (resolved_email and 
                    resolved_email.lower() != 'enquiry@smu.edu.sg' and 
                    resolved_email.lower() in email_to_professor):
                    existing_prof = email_to_professor[resolved_email.lower()]
                    if not hasattr(self, 'professor_lookup'):
                        self.professor_lookup = {}
                    self.professor_lookup[prof_name.upper()] = {
                        'database_id': existing_prof['id'],
                        'boss_name': boss_name,
                        'afterclass_name': existing_prof.get('name', afterclass_name)
                    }
                    continue
                
                self._create_new_professor(prof_name, professor_variations, email_to_professor)

            except Exception as e:
                logger.error(f"❌ Error processing professor '{prof_name}': {e}")
                continue
        
        if fuzzy_matched_professors:
            fuzzy_df = pd.DataFrame(fuzzy_matched_professors)
            fuzzy_path = os.path.join(self.verify_dir, 'fuzzy_matched_professors.csv')
            fuzzy_df.to_csv(fuzzy_path, index=False)
            logger.info(f"🔍 Saved {len(fuzzy_matched_professors)} fuzzy matched professors for validation")
        
        logger.info(f"✅ Created {self.stats['professors_created']} new professors")

    def _calculate_fuzzy_score(self, new_name: str, known_alias: str) -> float:
        """
        Calculates a fuzzy match score using a hybrid strategy that prioritizes
        ordered matches and handles permutations.
        """
        if not new_name or not known_alias:
            return 0.0
        
        # Clean and normalize names for consistent comparison
        new_name_clean = ' '.join(str(new_name).upper().replace(',', ' ').split())
        known_alias_clean = ' '.join(str(known_alias).upper().replace(',', ' ').split())
        
        # --- Layer 1: High-Precision Substring Check ---
        # This is the most important check. It handles short forms like 
        # 'WARREN B. CHIK' being perfectly contained within 'KAM WAI WARREN BARTHOLOMEW CHIK'.
        # This check is fast and respects word order.
        if new_name_clean in known_alias_clean or known_alias_clean in new_name_clean:
            # We return a very high score, but not 100, to indicate a strong partial match.
            return 95

        # --- Layer 2: Hybrid Fuzzy Logic ---
        # If it's not a direct substring, we use two different fuzzy algorithms.
        
        # a) Partial Ratio: Good for ordered, partial matches.
        # This respects word order. A jumbled name will get a LOW score here.
        # e.g., 'KAM WAI CHIK' vs 'KAM WAI WARREN CHIK' will score high.
        partial_score = fuzz.partial_ratio(new_name_clean, known_alias_clean)
        
        # b) Token Set Ratio: Good for permutations and names with extra/missing words.
        # This handles cases like 'RACHEL TAN YEN JUN' vs 'RACHEL TAN'.
        token_set_score = fuzz.token_set_ratio(new_name_clean, known_alias_clean)
        
        # We take the best score from the two fuzzy methods. This gives us the
        # flexibility to catch both ordered variations and unordered permutations.
        return max(partial_score, token_set_score)

    def process_courses(self):
        """
        Processes courses from the standalone sheet. It correctly identifies if a course
        is new or if an existing course needs to be updated.
        """
        logger.info("📚 Processing courses with robust CREATE vs. UPDATE logic...")
        
        # Use a set to only process each unique course code once from the input file.
        processed_course_codes_in_run = set()

        for idx, row in self.standalone_data.iterrows():
            course_code = row.get('course_code')
            if pd.isna(course_code) or course_code in processed_course_codes_in_run:
                continue
            
            processed_course_codes_in_run.add(course_code)

            # Check if the course already exists in our database cache
            if course_code in self.courses_cache:
                # --- UPDATE LOGIC ---
                existing_course = self.courses_cache[course_code]
                update_record = {'id': existing_course['id'], 'code': course_code}
                
                # Define fields to check for potential updates
                field_mapping = {
                    'name': 'course_name', 'description': 'course_description',
                    'credit_units': 'credit_units', 'course_area': 'course_area',
                    'enrolment_requirements': 'enrolment_requirements'
                }

                if self.needs_update(existing_course, row, field_mapping):
                    # self.needs_update will populate the update_record
                    for db_field, raw_field in field_mapping.items():
                        new_value = row.get(raw_field)
                        if pd.notna(new_value) and str(new_value) != str(existing_course.get(db_field)):
                            update_record[db_field] = new_value
                    
                    self.update_courses.append(update_record)
                    self.stats['courses_updated'] += 1
            else:
                # --- CREATE LOGIC ---
                course_id = str(uuid.uuid4())
                new_course = {
                    'id': course_id, 'code': course_code,
                    'name': row.get('course_name', 'Unknown Course'),
                    'description': row.get('course_description', 'No description available'),
                    'credit_units': float(row.get('credit_units', 1.0)) if pd.notna(row.get('credit_units')) else 1.0,
                    'belong_to_university': 1, 'belong_to_faculty': None,
                    'course_area': row.get('course_area'),
                    'enrolment_requirements': row.get('enrolment_requirements')
                }
                self.new_courses.append(new_course)
                self.courses_cache[course_code] = new_course  # Add to cache for this run
                self.stats['courses_created'] += 1
        
        logger.info(f"✅ Course processing complete. New: {self.stats['courses_created']}, Updated: {self.stats['courses_updated']}.")

    def assign_course_faculties_interactive(self):
        """Interactive faculty assignment with option to create new faculties"""
        if not self.courses_needing_faculty:
            logger.info("✅ No courses need faculty assignment")
            return
        
        logger.info(f"🎓 Starting interactive faculty assignment for {len(self.courses_needing_faculty)} courses")
        
        # Get current max faculty ID for incrementing
        max_faculty_id = max(self.faculties_cache.keys()) if self.faculties_cache else 0
        
        faculty_assignments = []
        
        for course_info in self.courses_needing_faculty:
            print(f"\n{'='*60}")
            print(f"🎓 FACULTY ASSIGNMENT NEEDED")
            print(f"{'='*60}")
            print(f"Course Code: {course_info['course_code']}")
            print(f"Course Name: {course_info['course_name']}")
            
            # Get the last filepath for this course from multiple sheet
            driver = None
            course_code = course_info['course_code']
            last_filepath = self.get_last_filepath_by_course(course_code)
            
            if last_filepath:
                print(f"\nOpening scraped HTML file: {last_filepath}")
                
                try:
                    # Setup Chrome options
                    chrome_options = Options()
                    chrome_options.add_argument("--new-window")
                    chrome_options.add_argument("--start-maximized")
                    
                    # Initialize driver
                    driver = webdriver.Chrome(options=chrome_options)
                    
                    # Open the HTML file
                    abs_path = os.path.abspath(last_filepath)
                    from pathlib import Path
                    file_path = Path(abs_path)
                    
                    if file_path.exists():
                        # Use pathlib's as_uri() method for proper file:// URL
                        file_url = file_path.as_uri()
                        driver.get(file_url)
                        print("✅ Scraped HTML file opened in browser")
                        print("📋 Review the course content to determine the correct faculty")
                    else:
                        print(f"⚠️ HTML file not found: {abs_path}")
                        print("📋 Proceeding without file preview")
                        
                except Exception as e:
                    print(f"⚠️ Could not open HTML file: {e}")
                    print("📋 Proceeding without file preview")
            else:
                print(f"⚠️ No scraped HTML file found for course {course_code}")
                print("📋 Proceeding without file preview")
            
            # Show existing faculties
            print("\nExisting Faculty Options:")
            faculty_list = sorted(self.faculties_cache.values(), key=lambda x: x['id'])
            for faculty in faculty_list:
                print(f"{faculty['id']}. {faculty['name']} ({faculty['acronym']})")
            
            print(f"\n0. Skip (will need manual review)")
            print(f"99. Create new faculty")
            
            while True:
                choice = input(f"\nEnter faculty number (0-{max(f['id'] for f in faculty_list)}, 99): ").strip()
                
                if choice == '0':
                    faculty_id = None
                    break
                elif choice == '99':
                    # Create new faculty
                    print("\n📝 Creating new faculty:")
                    faculty_name = input("Enter faculty name: ").strip()
                    faculty_acronym = input("Enter faculty acronym (e.g., SCIS): ").strip().upper()
                    faculty_url = input("Enter faculty website URL (or press Enter for default): ").strip()
                    
                    if not faculty_url:
                        faculty_url = f"https://smu.edu.sg/{faculty_acronym.lower()}"
                    
                    # Increment faculty ID
                    max_faculty_id += 1
                    new_faculty = {
                        'id': max_faculty_id,
                        'name': faculty_name,
                        'acronym': faculty_acronym,
                        'site_url': faculty_url,
                        'belong_to_university': 1,  # SMU
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Add to cache
                    self.faculties_cache[max_faculty_id] = new_faculty
                    self.faculty_acronym_to_id[faculty_acronym] = max_faculty_id
                    
                    # Save to new_faculties list
                    if not hasattr(self, 'new_faculties'):
                        self.new_faculties = []
                    self.new_faculties.append(new_faculty)
                    
                    faculty_id = max_faculty_id
                    print(f"✅ Created new faculty: {faculty_name} (ID: {faculty_id})")
                    break
                else:
                    try:
                        faculty_id = int(choice)
                        if faculty_id in [f['id'] for f in faculty_list]:
                            break
                        else:
                            print(f"Invalid choice. Please enter a valid faculty ID.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            
            # Close browser after selection
            if driver:
                try:
                    print("\n🔄 Closing browser...")
                    driver.quit()
                except Exception as e:
                    print(f"⚠️ Error closing browser: {e}")
            
            # Store assignment
            faculty_assignments.append({
                'course_id': course_info['course_id'],
                'course_code': course_info['course_code'],
                'faculty_id': faculty_id
            })
        
        # Apply assignments
        for assignment in faculty_assignments:
            if assignment['faculty_id'] is not None:
                # Update new_courses
                for course in self.new_courses:
                    if course['id'] == assignment['course_id']:
                        course['belong_to_faculty'] = assignment['faculty_id']
                        break
                
                # Update cache
                if assignment['course_code'] in self.courses_cache:
                    self.courses_cache[assignment['course_code']]['belong_to_faculty'] = assignment['faculty_id']
        
        # Save outputs
        if self.new_courses:
            df = pd.DataFrame(self.new_courses)
            df.to_csv(os.path.join(self.verify_dir, 'new_courses.csv'), index=False)
            logger.info(f"✅ Updated new_courses.csv with faculty assignments")
        
        if hasattr(self, 'new_faculties') and self.new_faculties:
            df = pd.DataFrame(self.new_faculties)
            df.to_csv(os.path.join(self.verify_dir, 'new_faculties.csv'), index=False)
            logger.info(f"✅ Saved {len(self.new_faculties)} new faculties")
        
        logger.info("✅ Faculty assignment completed")

    # Also add this as an alias to the existing method name
    def assign_course_faculties(self):
        """Alias for assign_course_faculties_interactive"""
        return self.assign_course_faculties_interactive()

    def process_acad_terms(self):
        """Process academic terms from standalone sheet"""
        logger.info("📅 Processing academic terms...")
        
        # Group by (acad_year_start, acad_year_end, term)
        term_groups = defaultdict(list)
        
        for _, row in self.standalone_data.iterrows():
            # Try to extract from row data first
            year_start = row.get('acad_year_start')
            year_end = row.get('acad_year_end')
            term = row.get('term')
            
            # If any are missing, try to extract from source file path if available
            if pd.isna(year_start) or pd.isna(year_end) or pd.isna(term):
                if 'source_file' in row and pd.notna(row['source_file']):
                    fallback_term_id = self.extract_acad_term_from_path(row['source_file'])
                    if fallback_term_id:
                        # Parse the fallback
                        match = re.match(r'AY(\d{4})(\d{2})T(\w+)', fallback_term_id)
                        if match:
                            year_start = int(match.group(1)) if pd.isna(year_start) else year_start
                            year_end = int(match.group(2)) if pd.isna(year_end) else year_end
                            term = f"T{match.group(3)}" if pd.isna(term) else term
            
            key = (year_start, year_end, term)
            if all(pd.notna(v) for v in key):
                term_groups[key].append(row)
        
        # Rest of the function remains the same...
        for (year_start, year_end, term), rows in term_groups.items():
            # Generate acad_term_id (keep T for ID)
            acad_term_id = f"AY{int(year_start)}{int(year_end) % 100:02d}{term}"
            
            # Check if already exists
            if acad_term_id in self.acad_term_cache:
                continue
            
            # Find most common period_text and dates
            period_counter = Counter()
            date_info = {}
            
            for row in rows:
                period_text = row.get('period_text', '')
                if pd.notna(period_text):
                    period_counter[period_text] += 1
                    if period_text not in date_info:
                        date_info[period_text] = {
                            'start_dt': row.get('start_dt'),
                            'end_dt': row.get('end_dt')
                        }
            
            # Get most common period
            if period_counter:
                most_common_period = period_counter.most_common(1)[0][0]
                dates = date_info[most_common_period]
            else:
                dates = {'start_dt': None, 'end_dt': None}
            
            # Get boss_id from first row
            boss_id = rows[0].get('acad_term_boss_id')
            
            # Remove T prefix from term field for database storage
            clean_term = str(term)[1:] if str(term).startswith('T') else str(term)
            
            new_term = {
                'id': acad_term_id,
                'acad_year_start': int(year_start),
                'acad_year_end': int(year_end),
                'term': clean_term,  # Store without T prefix
                'boss_id': int(boss_id) if pd.notna(boss_id) else None,
                'start_dt': dates['start_dt'],
                'end_dt': dates['end_dt']
            }
            
            self.new_acad_terms.append(new_term)
            self.acad_term_cache[acad_term_id] = new_term
            
            logger.info(f"✅ Created academic term: {acad_term_id} (term: {clean_term})")
        
        logger.info(f"✅ Created {len(self.new_acad_terms)} new academic terms")

    def process_classes(self, use_db_cache_for_classes=True):
        """
        Process classes from the standalone sheet. For single professor classes,
        uniqueness is determined by course_id + section + acad_term_id. For multi-professor classes, uniqueness includes professor_id.
        """
        logger.info("🏫 Processing classes with robust CREATE vs. UPDATE logic...")
        
        # Use sets to track processed classes and prevent duplicate operations
        processed_class_keys = set()
        processed_update_class_ids = set() # FIX: Prevents duplicate TBA updates

        # Initialize update_classes if it doesn't exist
        if not hasattr(self, 'update_classes'):
            self.update_classes = []

        # Build a lookup of existing classes from the cache for quick checks
        self.existing_class_lookup = {}
        if use_db_cache_for_classes:
            self.load_existing_classes_cache()
            if self.existing_classes_cache:
                for c in self.existing_classes_cache:
                    # For existing classes, use acad_term_id + boss_id + professor_id as the key
                    acad_term_id = c.get('acad_term_id')
                    class_boss_id = c.get('boss_id')
                    professor_id = c.get('professor_id')

                    # The primary key parts must exist to create a valid entry
                    if acad_term_id and class_boss_id is not None:
                        key = (acad_term_id, class_boss_id, professor_id)
                        self.existing_class_lookup[key] = c

        for idx, row in self.standalone_data.iterrows():
            try:
                acad_term_id = row.get('acad_term_id')
                class_boss_id = row.get('class_boss_id')
                course_code = row.get('course_code')
                section = str(row.get('section'))

                if pd.isna(acad_term_id) or pd.isna(class_boss_id):
                    continue
                
                course_id = self.get_course_id(course_code)
                if not course_id:
                    continue
                
                record_key = row.get('record_key')
                
                # Get all professors for this class
                professor_mappings = self._find_professors_for_class(record_key) if record_key else []
                
                # Handle the specific case where a TBA class gets a professor assigned.
                class_rows_in_multiple = [r for r in self.multiple_lookup.get(record_key, []) if r.get('type') == 'CLASS']
                
                # Condition: Exactly one professor is assigned now, and there's only one CLASS row for it.
                if len(professor_mappings) == 1 and len(class_rows_in_multiple) == 1:
                    new_prof_id = professor_mappings[0][0]
                    
                    # Search for a matching TBA record in the cache (where professor_id is null).
                    tba_class_to_update = None
                    if hasattr(self, 'existing_classes_cache') and self.existing_classes_cache:
                        for existing_class in self.existing_classes_cache:
                            if (existing_class.get('course_id') == course_id and
                                str(existing_class.get('section')) == section and
                                existing_class.get('acad_term_id') == acad_term_id and
                                pd.isna(existing_class.get('professor_id'))):
                                tba_class_to_update = existing_class
                                break
                    
                    # If we found a TBA record and haven't updated it yet, perform an UPDATE.
                    if tba_class_to_update and tba_class_to_update['id'] not in processed_update_class_ids:
                        logger.info(f"🔄 Converting TBA class {course_code}-{section} to assigned.")
                        
                        # Add to set to prevent this specific update from repeating.
                        processed_update_class_ids.add(tba_class_to_update['id'])

                        # Create the update record with the new professor.
                        update_record = {
                            'id': tba_class_to_update['id'],
                            'professor_id': new_prof_id
                        }
                        self.update_classes.append(update_record)
                        
                        # Map the record_key to the now-updated class ID for timing processing.
                        if record_key:
                            if record_key not in self.class_id_mapping:
                                self.class_id_mapping[record_key] = []
                            if tba_class_to_update['id'] not in self.class_id_mapping[record_key]:
                                self.class_id_mapping[record_key].append(tba_class_to_update['id'])
                                
                        # Update the in-memory cache to prevent a duplicate record from being created.
                        # This tells the rest of the function that this class now exists with an assigned professor.
                        new_assigned_key = (acad_term_id, class_boss_id, new_prof_id)
                        old_tba_key = (acad_term_id, class_boss_id, None)

                        updated_class_record = tba_class_to_update.copy()
                        updated_class_record['professor_id'] = new_prof_id
                        
                        self.existing_class_lookup[new_assigned_key] = updated_class_record
                        
                        # Clean up the old TBA entry from the in-memory cache
                        if old_tba_key in self.existing_class_lookup:
                            del self.existing_class_lookup[old_tba_key]

                        logger.info(f"🧠 In-memory cache updated for {course_code}-{section} to prevent duplicate creation.")

                        # Mark the new professor-class combination as processed to prevent the default logic
                        # from creating a duplicate class later in the loop.
                        class_key_for_processing = (acad_term_id, class_boss_id, new_prof_id)
                        processed_class_keys.add(class_key_for_processing)
                
                # If no professors found, create one class with professor_id = None
                if not professor_mappings:
                    professor_mappings = [(None, '')]
                
                # Check if this is a multi-professor class
                is_multi_professor = len(professor_mappings) > 1
                warn_inaccuracy = is_multi_professor
                
                # Process each professor
                for prof_id, prof_name in professor_mappings:
                    # Create unique key using boss_id
                    class_key = (acad_term_id, class_boss_id, prof_id)
                    
                    # Skip if we've already processed this exact class
                    if class_key in processed_class_keys:
                        continue
                        
                    processed_class_keys.add(class_key)
                    
                    # Check if this exact class exists (including professor_id match)
                    existing_class = self.existing_class_lookup.get(class_key)
                    
                    if existing_class:
                        # --- UPDATE LOGIC ---
                        update_record = {'id': existing_class['id']}
                        needs_update = False
                        
                        fields_to_check = {
                            'grading_basis': row.get('grading_basis'),
                            'course_outline_url': row.get('course_outline_url'),
                            'boss_id': int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None,
                            'warn_inaccuracy': warn_inaccuracy
                        }
                        
                        for field, new_value in fields_to_check.items():
                            old_value = existing_class.get(field)

                            # FIX: Safely handle array-like objects and pandas/numpy types
                            if hasattr(old_value, '__iter__') and not isinstance(old_value, str):
                                try:
                                    if hasattr(old_value, 'item'):  # numpy array
                                        old_value = old_value.item()
                                    elif hasattr(old_value, '__len__') and len(old_value) > 0:
                                        old_value = old_value[0]
                                    else:
                                        old_value = None
                                except:
                                    old_value = None
                            
                            # Convert new_value if it's also an array
                            if hasattr(new_value, '__iter__') and not isinstance(new_value, str):
                                try:
                                    if hasattr(new_value, 'item'):  # numpy array
                                        new_value = new_value.item()
                                    elif hasattr(new_value, '__len__') and len(new_value) > 0:
                                        new_value = new_value[0]
                                    else:
                                        new_value = None
                                except:
                                    new_value = None
                            
                            # Safe comparison
                            try:
                                # Handle None/NaN values first
                                if pd.isna(new_value) and pd.isna(old_value):
                                    continue
                                elif pd.isna(new_value) or pd.isna(old_value):
                                    if pd.notna(new_value):
                                        update_record[field] = new_value
                                        needs_update = True
                                    continue
                                
                                # Convert to strings for comparison
                                new_val_str = str(new_value).strip()
                                old_val_str = str(old_value).strip()
                                
                                if new_val_str != old_val_str:
                                    update_record[field] = new_value
                                    needs_update = True
                                    
                            except Exception as e:
                                # If any comparison fails, check if we have a new value to update
                                if pd.notna(new_value):
                                    update_record[field] = new_value
                                    needs_update = True
                        
                        if needs_update:
                            self.update_classes.append(update_record)
                        
                        # Map record_key to existing class ID for timings
                        if record_key:
                            if record_key not in self.class_id_mapping:
                                self.class_id_mapping[record_key] = []
                            if existing_class['id'] not in self.class_id_mapping[record_key]:
                                self.class_id_mapping[record_key].append(existing_class['id'])
                    else:
                        # --- CREATE LOGIC ---
                        # Check if we already created this class in new_classes
                        already_created = False
                        for new_class in self.new_classes:
                            if (new_class['acad_term_id'] == acad_term_id and
                                str(new_class.get('boss_id')) == str(class_boss_id) and
                                new_class.get('professor_id') == prof_id):
                                already_created = True
                                # Map record_key to this class ID
                                if record_key:
                                    if record_key not in self.class_id_mapping:
                                        self.class_id_mapping[record_key] = []
                                    if new_class['id'] not in self.class_id_mapping[record_key]:
                                        self.class_id_mapping[record_key].append(new_class['id'])
                                break
                        
                        if not already_created:
                            class_id = str(uuid.uuid4())
                            new_class = {
                                'id': class_id,
                                'section': section,
                                'course_id': course_id,
                                'professor_id': prof_id,
                                'acad_term_id': acad_term_id,
                                'created_at': datetime.now().isoformat(),
                                'updated_at': datetime.now().isoformat(),
                                'grading_basis': row.get('grading_basis'),
                                'course_outline_url': row.get('course_outline_url'),
                                'boss_id': int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None,
                                'raw_professor_name': prof_name,
                                'warn_inaccuracy': warn_inaccuracy
                            }
                            
                            self.new_classes.append(new_class)
                            self.stats['classes_created'] += 1
                            
                            # Also add to existing_class_lookup to prevent duplicates in same run
                            self.existing_class_lookup[class_key] = new_class
                            
                            # Map record_key to new class ID
                            if record_key:
                                if record_key not in self.class_id_mapping:
                                    self.class_id_mapping[record_key] = []
                                self.class_id_mapping[record_key].append(class_id)
            
            except Exception as e:
                logger.error(f"❌ Exception processing class row {idx}: {e}")
        
        logger.info(f"✅ Class processing complete. New: {self.stats['classes_created']}, Updates: {len(self.update_classes)}.")
        return True
        
    def _find_professors_for_class(self, record_key: str) -> List[tuple]:
        """Find professor IDs for a class and return list of (professor_id, original_name) tuples
        Deduplicates by professor_id to avoid creating multiple class records for same professor"""
        if not record_key or pd.isna(record_key):
            return []
        
        rows = self.multiple_lookup.get(record_key, [])
        professor_mappings = []
        seen_professor_ids = set()  # Track unique professor IDs
        
        # Ensure professor lookup is loaded
        if not hasattr(self, 'professor_lookup_loaded'):
            self.load_professor_lookup_csv()
        
        for row in rows:
            prof_name_raw = row.get('professor_name')
            
            # FIXED: Better handling of NaN values from raw_data.xlsx
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue
            
            # Convert to string and strip - handles float NaN properly
            original_prof_name = str(prof_name_raw).strip()
            
            # Skip empty strings and 'nan' strings
            if not original_prof_name or original_prof_name.lower() == 'nan':
                continue
            
            # Split the professor names intelligently
            split_professors = self._split_professor_names(original_prof_name)
            
            # Process each split professor
            for prof_name in split_professors:
                if prof_name and prof_name.strip():  # Additional check for empty strings
                    prof_id = self._lookup_professor_with_fallback(prof_name.strip())
                    if prof_id and prof_id not in seen_professor_ids:
                        professor_mappings.append((prof_id, prof_name.strip()))
                        seen_professor_ids.add(prof_id)
        
        return professor_mappings

    def _split_professor_names(self, prof_name: str) -> List[str]:
        """
        Intelligently splits a string of professor names using a greedy, longest-match-first
        approach, which eliminates the need for hardcoded combinations.
        """
        if prof_name is None or pd.isna(prof_name) or not str(prof_name).strip():
            return []

        prof_name_str = str(prof_name).strip()
        
        # First, check if the entire string is already a known professor.
        # This handles names that include commas, like "LEE, MICHELLE PUI YEE".
        if prof_name_str.upper() in self.professor_lookup:
            return [prof_name_str]
            
        # If there are no commas, it can only be one professor.
        if ',' not in prof_name_str:
            return [prof_name_str]

        parts = [p.strip() for p in prof_name_str.split(',') if p.strip()]
        
        found_professors = []
        i = 0
        while i < len(parts):
            # Start from the longest possible combination of remaining parts and work backwards.
            match_found = False
            for j in range(len(parts), i, -1):
                # Create a candidate name by joining the parts
                candidate = ', '.join(parts[i:j])
                
                # Check if this longest possible candidate is a known professor
                if candidate.upper() in self.professor_lookup:
                    found_professors.append(candidate)
                    i = j  # Move the pointer past the parts we just consumed
                    match_found = True
                    break # Exit the inner loop and continue with the next part
            
            # If the inner loop finished without finding any match for the part(s)
            if not match_found:
                # This part is an unknown entity.
                # Per your logic, we can try to append it to the previously found professor.
                # This handles cases like "WONG LI DE, BRIAN" where "BRIAN" is unknown.
                unknown_part = parts[i]
                if found_professors and len(unknown_part.split()) == 1:
                    # Append the unknown single-word part to the last known professor
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                    logger.info(f"✅ Combined unknown single word '{unknown_part}' with previous professor -> '{found_professors[-1]}'")
                else:
                    # Otherwise, treat it as its own (potentially new) professor
                    found_professors.append(unknown_part)
                
                i += 1 # Move to the next part
                
        return found_professors

    def _lookup_professor_with_fallback(self, prof_name: str) -> Optional[str]:
        """Enhanced professor lookup with improved partial word matching and no phantom professor creation."""
        
        if prof_name is None or pd.isna(prof_name):
            return None
        
        prof_name = str(prof_name).strip()
        if not prof_name or prof_name.lower() == 'nan':
            return None
        
        # Strategy 1 & 2: Direct and variation-based lookup (unchanged).
        normalized_name = prof_name.upper()
        if hasattr(self, 'professor_lookup'):
            if normalized_name in self.professor_lookup:
                return self.professor_lookup[normalized_name]['database_id']
            
            variations = [
                prof_name.strip().upper(),
                prof_name.replace(',', '').strip().upper(),
                ' '.join(prof_name.replace(',', ' ').split()).upper()
            ]
            for variation in variations:
                if variation in self.professor_lookup:
                    return self.professor_lookup[variation]['database_id']
        
        # Strategy 3: Search boss_aliases in professors_cache using the new robust parser.
        search_name_normalized = normalized_name
        for prof_data in self.professors_cache.values():
            aliases_list = self._parse_boss_aliases(prof_data.get('boss_aliases'))
            
            for alias in aliases_list:
                alias_normalized = alias.strip().upper()
                
                if alias_normalized == search_name_normalized:
                    logger.info(f"✅ Found exact match in boss_aliases: {prof_name} → {alias} (ID: {prof_data.get('id')})")
                    if not hasattr(self, 'professor_lookup'): self.professor_lookup = {}
                    self.professor_lookup[search_name_normalized] = {
                        'database_id': str(prof_data.get('id')),
                        'boss_name': alias_normalized,
                        'afterclass_name': prof_data.get('name', prof_name)
                    }
                    return str(prof_data.get('id'))

        # Strategy 4: Enhanced partial word matching for cases like "DENNIS LIM" → "LIM CHONG BOON DENNIS"
        search_words = set(normalized_name.replace(',', ' ').split())
        if len(search_words) >= 2:  # Only try partial matching for multi-word names
            best_match = None
            best_score = 0
            
            for prof_data in self.professors_cache.values():
                # Check against afterclass_name (database name)
                db_name = prof_data.get('name', '').upper()
                db_words = set(db_name.replace(',', ' ').split())
                
                # Check if all search words are found in database name
                if search_words.issubset(db_words):
                    # Calculate match score (percentage of db_words that match search_words)
                    match_score = len(search_words) / len(db_words) if db_words else 0
                    
                    if match_score > best_score and match_score >= 0.5:  # At least 50% match
                        best_match = prof_data
                        best_score = match_score
                
                # Also check against boss_aliases
                aliases_list = self._parse_boss_aliases(prof_data.get('boss_aliases'))
                for alias in aliases_list:
                    alias_words = set(alias.upper().replace(',', ' ').split())
                    if search_words.issubset(alias_words):
                        match_score = len(search_words) / len(alias_words) if alias_words else 0
                        if match_score > best_score and match_score >= 0.5:
                            best_match = prof_data
                            best_score = match_score
            
            if best_match and best_score >= 0.5:
                logger.info(f"🔍 Partial word match found: '{prof_name}' → '{best_match.get('name')}' (score: {best_score:.2f})")
                
                # Add to lookup and save to partial matches tracking
                if not hasattr(self, 'professor_lookup'): self.professor_lookup = {}
                self.professor_lookup[normalized_name] = {
                    'database_id': str(best_match.get('id')),
                    'boss_name': normalized_name,
                    'afterclass_name': best_match.get('name', prof_name)
                }
                
                # Track partial matches for review
                if not hasattr(self, 'partial_matches'):
                    self.partial_matches = []
                self.partial_matches.append({
                    'boss_name': prof_name,
                    'afterclass_name': best_match.get('name'),
                    'database_id': str(best_match.get('id')),
                    'method': 'partial_match',
                    'match_score': f"{best_score:.2f}"
                })
                
                return str(best_match.get('id'))
        
        # Strategy 5: Exact fuzzy matching (unchanged)
        if hasattr(self, 'professor_lookup'):
            for lookup_name in self.professor_lookup.keys():
                if self._names_match_fuzzy_exact(normalized_name, lookup_name):
                    return self.professor_lookup[lookup_name]['database_id']
        
        if normalized_name in self.professors_cache:
            return self.professors_cache[normalized_name]['id']
        
        # Strategy 6: DO NOT create new professor - log as unmatched instead
        logger.warning(f"⚠️ Professor not found, will create new: {prof_name}")
        
        # Create new professor (only when absolutely necessary)
        return self._create_new_professor(prof_name)
        
    def _names_match_fuzzy_exact(self, name1: str, name2: str) -> bool:
        """Exact fuzzy matching for names - only matches if completely identical after normalization"""
        
        # Handle None and non-string values
        if name1 is None or name2 is None:
            return False
        
        # Ensure both names are strings
        name1 = str(name1) if name1 is not None else ""
        name2 = str(name2) if name2 is not None else ""
        
        # Remove common variations and normalize
        clean1 = ' '.join(name1.replace(',', ' ').replace('.', ' ').split()).upper()
        clean2 = ' '.join(name2.replace(',', ' ').replace('.', ' ').split()).upper()
        
        # Only return True if they are exactly the same after cleaning
        return clean1 == clean2

    def load_professor_lookup_csv(self):
        """Load professor lookup CSV once and cache it properly"""
        # Check if already loaded to prevent repeated loading
        if hasattr(self, 'professor_lookup_loaded') and self.professor_lookup_loaded:
            return
        
        lookup_file = 'script_input/professor_lookup.csv'
        
        if not os.path.exists(lookup_file):
            logger.warning("📋 professor_lookup.csv not found - will use database cache only")
            self.professor_lookup_loaded = True
            return
        
        try:
            # Load the CSV file
            lookup_df = pd.read_csv(lookup_file)
            
            # Validate required columns exist
            required_cols = ['boss_name', 'afterclass_name', 'database_id', 'method']
            missing_cols = [col for col in required_cols if col not in lookup_df.columns]
            if missing_cols:
                logger.error(f"❌ professor_lookup.csv missing required columns: {missing_cols}")
                self.professor_lookup_loaded = True
                return
            
            # Clear existing lookup and load fresh data
            self.professor_lookup = {}
            loaded_count = 0
            
            for _, row in lookup_df.iterrows():
                boss_name = row.get('boss_name')
                afterclass_name = row.get('afterclass_name')
                database_id = row.get('database_id')
                
                # Skip rows with critical missing values
                if pd.isna(boss_name) or pd.isna(database_id):
                    continue
                    
                # Use boss_name as the primary key for lookup (as you specified)
                boss_name_key = str(boss_name).strip().upper()
                self.professor_lookup[boss_name_key] = {
                    'database_id': str(database_id),
                    'boss_name': str(boss_name),
                    'afterclass_name': str(afterclass_name) if not pd.isna(afterclass_name) else str(boss_name)
                }
                loaded_count += 1
            
            logger.info(f"✅ Loaded {loaded_count} entries from professor_lookup.csv")
            self.professor_lookup_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Error loading professor_lookup.csv: {e}")
            logger.info("📋 Continuing with database cache only")
            self.professor_lookup_loaded = True

    def _create_new_professor(self, prof_name: str, professor_variations: dict = None, email_to_professor: dict = None) -> str:
        """
        Create a new professor record, ensure proper tracking, and handle both primary and fallback alias creation.
        """
        boss_name, afterclass_name = self._normalize_professor_name_fallback(prof_name)
        
        # Check if already created in this session to prevent duplicates
        for new_prof in self.new_professors:
            aliases_val = new_prof.get('boss_aliases', '[]')
            try:
                import json
                alias_list = json.loads(aliases_val) if isinstance(aliases_val, str) else aliases_val
            except (json.JSONDecodeError, TypeError):
                alias_list = []

            if boss_name in alias_list or afterclass_name == new_prof.get('name', ''):
                # This professor was already created in this run, just return its ID.
                return new_prof.get('id')
        
        # --- Unified Creation Logic ---
        professor_id = str(uuid.uuid4())
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', afterclass_name.lower()).strip('-')
        resolved_email = self.resolve_professor_email(afterclass_name)

        # --- Conditional Alias Creation ---
        boss_aliases_set = set()
        boss_aliases_set.add(boss_name)
        
        # SCENARIO A: Use sophisticated alias creation if professor_variations dictionary is provided
        if professor_variations:
            professor_specific_variations = professor_variations.get(prof_name, set())
            for variation in professor_specific_variations:
                if variation and variation.strip():
                    variation_boss_name, _ = self._normalize_professor_name_fallback(variation.strip())
                    boss_aliases_set.add(variation_boss_name)
        # SCENARIO B: Fallback to simple alias creation if not provided
        else:
            if boss_name != prof_name.upper():
                boss_aliases_set.add(prof_name.upper())

        boss_aliases_list = sorted(list(boss_aliases_set))
        import json
        boss_aliases_json = json.dumps(boss_aliases_list)
        
        # --- Create and Register the New Professor ---
        new_prof = {
            'id': professor_id,
            'name': afterclass_name,
            'email': resolved_email,
            'slug': slug,
            'photo_url': 'https://smu.edu.sg',
            'profile_url': 'https://smu.edu.sg',
            'belong_to_university': 1,
            'boss_aliases': boss_aliases_json,
            'original_scraped_name': prof_name
        }
        
        self.new_professors.append(new_prof)
        self.stats['professors_created'] += 1
        
        # Update lookup tables
        if not hasattr(self, 'professor_lookup'):
            self.professor_lookup = {}
        
        lookup_entry = {
            'database_id': professor_id,
            'boss_name': boss_name,
            'afterclass_name': afterclass_name
        }
        # Map the original name and all its aliases to the new ID
        self.professor_lookup[prof_name.upper()] = lookup_entry
        for alias in boss_aliases_list:
            self.professor_lookup[alias.upper()] = lookup_entry

        # Update the email duplicate checker dictionary if it was passed in
        if email_to_professor is not None and resolved_email and resolved_email.lower() != 'enquiry@smu.edu.sg':
            email_to_professor[resolved_email.lower()] = new_prof
        
        logger.info(f"✅ Created professor: {afterclass_name} with email: {resolved_email}")
        logger.info(f"   Boss aliases: {boss_aliases_list}")
        
        return professor_id

    def process_timings(self):
        """
        MODIFICATION 2 (REVISED): Process class and exam timings with robust deduplication that handles TBA cases.
        """
        logger.info("⏰ Processing class timings and exam timings with strict uniqueness checks...")

        # A tiny, local helper function to consistently handle missing values for key creation.
        # This prevents the 'nan' vs 'None' string issue.
        def _clean_key_val(v):
            return '' if pd.isna(v) else str(v)

        # Load existing timing keys from the database cache ONCE.
        if not self.processed_timing_keys:
            cache_file = os.path.join(self.cache_dir, 'class_timing_cache.pkl')
            if os.path.exists(cache_file):
                try:
                    df = pd.read_pickle(cache_file)
                    if not df.empty and 'class_id' in df.columns:
                        for _, record in df.iterrows():
                            # FIX: Use the robust key creation logic for cached data
                            key = (
                                record['class_id'],
                                _clean_key_val(record.get('day_of_week')),
                                _clean_key_val(record.get('start_time')),
                                _clean_key_val(record.get('end_time')),
                                _clean_key_val(record.get('venue'))
                            )
                            self.processed_timing_keys.add(key)
                        logger.info(f"✅ Pre-loaded {len(self.processed_timing_keys)} existing class timing keys from cache.")
                except Exception as e:
                    logger.warning(f"Could not preload class_timing_cache.pkl: {e}")

        # Load existing exam timing keys
        if not self.processed_exam_class_ids:
            exam_cache_file = os.path.join(self.cache_dir, 'class_exam_timing_cache.pkl')
            if os.path.exists(exam_cache_file):
                try:
                    df = pd.read_pickle(exam_cache_file)
                    if not df.empty and 'class_id' in df.columns:
                        self.processed_exam_class_ids.update(df['class_id'].unique())
                        logger.info(f"✅ Pre-loaded {len(self.processed_exam_class_ids)} existing exam class IDs from cache.")
                except Exception as e:
                    logger.warning(f"Could not preload class_exam_timing_cache.pkl: {e}")
        
        for _, row in self.multiple_data.iterrows():
            record_key = row.get('record_key')
            if record_key not in self.class_id_mapping:
                continue
            
            class_ids = self.class_id_mapping.get(record_key, [])
            timing_type = row.get('type', 'CLASS')
            
            for class_id in class_ids:
                if timing_type == 'CLASS':
                    # --- FIX: Use the same robust key generation for new data ---
                    timing_key = (
                        class_id,
                        _clean_key_val(row.get('day_of_week')),
                        _clean_key_val(row.get('start_time')),
                        _clean_key_val(row.get('end_time')),
                        _clean_key_val(row.get('venue'))
                    )
                    
                    # Check if this exact timing record has already been processed
                    if timing_key in self.processed_timing_keys:
                        continue
                    
                    # Add to set *before* appending to prevent duplicates within the same run
                    self.processed_timing_keys.add(timing_key)
                    
                    timing_record = {
                        'class_id': class_id, 'start_date': row.get('start_date'),
                        'end_date': row.get('end_date'), 'day_of_week': row.get('day_of_week'),
                        'start_time': row.get('start_time'), 'end_time': row.get('end_time'),
                        'venue': row.get('venue', '')
                    }
                    self.new_class_timings.append(timing_record)
                    self.stats['timings_created'] += 1
                
                elif timing_type == 'EXAM':
                    if class_id in self.processed_exam_class_ids:
                        continue
                    
                    self.processed_exam_class_ids.add(class_id)
                    
                    exam_record = {
                        'class_id': class_id, 'date': row.get('date'),
                        'day_of_week': row.get('day_of_week'), 
                        'start_time': str(row.get('start_time')),
                        'end_time': str(row.get('end_time')), 
                        'venue': row.get('venue')
                    }
                    self.new_class_exam_timings.append(exam_record)
                    self.stats['exams_created'] += 1
        
        logger.info(f"✅ Created {self.stats['timings_created']} new class timings (after deduplication).")
        logger.info(f"✅ Created {self.stats['exams_created']} new exam timings (after deduplication).")

    def save_outputs(self):
        """Save all generated CSV files, only creating files that have data."""
        logger.info("💾 Saving output files...")

        def to_csv_if_not_empty(data_list, filename):
            if data_list:
                df = pd.DataFrame(data_list)
                if not df.empty:
                    path = os.path.join(self.output_base, filename)
                    df.to_csv(path, index=False)
                    logger.info(f"✅ Saved {len(df)} records to {filename}")
                    
                    # DEBUG: For update_bid_result, show what we're saving
                    if filename == 'update_bid_result.csv':
                        logger.info(f"🔍 DEBUG: update_bid_result.csv columns: {list(df.columns)}")
                        # Show first few rows with median/min values
                        rows_with_bid_data = df[(df['median'].notna()) | (df['min'].notna())]
                        if not rows_with_bid_data.empty:
                            logger.info(f"🔍 DEBUG: {len(rows_with_bid_data)} rows have median/min data")
                            logger.info(f"🔍 DEBUG: Sample data:")
                            for idx, row in rows_with_bid_data.head(3).iterrows():
                                logger.info(f"  bid_window_id={row['bid_window_id']}, class_id={row['class_id']}, median={row.get('median')}, min={row.get('min')}")
                        else:
                            logger.warning("⚠️ DEBUG: No rows in update_bid_result have median/min values!")

        # Note: new_professors is handled in Phase 1 (verify folder)
        to_csv_if_not_empty(getattr(self, 'update_professors', []), 'update_professor.csv')
        to_csv_if_not_empty(self.update_courses, 'update_courses.csv')
        to_csv_if_not_empty(getattr(self, 'update_classes', []), 'update_classes.csv')
        to_csv_if_not_empty(self.new_acad_terms, 'new_acad_term.csv')
        to_csv_if_not_empty(self.new_classes, 'new_classes.csv')
        to_csv_if_not_empty(self.new_class_timings, 'new_class_timing.csv')
        to_csv_if_not_empty(self.new_class_exam_timings, 'new_class_exam_timing.csv')
        
        update_records = getattr(self, 'update_bid_result', [])
        if update_records:
            logger.info(f"📝 Preparing to save {len(update_records)} bid result updates")
            # Show sample of records with actual bid data
            records_with_bids = [r for r in update_records if r.get('median') is not None or r.get('min') is not None]
            logger.info(f"   - {len(records_with_bids)} records have median/min bid data")
            if records_with_bids:
                sample = records_with_bids[0]
                logger.info(f"   - Sample: bid_window_id={sample.get('bid_window_id')}, median={sample.get('median')}, min={sample.get('min')}")
        to_csv_if_not_empty(update_records, 'update_bid_result.csv')

        if self.courses_needing_faculty:
            df = pd.DataFrame(self.courses_needing_faculty)
            df.to_csv(os.path.join(self.verify_dir, 'courses_needing_faculty.csv'), index=False)
            logger.info(f"✅ Saved {len(self.courses_needing_faculty)} courses needing faculty assignment to the verify folder.")
            
    def update_professor_lookup_from_corrected_csv(self):
        """Update professor lookup from manually corrected new_professors.csv"""
        logger.info("🔄 Updating professor lookup from corrected CSV...")
        
        # Read corrected new_professors.csv
        corrected_csv_path = os.path.join(self.verify_dir, 'new_professors.csv')
        if not os.path.exists(corrected_csv_path):
            logger.info(f"📝 No corrected CSV found: {corrected_csv_path} - assuming all professors already exist")
            return True

        corrected_df = pd.read_csv(corrected_csv_path)
        if corrected_df.empty:
            logger.info(f"📝 Empty corrected CSV - no professors to update")
            return True

        try:
            logger.info(f"📖 Reading {len(corrected_df)} corrected professor records")
            
            # Clear and rebuild the new_professors list with corrected data
            self.new_professors = []
            
            # Update internal professor_lookup and rebuild new_professors
            updated_count = 0
            
            # FIXED: Initialize professor_lookup if it doesn't exist
            if not hasattr(self, 'professor_lookup'):
                self.professor_lookup = {}
            
            for _, row in corrected_df.iterrows():
                original_name = row.get('original_scraped_name', '')
                corrected_afterclass_name = row.get('name', '')  # This is the corrected name
                boss_aliases = row.get('boss_aliases', '')  # This should be JSON string
                professor_id = row.get('id', '')
                
                # Parse boss_aliases JSON string
                try:
                    import json
                    if isinstance(boss_aliases, str) and boss_aliases.strip():
                        boss_aliases_list = json.loads(boss_aliases)
                        if isinstance(boss_aliases_list, list) and boss_aliases_list:
                            boss_name = boss_aliases_list[0]  # Use first boss alias
                        else:
                            boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                    else:
                        boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                except (json.JSONDecodeError, TypeError):
                    # Fallback if JSON parsing fails
                    boss_name = original_name.upper() if original_name else corrected_afterclass_name.upper()
                
                # Rebuild the professor record with corrected data
                corrected_prof = {
                    'id': professor_id,
                    'name': corrected_afterclass_name,  # Use corrected name
                    'email': row.get('email', 'enquiry@smu.edu.sg'),
                    'slug': row.get('slug', ''),
                    'photo_url': row.get('photo_url', 'https://smu.edu.sg'),
                    'profile_url': row.get('profile_url', 'https://smu.edu.sg'),
                    'belong_to_university': row.get('belong_to_university', 1),
                    'boss_aliases': boss_aliases,  # Keep as JSON string
                    'afterclass_name': corrected_afterclass_name,
                    'original_scraped_name': original_name
                }
                
                # Add to new_professors list
                self.new_professors.append(corrected_prof)
                
                # FIXED: Update professor_lookup with ALL variations
                if professor_id:
                    lookup_entry = {
                        'database_id': professor_id,
                        'boss_name': boss_name,
                        'afterclass_name': corrected_afterclass_name
                    }
                    
                    # Add original scraped name to lookup
                    if original_name:
                        self.professor_lookup[original_name.upper()] = lookup_entry
                        updated_count += 1
                    
                    # Add corrected afterclass name to lookup
                    if corrected_afterclass_name:
                        self.professor_lookup[corrected_afterclass_name.upper()] = lookup_entry
                    
                    # Add boss_name to lookup
                    if boss_name:
                        self.professor_lookup[boss_name.upper()] = lookup_entry
                    
                    # FIXED: Add all boss aliases to lookup
                    try:
                        if isinstance(boss_aliases, str) and boss_aliases.strip():
                            boss_aliases_list = json.loads(boss_aliases)
                            if isinstance(boss_aliases_list, list):
                                for alias in boss_aliases_list:
                                    if alias and str(alias).strip():
                                        self.professor_lookup[str(alias).upper()] = lookup_entry
                    except (json.JSONDecodeError, TypeError):
                        pass  # Skip if JSON parsing fails
            
            # Save updated professor lookup to CSV
            self._save_corrected_professor_lookup()
            
            logger.info(f"✅ Updated {updated_count} professor lookup entries")
            logger.info(f"✅ Rebuilt {len(self.new_professors)} professor records with corrections")
            logger.info(f"✅ Total lookup entries now: {len(self.professor_lookup)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update professor lookup: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_professors_with_boss_names(self):
        """
        Update professors with missing/additional boss_names by comparing professor_lookup.csv
        with database boss_aliases and combining new variations from high-confidence fuzzy matches.
        """
        logger.info("👤 Updating professors with boss_names and detecting new variations...")

        # --- Step 1: Load high-confidence fuzzy matches from Phase 1 ---
        fuzzy_path = os.path.join(self.verify_dir, 'fuzzy_matched_professors.csv')
        new_aliases_by_id = defaultdict(list)

        if os.path.exists(fuzzy_path):
            try:
                fuzzy_df = pd.read_csv(fuzzy_path)
                high_confidence_matches = fuzzy_df[fuzzy_df['confidence_score'] >= 95]
                logger.info(f"🔍 Found {len(high_confidence_matches)} high-confidence (>=95) fuzzy matches to process.")

                for _, row in high_confidence_matches.iterrows():
                    database_id = str(row['database_id'])
                    afterclass_name = row['afterclass_name']
                    
                    try:
                        import json
                        aliases_val = row.get('boss_aliases', '[]')
                        new_aliases = json.loads(aliases_val) if isinstance(aliases_val, str) else []
                        
                        for alias in new_aliases:
                            if alias and str(alias).strip():
                                clean_alias = str(alias).strip()
                                new_aliases_by_id[database_id].append(clean_alias)
                                
                                # Add to in-memory professor_lookup to be saved later
                                if not hasattr(self, 'professor_lookup'):
                                    self.professor_lookup = {}
                                
                                alias_key = clean_alias.upper()
                                if alias_key not in self.professor_lookup:
                                    self.professor_lookup[alias_key] = {
                                        'database_id': database_id,
                                        'boss_name': clean_alias,
                                        'afterclass_name': afterclass_name,
                                        'method': 'fuzzy_match' # Add method for tracking
                                    }
                                    logger.info(f"➕ Adding fuzzy match to lookup: '{clean_alias}' -> '{afterclass_name}'")

                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"⚠️ Could not parse boss_aliases from fuzzy_matched_professors.csv for row: {row.to_dict()}. Error: {e}")
            
            except Exception as e:
                logger.error(f"❌ Error processing fuzzy_matched_professors.csv: {e}")

        # --- Step 2: Load existing variations from professor_lookup.csv ---
        lookup_file = 'script_input/professor_lookup.csv'
        lookup_groups = defaultdict(list)
        if os.path.exists(lookup_file):
            try:
                lookup_df = pd.read_csv(lookup_file)
                for _, row in lookup_df.iterrows():
                    database_id = row.get('database_id')
                    boss_name = row.get('boss_name')
                    if pd.notna(database_id) and pd.notna(boss_name):
                        lookup_groups[str(database_id)].append(str(boss_name).strip())
            except Exception as e:
                logger.error(f"❌ Error loading professor_lookup.csv: {e}")

        # --- Step 3: Iterate through professors and combine all alias sources ---
        updated_professor_ids = set() 
        self.update_professors = []
        new_variations_found = []
        import json

        for prof_key, prof_data in self.professors_cache.items():
            professor_id = str(prof_data.get('id'))
            if professor_id in updated_professor_ids:
                continue

            # Get all sources of aliases as sets for easy combination
            current_boss_aliases = set(self._parse_boss_aliases(prof_data.get('boss_aliases')))
            lookup_variations = set(lookup_groups.get(professor_id, []))
            fuzzy_variations = set(new_aliases_by_id.get(professor_id, []))

            # Combine all unique variations using set union
            final_aliases_raw = current_boss_aliases.union(lookup_variations).union(fuzzy_variations)

            # Normalize both sets for a stable comparison, preventing repeated updates
            current_aliases_normalized = {name.replace("’", "'") for name in current_boss_aliases}
            final_aliases_normalized = {name.replace("’", "'") for name in final_aliases_raw}

            # Check for changes using the normalized sets
            if final_aliases_normalized != current_aliases_normalized:
                # Save the raw, original names to preserve the smart quote from the source
                unique_boss_names = sorted(list(final_aliases_raw))
                # Use ensure_ascii=False to prevent encoding '’' to '\u2019'
                boss_aliases_json = json.dumps(unique_boss_names, ensure_ascii=False)

                self.update_professors.append({
                    'id': professor_id,
                    'boss_aliases': boss_aliases_json,
                })
                
                # For logging, find the newly added variations
                newly_added = final_aliases_raw - current_boss_aliases
                if newly_added:
                    logger.info(f"✅ Adding {len(newly_added)} new variations for professor {professor_id}: {sorted(list(newly_added))}")
                    new_variations_found.append({
                        'professor_id': professor_id,
                        'professor_name': prof_data.get('name', 'Unknown'),
                        'existing_aliases': sorted(list(current_boss_aliases)),
                        'new_variations': sorted(list(newly_added)),
                        'final_aliases': unique_boss_names
                    })
                
                updated_professor_ids.add(professor_id)

        # --- Step 4: Save all outputs ---
        # Save partial matches if any were found
        if hasattr(self, 'partial_matches') and self.partial_matches:
            partial_df = pd.DataFrame(self.partial_matches)
            partial_path = os.path.join(self.verify_dir, 'partial_matches.csv')
            partial_df.to_csv(partial_path, index=False)
            logger.info(f"🔍 Saved {len(self.partial_matches)} partial matches to partial_matches.csv")

        # Save new variations summary
        if new_variations_found:
            report_data = [{'professor_id': item.get('professor_id'),'professor_name': item.get('professor_name'), 'existing_aliases': '|'.join(item.get('existing_aliases', [])), 'new_variations': '|'.join(item.get('new_variations', [])),'final_aliases': '|'.join(item.get('final_aliases', []))} for item in new_variations_found]
            variations_df = pd.DataFrame(report_data)
            variations_path = os.path.join(self.verify_dir, 'new_variations_found.csv')
            variations_df.to_csv(variations_path, index=False, encoding='utf-8-sig')
            logger.info(f"🆕 Saved {len(new_variations_found)} professors with new variations to new_variations_found.csv")

        # Save the update_professor.csv file
        if self.update_professors:
            df = pd.DataFrame(self.update_professors)
            update_path = os.path.join(self.output_base, 'update_professor.csv')
            df.to_csv(update_path, index=False, encoding='utf-8')
            logger.info(f"✅ Saved {len(self.update_professors)} unique professor updates to update_professor.csv")
            self.stats['professors_updated'] = len(self.update_professors)
        else:
            logger.info("ℹ️ No professors need boss_name updates.")
            self.stats['professors_updated'] = 0

        # --- Step 5: Persist the updated professor lookup table ---
        self._save_corrected_professor_lookup()

    def process_remaining_tables(self):
        """Process classes and timings after professor lookup is updated"""
        logger.info("🏫 Processing remaining tables (classes, timings)...")
        
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
            self.process_classes()
            
            # Process timings (depends on classes)
            self.process_timings()
            
            logger.info("✅ Remaining tables processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process remaining tables: {e}")
            return False

    def _save_corrected_professor_lookup(self):
        """Save professor lookup preserving all input entries, adding new ones, and including partial matches"""
        # Start with all existing entries from input professor_lookup.csv
        all_lookup_data = {}
        
        # Step 1: Load ALL entries from input professor_lookup.csv (preserve existing)
        input_lookup_file = 'script_input/professor_lookup.csv'
        if os.path.exists(input_lookup_file):
            try:
                input_df = pd.read_csv(input_lookup_file)
                for _, row in input_df.iterrows():
                    boss_name = row.get('boss_name')
                    afterclass_name = row.get('afterclass_name')
                    database_id = row.get('database_id')
                    method = row.get('method', 'exists')
                    
                    # Only require database_id to be present
                    if pd.notna(database_id):
                        if pd.isna(boss_name) or str(boss_name).strip() == '':
                            if pd.notna(afterclass_name):
                                lookup_key = f"EMPTY_BOSS_{str(afterclass_name).upper().replace(' ', '_')}"
                                boss_name_value = ""
                            else:
                                lookup_key = f"EMPTY_BOSS_{str(database_id)}"
                                boss_name_value = ""
                        else:
                            lookup_key = str(boss_name).upper()
                            boss_name_value = str(boss_name)
                        
                        all_lookup_data[lookup_key] = {
                            'boss_name': boss_name_value,
                            'afterclass_name': str(afterclass_name) if pd.notna(afterclass_name) else "",
                            'database_id': str(database_id),
                            'method': str(method)
                        }
                
                logger.info(f"📖 Loaded {len(all_lookup_data)} existing entries from input professor_lookup.csv")
            except Exception as e:
                logger.warning(f"⚠️ Could not load input professor_lookup.csv: {e}")
        
        # Step 2: Add/update with new entries from current processing
        new_entries_count = 0
        updated_entries_count = 0
        
        for scraped_name, data in self.professor_lookup.items():
            boss_name = data.get('boss_name', scraped_name.upper())
            afterclass_name = data.get('afterclass_name', scraped_name)
            database_id = data['database_id']
            
            # Determine method: check if this is a newly created professor or partial match
            method = 'exists'  # default
            if any(prof['id'] == database_id for prof in self.new_professors):
                method = 'created'
            elif hasattr(self, 'partial_matches') and any(match['database_id'] == database_id and match['boss_name'] == scraped_name for match in self.partial_matches):
                method = 'partial_match'
            
            boss_name_key = str(boss_name).upper()
            
            if boss_name_key in all_lookup_data:
                # Update existing entry if method changed
                if method in ['created', 'partial_match']:
                    all_lookup_data[boss_name_key]['method'] = method
                    updated_entries_count += 1
            else:
                # Add new entry
                all_lookup_data[boss_name_key] = {
                    'boss_name': str(boss_name),
                    'afterclass_name': str(afterclass_name),
                    'database_id': str(database_id),
                    'method': method
                }
                new_entries_count += 1
                logger.info(f"   -> NEW LOOKUP: Adding '{boss_name}' for '{afterclass_name}' (ID: {database_id}, method: {method})")
        
        # Step 3: Add partial matches that weren't already in professor_lookup
        if hasattr(self, 'partial_matches'):
            for match in self.partial_matches:
                boss_name_key = match['boss_name'].upper()
                if boss_name_key not in all_lookup_data:
                    all_lookup_data[boss_name_key] = {
                        'boss_name': match['boss_name'],
                        'afterclass_name': match['afterclass_name'],
                        'database_id': match['database_id'],
                        'method': f"partial_match_{match.get('match_score', '')}"
                    }
                    new_entries_count += 1
                    logger.info(f"   -> PARTIAL MATCH: Adding '{match['boss_name']}' → '{match['afterclass_name']}' (score: {match.get('match_score', 'N/A')})")
        
        # Step 4: Convert to list and sort
        lookup_data = list(all_lookup_data.values())
        lookup_data.sort(key=lambda x: x['boss_name'] if x['boss_name'] else x['afterclass_name'])
        
        # Step 5: Save main lookup file
        df = pd.DataFrame(lookup_data)
        df.to_csv(input_lookup_file, index=False)
        
        # Step 6: Save separate tracking files for manual review
        if hasattr(self, 'partial_matches') and self.partial_matches:
            partial_df = pd.DataFrame(self.partial_matches)
            partial_path = os.path.join(self.verify_dir, 'partial_matches_log.csv')
            partial_df.to_csv(partial_path, index=False)
            logger.info(f"🔍 Saved {len(self.partial_matches)} partial matches to partial_matches_log.csv")
        
        logger.info(f"✅ Updated professor_lookup.csv:")
        logger.info(f"   • Total entries: {len(lookup_data)}")
        logger.info(f"   • New entries added: {new_entries_count}")
        logger.info(f"   • Existing entries updated: {updated_entries_count}")
        
        # Step 7: Log summary of different methods
        method_counts = {}
        for entry in lookup_data:
            method = entry.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logger.info("📊 Entries by method:")
        for method, count in sorted(method_counts.items()):
            logger.info(f"   • {method}: {count}")

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"✅ Professors created: {self.stats['professors_created']}")
        print(f"✅ Courses created: {self.stats['courses_created']}")
        print(f"✅ Courses updated: {self.stats['courses_updated']}")
        print(f"⚠️  Courses needing faculty: {self.stats['courses_needing_faculty']}")
        print(f"✅ Classes created: {self.stats['classes_created']}")
        print(f"✅ Class timings created: {self.stats['timings_created']}")
        print(f"✅ Exam timings created: {self.stats['exams_created']}")
        print("="*70)
        
        print("\n📁 OUTPUT FILES:")
        print(f"   Verify folder: {self.verify_dir}/")
        print(f"   - new_professors.csv ({self.stats['professors_created']} records)")
        print(f"   - new_courses.csv ({self.stats['courses_created']} records)")
        print(f"   Output folder: {self.output_base}/")
        print(f"   - update_courses.csv ({self.stats['courses_updated']} records)")
        print(f"   - new_acad_term.csv ({len(self.new_acad_terms)} records)")
        print(f"   - new_classes.csv ({self.stats['classes_created']} records)")
        print(f"   - new_class_timing.csv ({self.stats['timings_created']} records)")
        print(f"   - new_class_exam_timing.csv ({self.stats['exams_created']} records)")
        print(f"   - professor_lookup.csv (updated)")
        print(f"   - courses_needing_faculty.csv ({self.stats['courses_needing_faculty']} records)")
        print("="*70)

    def run_phase1_professors_and_courses(self):
        """Phase 1: Process professors and courses with automated faculty mapping and cache checking"""
        try:
            logger.info("🚀 Starting Phase 1: Professors and Courses with Cache Checking")
            logger.info("="*60)
            
            # Step 1: Load or cache database data.
            # The new _load_from_cache now handles all professor lookup validation and synchronization internally.
            if not self.load_or_cache_data_with_freshness_check():
                logger.error("❌ Failed to load or validate database data")
                return False
            
            # Step 2: Load the raw input data from Excel
            if not self.load_raw_data():
                logger.error("❌ Failed to load raw data")
                return False
            
            # Step 3: Process the data using the now-validated caches
            logger.info("\n🎓 Running automated faculty mapping...")
            self.process_professors()
            self.process_courses()
            
            try:
                self.map_courses_to_faculties_from_boss()
            except Exception as e:
                logger.warning(f"⚠️ Automated faculty mapping failed: {e}")
                logger.info("  Continuing with manual faculty assignment...")
            
            self.process_acad_terms()
            
            # Step 4: Save phase 1 outputs
            self._save_phase1_outputs()
            
            # Step 5: Print faculty mapping summary
            if hasattr(self, 'courses_needing_faculty') and self.courses_needing_faculty:
                logger.info(f"\n📋 Faculty Assignment Summary:")
                logger.info(f"  • Automated mappings applied to {self.stats.get('courses_created', 0) - len(self.courses_needing_faculty)} courses")
                logger.info(f"  • {len(self.courses_needing_faculty)} courses still need manual review")
                
                if len(self.courses_needing_faculty) <= 10:
                    logger.info(f"  Courses needing manual review:")
                    for course_info in self.courses_needing_faculty:
                        logger.info(f"    - {course_info['course_code']}: {course_info['course_name']}")
            
            logger.info("✅ Phase 1 completed - Review files in verify/ folder")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 1 failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_phase2_remaining_tables(self):
        """Phase 2: Process classes and timings after professor correction with cache checking"""
        try:
            logger.info("🚀 Starting Phase 2: Classes and Timings with Cache Checking")
            logger.info("="*60)
            
            # Set phase 2 mode to prevent overwriting corrected professors
            self._phase2_mode = True
            
            # Ensure cache is fresh
            if not self.load_or_cache_data_with_freshness_check():
                logger.error("❌ Failed to load fresh database data")
                return False
            
            # Update professor lookup from corrected CSV
            if not self.update_professor_lookup_from_corrected_csv():
                logger.error("❌ Failed to update professor lookup")
                return False
            
            # Update professors with missing boss_names
            self.update_professors_with_boss_names()
            
            # Process remaining tables with cache checking
            if not self.process_remaining_tables():
                logger.error("❌ Failed to process remaining tables")
                return False
            
            # Save all outputs
            self.save_outputs()
            
            # Print summary
            self.print_summary()
            
            logger.info("✅ Phase 2 completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 2 failed: {e}")
            return False

    def _save_phase1_outputs(self):
        """Save Phase 1 outputs (professors, courses, acad_terms)"""
        # Save new professors (to verify folder for manual correction)
        # Always create the file, even if empty
        df = pd.DataFrame(self.new_professors) if self.new_professors else pd.DataFrame(columns=['id', 'name', 'boss_name', 'afterclass_name', 'original_scraped_name'])
        df.to_csv(os.path.join(self.verify_dir, 'new_professors.csv'), index=False)
        if self.new_professors:
            logger.info(f"✅ Saved {len(self.new_professors)} new professors for review")
        else:
            logger.info(f"✅ Created empty new_professors.csv (all professors already exist)")
        
        # Save new courses (to verify folder)
        if self.new_courses:
            df = pd.DataFrame(self.new_courses)
            df.to_csv(os.path.join(self.verify_dir, 'new_courses.csv'), index=False)
            logger.info(f"✅ Saved {len(self.new_courses)} new courses")
        
        # Save course updates
        if self.update_courses:
            df = pd.DataFrame(self.update_courses)
            df.to_csv(os.path.join(self.output_base, 'update_courses.csv'), index=False)
            logger.info(f"✅ Saved {len(self.update_courses)} course updates")
        
        # Save academic terms
        if self.new_acad_terms:
            df = pd.DataFrame(self.new_acad_terms)
            df.to_csv(os.path.join(self.output_base, 'new_acad_term.csv'), index=False)
            logger.info(f"✅ Saved {len(self.new_acad_terms)} academic terms")

    def setup_boss_processing(self):
        """Initialize BOSS results processing with logging and caches"""
        # Setup logging for BOSS processing
        self.boss_log_file = os.path.join(self.output_base, 'boss_result_log.txt')
        
        # Create the log file and write header
        try:
            with open(self.boss_log_file, 'w') as f:
                f.write(f"BOSS Results Processing Log - {datetime.now().isoformat()}\n")
                f.write("="*70 + "\n\n")
            print(f"📝 Log file created: {self.boss_log_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not create log file {self.boss_log_file}: {e}")
            self.boss_log_file = None
        
        # Initialize existing classes cache
        self.existing_classes_cache = []
        
        # Data storage for BOSS results
        self.boss_data = []
        self.failed_mappings = []
        
        # Output collectors
        self.new_bid_windows = []
        self.new_class_availability = []
        self.new_bid_result = []
        self.update_bid_result = []
        
        # Caches for deduplication
        self.bid_window_cache = {}  # (acad_term_id, round, window) -> bid_window_id
        
        # PROPERLY INITIALIZE bid_window_id_counter from database cache
        self.bid_window_id_counter = 1  # Default fallback
        
        # Load existing bid_window data and find max ID
        try:
            cache_file = os.path.join(self.cache_dir, 'bid_window_cache.pkl')
            if os.path.exists(cache_file):
                bid_window_df = pd.read_pickle(cache_file)
                if not bid_window_df.empty:
                    max_id = bid_window_df['id'].max()
                    self.bid_window_id_counter = max_id + 1
                    
                    # Build deduplication cache
                    for _, row in bid_window_df.iterrows():
                        window_key = (row['acad_term_id'], row['round'], row['window'])
                        self.bid_window_cache[window_key] = row['id']
                    
                    self.log_boss_activity(f"✅ Loaded {len(bid_window_df)} existing bid windows, next ID will be {self.bid_window_id_counter}")
                else:
                    self.log_boss_activity("⚠️ Bid window cache exists but is empty, starting from ID 1")
            else:
                # Try to download from database if cache doesn't exist
                if hasattr(self, 'connection') and self.connection:
                    try:
                        query = "SELECT * FROM bid_window ORDER BY id"
                        bid_window_df = pd.read_sql_query(query, self.connection)
                        if not bid_window_df.empty:
                            # Save to cache for future use
                            bid_window_df.to_pickle(cache_file)
                            
                            max_id = bid_window_df['id'].max()
                            self.bid_window_id_counter = max_id + 1
                            
                            # Build deduplication cache
                            for _, row in bid_window_df.iterrows():
                                window_key = (row['acad_term_id'], row['round'], row['window'])
                                self.bid_window_cache[window_key] = row['id']
                            
                            self.log_boss_activity(f"✅ Downloaded {len(bid_window_df)} bid windows from database, next ID will be {self.bid_window_id_counter}")
                        else:
                            self.log_boss_activity("⚠️ Database bid_window table is empty, starting from ID 1")
                    except Exception as e:
                        self.log_boss_activity(f"⚠️ Could not download bid_window from database: {e}")
                else:
                    self.log_boss_activity("⚠️ No bid window cache found and no database connection, starting from ID 1")
        
        except Exception as e:
            self.log_boss_activity(f"⚠️ Error initializing bid_window counter: {e}")
            self.bid_window_id_counter = 1
        
        # Statistics
        self.boss_stats = {
            'files_processed': 0,
            'total_rows': 0,
            'bid_windows_created': 0,
            'class_availability_created': 0,
            'bid_results_created': 0,
            'failed_mappings': 0
        }
        
        print("🔄 BOSS results processing setup completed")

    def log_boss_activity(self, message, print_to_stdout=True):
        """Log activity to both file and optionally stdout"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Only write to file if boss_log_file exists (after setup_boss_processing is called)
        if hasattr(self, 'boss_log_file') and self.boss_log_file:
            try:
                with open(self.boss_log_file, 'a') as f:
                    f.write(log_message)
            except Exception as e:
                print(f"⚠️ Warning: Could not write to log file: {e}")
        
        if print_to_stdout:
            print(f"📝 {message}")

    def parse_bidding_window(self, bidding_window_str):
        """Complete parser for bidding window string to extract round and window
        
        Examples:
        "Round 1 Window 1" -> ("1", 1)
        "Round 1A Window 2" -> ("1A", 2)
        "Round 2A Window 3" -> ("2A", 3)
        "Incoming Exchange Rnd 1C Win 1" -> ("1C", 1)
        "Incoming Freshmen Rnd 1 Win 4" -> ("1F", 4)
        """
        if not bidding_window_str or pd.isna(bidding_window_str):
            return None, None
        
        # Clean the string
        bidding_window_str = str(bidding_window_str).strip()
        
        # Pattern 1: Standard format "Round X[A/B/C] Window Y"
        pattern1 = r'Round\s+(\w+)\s+Window\s+(\d+)'
        match1 = re.match(pattern1, bidding_window_str)
        if match1:
            round_str = match1.group(1)
            window_num = int(match1.group(2))
            return round_str, window_num
        
        # Pattern 2: Incoming Exchange format "Incoming Exchange Rnd X[A/B/C] Win Y"
        # Map to same round but keep distinction if needed
        pattern2 = r'Incoming\s+Exchange\s+Rnd\s+(\w+)\s+Win\s+(\d+)'
        match2 = re.match(pattern2, bidding_window_str)
        if match2:
            round_str = match2.group(1)  # Keep original round (1C)
            window_num = int(match2.group(2))
            return round_str, window_num
        
        # Pattern 3: Incoming Freshmen format "Incoming Freshmen Rnd X Win Y"
        # Map Round 1 -> Round 1F for distinction
        pattern3 = r'Incoming\s+Freshmen\s+Rnd\s+(\w+)\s+Win\s+(\d+)'
        match3 = re.match(pattern3, bidding_window_str)
        if match3:
            original_round = match3.group(1)
            window_num = int(match3.group(2))
            # Map Incoming Freshmen Round 1 to Round 1F
            if original_round == "1":
                round_str = "1F"
            else:
                round_str = f"{original_round}F"  # For other rounds if they exist
            return round_str, window_num
        
        return None, None

    def load_boss_results(self):
        """Load BOSS results from raw_data.xlsx standalone sheet"""
        self.log_boss_activity("🔍 Loading BOSS results from raw_data.xlsx...")
        
        # Use existing standalone_data that's already loaded
        if not hasattr(self, 'standalone_data') or self.standalone_data is None:
            self.log_boss_activity("❌ No standalone data loaded")
            return False
        
        # Filter rows that have bidding data using the correct column names
        bidding_data = self.standalone_data[
            self.standalone_data['bidding_window'].notna() & 
            self.standalone_data['total'].notna()
        ].copy()
        
        if bidding_data.empty:
            self.log_boss_activity("❌ No bidding data found in raw_data.xlsx")
            return False
        
        self.boss_data = bidding_data
        self.boss_stats['total_rows'] = len(self.boss_data)
        self.boss_stats['files_processed'] = 1
        
        self.log_boss_activity(f"✅ Loaded {self.boss_stats['total_rows']} bidding records from raw_data.xlsx")
        return True

    def process_bid_windows(self):
        """Process and create bid_window entries from raw_data bidding_window column"""
        self.log_boss_activity("🪟 Processing bid windows from raw_data...")
        
        if self.boss_data is None or len(self.boss_data) == 0:
            self.log_boss_activity("❌ No BOSS data loaded")
            return False
        
        # Track all unique bid windows found in data
        found_windows = defaultdict(set)  # acad_term_id -> set of (round, window) tuples
        
        # Discover all windows that exist in the data
        for _, row in self.boss_data.iterrows():
            acad_term_id = row.get('acad_term_id')
            bidding_window_str = row.get('bidding_window')
            
            if pd.isna(acad_term_id) or pd.isna(bidding_window_str):
                continue
            
            round_str, window_num = self.parse_bidding_window(bidding_window_str)
            
            if acad_term_id and round_str and window_num:
                found_windows[acad_term_id].add((round_str, window_num))
        
        # Use the counter that was set from existing data
        bid_window_id = self.bid_window_id_counter
        round_order = {'1': 1, '1A': 2, '1B': 3, '1C': 4, '1F': 5, '2': 6, '2A': 7}
        
        for acad_term_id in sorted(found_windows.keys()):
            windows_for_term = found_windows[acad_term_id]
            sorted_windows = sorted(windows_for_term, key=lambda x: (round_order.get(x[0], 99), x[1]))
            
            self.log_boss_activity(f"📅 Processing {acad_term_id}: found {len(sorted_windows)} windows")
            
            for round_str, window_num in sorted_windows:
                window_key = (acad_term_id, round_str, window_num)
                
                # Skip if already exists in database
                if window_key in self.bid_window_cache:
                    self.log_boss_activity(f"⏭️ Bid window already exists: {acad_term_id} Round {round_str} Window {window_num}")
                    continue
                
                new_bid_window = {
                    'id': bid_window_id,
                    'acad_term_id': acad_term_id,
                    'round': round_str,
                    'window': window_num
                }
                
                self.new_bid_windows.append(new_bid_window)
                self.bid_window_cache[window_key] = bid_window_id
                self.boss_stats['bid_windows_created'] += 1
                
                self.log_boss_activity(f"✅ Created bid_window {bid_window_id}: {acad_term_id} Round {round_str} Window {window_num}")
                bid_window_id += 1
        
        self.bid_window_id_counter = bid_window_id
        self.log_boss_activity(f"✅ Created {self.boss_stats['bid_windows_created']} bid windows")
        return True

    def get_course_id(self, course_code):
        """Get course_id from course_code, checking multiple sources"""
        # Check courses cache (from database)
        if course_code in self.courses_cache:
            return self.courses_cache[course_code]['id']
        
        # Check in new_courses (newly created)
        for course in self.new_courses:
            if course['code'] == course_code:
                return course['id']
        
        # Check new_courses.csv file
        try:
            new_courses_path = os.path.join(self.output_base, 'new_courses.csv')
            verify_courses_path = os.path.join(self.verify_dir, 'new_courses.csv')
            
            for path in [verify_courses_path, new_courses_path]:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    matching_courses = df[df['code'] == course_code]
                    if not matching_courses.empty:
                        return matching_courses.iloc[0]['id']
        except Exception as e:
            self.log_boss_activity(f"⚠️ Error reading new_courses.csv: {e}", print_to_stdout=False)
        
        return None

    def load_existing_classes_cache(self):
        """Load existing classes from database cache with proper full extraction"""
        self.existing_classes_cache = []
        
        try:
            cache_file = os.path.join(self.cache_dir, 'classes_cache.pkl')
            
            # Try loading from cache file first
            if os.path.exists(cache_file):
                try:
                    classes_df = pd.read_pickle(cache_file)
                    if not classes_df.empty:
                        self.existing_classes_cache = classes_df.to_dict('records')
                        logger.info(f"📚 Loaded {len(self.existing_classes_cache)} existing classes from cache")
                        return
                    else:
                        logger.info("⚠️ Cache file exists but is empty")
                except Exception as e:
                    logger.warning(f"⚠️ Error reading cache file: {e}")
            
            # If cache doesn't exist or is empty, try database with SELECT *
            if self.connection:
                try:
                    query = "SELECT * FROM classes"
                    classes_df = pd.read_sql_query(query, self.connection)
                    if not classes_df.empty:
                        # Save to cache for future use
                        classes_df.to_pickle(cache_file)
                        self.existing_classes_cache = classes_df.to_dict('records')
                        logger.info(f"📚 Downloaded and cached {len(self.existing_classes_cache)} existing classes")
                        return
                    else:
                        logger.warning("⚠️ Database classes table is empty")
                except Exception as e:
                    logger.warning(f"⚠️ Error downloading classes from database: {e}")
            
            # Final fallback
            logger.warning("⚠️ All class loading methods failed - using empty cache")
                    
        except Exception as e:
            self.existing_classes_cache = []
            logger.error(f"⚠️ Critical error in load_existing_classes_cache: {e}")

    def process_class_availability(self):
        """
        Process class availability data, preventing the creation of duplicate records
        by checking against existing cache data first.
        """
        self.log_boss_activity("📊 Processing class availability from raw_data...")
        
        # === STEP 1: Determine Current Bidding Window ===
        now = datetime.now()
        current_window_name = None
        
        # Get the bidding schedule for the current term
        bidding_schedule_for_term = BIDDING_SCHEDULES.get(START_AY_TERM, [])
        
        if bidding_schedule_for_term:
            # Find the current window (first future window)
            for i, (results_date, window_name, folder_suffix) in enumerate(bidding_schedule_for_term):
                if now < results_date:
                    current_window_name = window_name
                    break
            
            # If no future window found, we're past all scheduled windows
            if current_window_name is None and bidding_schedule_for_term:
                # Use the last window as current
                current_window_name = bidding_schedule_for_term[-1][1]
        
        logger.info(f"🎯 Processing class availability for current window: '{current_window_name}'")

        # === STEP 2: Filter the data to only current window and current term records ===
        if current_window_name and hasattr(self, 'standalone_data') and not self.standalone_data.empty:
            if 'bidding_window' in self.standalone_data.columns:
                original_count = len(self.standalone_data)
                
                # Filter by bidding window
                current_window_data = self.standalone_data[
                    self.standalone_data['bidding_window'] == current_window_name
                ].copy()
                
                # Also filter by current academic term to prevent cross-term contamination
                if 'acad_term_id' in current_window_data.columns:
                    # Extract expected term from START_AY_TERM (e.g., '2025-26_T1' -> 'AY202526T1')
                    expected_term_id = START_AY_TERM.replace('-', '').replace('_', '')
                    expected_term_id = f"AY{expected_term_id}"
                    
                    before_term_filter = len(current_window_data)
                    current_window_data = current_window_data[
                        current_window_data['acad_term_id'] == expected_term_id
                    ].copy()
                    
                    self.log_boss_activity(f"🔽 Filtered data: {original_count} → {before_term_filter} (window) → {len(current_window_data)} (window + term)")
                    self.log_boss_activity(f"    Window filter: '{current_window_name}', Term filter: '{expected_term_id}'")
                else:
                    self.log_boss_activity(f"🔽 Filtered data from {original_count} to {len(current_window_data)} records for current window: '{current_window_name}'")
            else:
                self.log_boss_activity("⚠️ No 'bidding_window' column found - processing all data")
                current_window_data = self.standalone_data.copy()
        else:
            self.log_boss_activity("⚠️ Could not determine current window or no standalone data - processing all data")
            current_window_data = self.standalone_data.copy() if hasattr(self, 'standalone_data') else pd.DataFrame()
        
        # Load existing class availability data to prevent duplicates
        existing_availability_keys = set()
        cache_file = os.path.join(self.cache_dir, 'class_availability_cache.pkl')
        if os.path.exists(cache_file):
            try:
                existing_df = pd.read_pickle(cache_file)
                if not existing_df.empty:
                    for _, record in existing_df.iterrows():
                        key = (record['class_id'], record['bid_window_id'])
                        existing_availability_keys.add(key)
                    self.log_boss_activity(f"✅ Pre-loaded {len(existing_availability_keys)} existing class availability keys from cache.")
            except Exception as e:
                self.log_boss_activity(f"⚠️ Could not pre-load class_availability_cache: {e}")
        
        # ADDED: Track keys from current run to prevent duplicates within the same processing
        current_run_keys = set()
        for availability_record in self.new_class_availability:
            key = (availability_record['class_id'], availability_record['bid_window_id'])
            current_run_keys.add(key)

        newly_created_count = 0
        updated_count = 0
        
        # === STEP 3: Process only the filtered current window data ===
        for _, row in current_window_data.iterrows():
            course_code = row.get('course_code')
            section = row.get('section')
            acad_term_id = row.get('acad_term_id')
            bidding_window_str = row.get('bidding_window')
            
            if pd.isna(course_code) or pd.isna(section) or pd.isna(acad_term_id) or pd.isna(bidding_window_str):
                continue
            
            round_str, window_num = self.parse_bidding_window(bidding_window_str)
            if not all([round_str, window_num]):
                continue
            
            class_boss_id = row.get('class_boss_id')
            class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)

            if not class_ids:
                failed_row = {
                    'course_code': course_code, 'section': section, 'acad_term_id': acad_term_id,
                    'bidding_window_str': bidding_window_str, 'reason': 'class_not_found'
                }
                self.failed_mappings.append(failed_row)
                self.boss_stats['failed_mappings'] += 1
                continue
            
            window_key = (acad_term_id, round_str, window_num)
            bid_window_id = self.bid_window_cache.get(window_key)
            if not bid_window_id:
                self.log_boss_activity(f"⚠️ No bid_window_id for {window_key}")
                continue
            
            # Extract values safely
            total_val = int(row.get('total')) if pd.notna(row.get('total')) else 0
            current_enrolled_val = int(row.get('current_enrolled')) if pd.notna(row.get('current_enrolled')) else 0
            reserved_val = int(row.get('reserved')) if pd.notna(row.get('reserved')) else 0
            available_val = int(row.get('available')) if pd.notna(row.get('available')) else 0
            
            for class_id in class_ids:
                # Check for existence in both existing data and current run
                availability_key = (class_id, bid_window_id)
                
                # FIXED: Check both existing and current run keys
                if availability_key in existing_availability_keys or availability_key in current_run_keys:
                    # Check if update is needed
                    if availability_key in existing_availability_keys:
                        # Could implement update logic here if needed
                        pass
                    continue

                # Create new record
                availability_record = {
                    'class_id': class_id,
                    'bid_window_id': bid_window_id,
                    'total': total_val,
                    'current_enrolled': current_enrolled_val,
                    'reserved': reserved_val,
                    'available': available_val
                }
                
                self.new_class_availability.append(availability_record)
                current_run_keys.add(availability_key)
                newly_created_count += 1
        
        self.boss_stats['class_availability_created'] = newly_created_count
        self.log_boss_activity(f"✅ Class availability checks complete. Created {newly_created_count} new records, Updated {updated_count} records.")
        return True

    def process_bid_results(self):
        """
        Process bid data from raw_data.xlsx. Creates update records when median/min data exists,
        and new records only when they don't exist yet.
        """
        logger.info("📈 Processing bid results from raw_data...")
        
        # Ensure the list for update records exists
        if not hasattr(self, 'update_bid_result'):
            self.update_bid_result = []

        # === STEP 1: Determine Current and Previous Bidding Windows ===
        now = datetime.now()
        current_window_name = None
        previous_window_name = None
        
        # Get the bidding schedule for the current term
        bidding_schedule_for_term = BIDDING_SCHEDULES.get(START_AY_TERM, [])
        
        if bidding_schedule_for_term:
            # Find the current window (first future window) and previous window
            for i, (results_date, window_name, folder_suffix) in enumerate(bidding_schedule_for_term):
                if now < results_date:
                    current_window_name = window_name
                    # Previous window is the one before current (if exists)
                    if i > 0:
                        previous_window_name = bidding_schedule_for_term[i-1][1]
                    break
            
            # If no future window found, we're past all scheduled windows
            if current_window_name is None and bidding_schedule_for_term:
                # Use the last window as current, and second-to-last as previous
                current_window_name = bidding_schedule_for_term[-1][1]
                if len(bidding_schedule_for_term) > 1:
                    previous_window_name = bidding_schedule_for_term[-2][1]
        
        logger.info(f"🎯 Determined bidding windows - Current: '{current_window_name}', Previous: '{previous_window_name}'")

        # === STEP 2: Filter the data to only current window and current term records ===
        # For new bid results, we only want data from the CURRENT window and CURRENT term
        if current_window_name and hasattr(self, 'standalone_data') and not self.standalone_data.empty:
            if 'bidding_window' in self.standalone_data.columns:
                original_count = len(self.standalone_data)
                
                # Filter by bidding window
                current_window_data = self.standalone_data[
                    self.standalone_data['bidding_window'] == current_window_name
                ].copy()
                
                # Also filter by current academic term to prevent cross-term contamination
                if 'acad_term_id' in current_window_data.columns:
                    # Extract expected term from START_AY_TERM (e.g., '2025-26_T1' -> 'AY202526T1')
                    expected_term_id = START_AY_TERM.replace('-', '').replace('_', '')
                    expected_term_id = f"AY{expected_term_id}"
                    
                    before_term_filter = len(current_window_data)
                    current_window_data = current_window_data[
                        current_window_data['acad_term_id'] == expected_term_id
                    ].copy()
                    
                    logger.info(f"🔽 Filtered standalone_data: {original_count} → {before_term_filter} (window) → {len(current_window_data)} (window + term)")
                    logger.info(f"    Window filter: '{current_window_name}', Term filter: '{expected_term_id}'")
                else:
                    logger.info(f"🔽 Filtered standalone_data from {original_count} to {len(current_window_data)} records for current window: '{current_window_name}'")
            else:
                logger.warning("⚠️ No 'bidding_window' column found - processing all data")
                current_window_data = self.standalone_data.copy()
        else:
            logger.warning("⚠️ Could not determine current window or no standalone data - processing all data")
            current_window_data = self.standalone_data.copy() if hasattr(self, 'standalone_data') else pd.DataFrame()

        # Load existing bid_result data to check for duplicates
        existing_bid_result_keys = set()
        existing_bid_results = {}  # Store full records for update comparison
        cache_file = os.path.join(self.cache_dir, 'bid_result_cache.pkl')
        if os.path.exists(cache_file):
            try:
                existing_df = pd.read_pickle(cache_file)
                if not existing_df.empty:
                    for _, record in existing_df.iterrows():
                        key = (record['bid_window_id'], record['class_id'])
                        existing_bid_result_keys.add(key)
                        existing_bid_results[key] = record.to_dict()
                    self.log_boss_activity(f"✅ Pre-loaded {len(existing_bid_result_keys)} existing bid result keys from cache.")
            except Exception as e:
                self.log_boss_activity(f"⚠️ Could not pre-load bid_result_cache: {e}")

        newly_created_count = 0
        updated_count = 0
        
        # DEBUG: Check column names in the data
        if not current_window_data.empty:
            # Check for any rows with median/min data
            median_cols = [col for col in current_window_data.columns if 'median' in col.lower() or 'bid' in col.lower()]
            min_cols = [col for col in current_window_data.columns if 'min' in col.lower()]
        
        # === STEP 3: Process only the filtered current window data ===
        for idx, row in current_window_data.iterrows():
            try:
                course_code = row.get('course_code')
                section = row.get('section')
                acad_term_id = row.get('acad_term_id')
                class_boss_id = row.get('class_boss_id')
                bidding_window_str = row.get('bidding_window')
                
                if pd.isna(acad_term_id) or pd.isna(class_boss_id):
                    continue
                
                round_str, window_num = self.parse_bidding_window(bidding_window_str)
                if not all([round_str, window_num]):
                    continue
                
                class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)
                if not class_ids:
                    continue
                
                window_key = (acad_term_id, round_str, window_num)
                bid_window_id = self.bid_window_cache.get(window_key)
                if not bid_window_id:
                    continue

                # FIXED: Check all possible column names for median and min
                median_bid = None
                min_bid = None
                
                # Try all possible column names for median
                median_column_names = ['median', 'Median', 'Median Bid', 'median_bid', 'Median_Bid', 'MEDIAN']
                for col_name in median_column_names:
                    if col_name in row.index:
                        val = row[col_name]
                        if pd.notna(val):
                            median_bid = val
                            if idx < 5:  # Log first few for debugging
                                self.log_boss_activity(f"🔍 DEBUG Row {idx}: Found median value {median_bid} in column '{col_name}'")
                            break
                
                # Try all possible column names for min
                min_column_names = ['min', 'Min', 'Min Bid', 'min_bid', 'Min_Bid', 'MIN']
                for col_name in min_column_names:
                    if col_name in row.index:
                        val = row[col_name]
                        if pd.notna(val):
                            min_bid = val
                            if idx < 5:  # Log first few for debugging
                                self.log_boss_activity(f"🔍 DEBUG Row {idx}: Found min value {min_bid} in column '{col_name}'")
                            break
                
                has_bid_data = pd.notna(median_bid) or pd.notna(min_bid)
                
                # DEBUG: Log if we have bid data
                if has_bid_data and idx < 10:
                    self.log_boss_activity(f"🔍 DEBUG: Row {idx} {course_code}-{section} has bid data: median={median_bid}, min={min_bid}")

                # Prepare data record
                def safe_int(val): return int(val) if pd.notna(val) else None
                def safe_float(val): return float(val) if pd.notna(val) else None
                
                total_val = safe_int(row.get('total'))
                enrolled_val = safe_int(row.get('current_enrolled'))
                
                for class_id in class_ids:
                    # Check if record exists
                    bid_result_key = (bid_window_id, class_id)
                    
                    result_data = {
                        'bid_window_id': bid_window_id, 
                        'class_id': class_id,
                        'vacancy': total_val,
                        'opening_vacancy': safe_int(row.get('opening_vacancy')),
                        'before_process_vacancy': total_val - enrolled_val if total_val is not None and enrolled_val is not None else None,
                        'dice': safe_int(row.get('d_i_c_e') or row.get('dice')),
                        'after_process_vacancy': safe_int(row.get('after_process_vacancy')),
                        'enrolled_students': enrolled_val,
                        'median': safe_float(median_bid),
                        'min': safe_float(min_bid)
                    }

                    if bid_result_key in existing_bid_result_keys:
                        # Check if update is needed
                        existing_record = existing_bid_results.get(bid_result_key, {})
                        needs_update = False
                        
                        # Check if median or min values have changed
                        if has_bid_data:
                            if (pd.notna(median_bid) and safe_float(median_bid) != existing_record.get('median')):
                                needs_update = True
                            if (pd.notna(min_bid) and safe_float(min_bid) != existing_record.get('min')):
                                needs_update = True
                        
                        # Also check other fields for updates
                        for field in ['vacancy', 'opening_vacancy', 'before_process_vacancy', 'dice', 
                                    'after_process_vacancy', 'enrolled_students']:
                            if result_data.get(field) is not None and result_data[field] != existing_record.get(field):
                                needs_update = True
                        
                        if needs_update:
                            self.update_bid_result.append(result_data)
                            updated_count += 1
                            if has_bid_data:
                                self.log_boss_activity(f"📊 Update bid_result: {course_code}-{section} with median={median_bid}, min={min_bid}")
                    else:
                        # This is a NEW record
                        self.new_bid_result.append(result_data)
                        existing_bid_result_keys.add(bid_result_key)
                        newly_created_count += 1
            
            except Exception as e:
                logger.error(f"Error processing bid result row for {row.get('course_code')}-{row.get('section')}: {e}")

        self.boss_stats['bid_results_created'] += newly_created_count
        self.log_boss_activity(f"✅ Bid result checks complete. Created: {newly_created_count}, Updated: {updated_count}.")
        return True

    def find_all_class_ids(self, acad_term_id, class_boss_id):
        """Finds all class_ids for a given acad_term_id and class_boss_id.
        Returns ALL class records for multi-professor classes."""
        
        if pd.isna(acad_term_id) or pd.isna(class_boss_id):
            return []

        found_class_ids = []

        # Source 1: Check newly created classes in this run
        if hasattr(self, 'new_classes') and self.new_classes:
            for class_obj in self.new_classes:
                if (class_obj.get('acad_term_id') == acad_term_id and 
                    str(class_obj.get('boss_id')) == str(class_boss_id)):
                    found_class_ids.append(class_obj['id'])

        # Source 2: Check classes that existed before this run (from cache)
        if hasattr(self, 'existing_classes_cache') and self.existing_classes_cache:
            for class_obj in self.existing_classes_cache:
                if (class_obj.get('acad_term_id') == acad_term_id and 
                    str(class_obj.get('boss_id')) == str(class_boss_id)):
                    found_class_ids.append(class_obj['id'])
        
        # Source 3: Check new_classes.csv file if cache is incomplete
        try:
            new_classes_path = os.path.join(self.output_base, 'new_classes.csv')
            if os.path.exists(new_classes_path):
                df = pd.read_csv(new_classes_path)
                # This logic now correctly searches the CSV using boss_id
                matching_classes = df[
                    (df['acad_term_id'] == acad_term_id) & 
                    (df['boss_id'].astype(str) == str(class_boss_id))
                ]
                for _, row in matching_classes.iterrows():
                    if row['id'] not in found_class_ids:
                        found_class_ids.append(row['id'])
        except Exception as e:
            self.log_boss_activity(f"⚠️ Error reading new_classes.csv: {e}", print_to_stdout=False)
        
        # Remove duplicates while preserving order
        unique_class_ids = []
        seen = set()
        for class_id in found_class_ids:
            if class_id not in seen:
                unique_class_ids.append(class_id)
                seen.add(class_id)
        
        # Debug logging for multi-professor classes
        if len(unique_class_ids) > 1:
            self.log_boss_activity(f"📚 Found {len(unique_class_ids)} class records for boss_id {class_boss_id}: multi-professor class")
        
        return unique_class_ids

    def save_boss_outputs(self):
        """Save all BOSS-related output files"""
        self.log_boss_activity("💾 Saving BOSS output files...")
        
        # Save bid windows
        if self.new_bid_windows:
            df = pd.DataFrame(self.new_bid_windows)
            output_path = os.path.join(self.output_base, 'new_bid_window.csv')
            df.to_csv(output_path, index=False)
            self.log_boss_activity(f"✅ Saved {len(self.new_bid_windows)} bid windows to new_bid_window.csv")
        
        # Save class availability
        if self.new_class_availability:
            df = pd.DataFrame(self.new_class_availability)
            output_path = os.path.join(self.output_base, 'new_class_availability.csv')
            df.to_csv(output_path, index=False)
            self.log_boss_activity(f"✅ Saved {len(self.new_class_availability)} availability records to new_class_availability.csv")
        
        # Save bid results
        if self.new_bid_result:
            df = pd.DataFrame(self.new_bid_result)
            output_path = os.path.join(self.output_base, 'new_bid_result.csv')
            df.to_csv(output_path, index=False)
            self.log_boss_activity(f"✅ Saved {len(self.new_bid_result)} bid results to new_bid_result.csv")
        
        # Save failed mappings
        if self.failed_mappings:
            df = pd.DataFrame(self.failed_mappings)
            output_path = os.path.join(self.output_base, 'failed_boss_results_mapping.csv')
            df.to_csv(output_path, index=False)
            self.log_boss_activity(f"⚠️ Saved {len(self.failed_mappings)} failed mappings to failed_boss_results_mapping.csv")
        
        self.log_boss_activity("✅ All BOSS output files saved successfully")

    def print_boss_summary(self):
        """Print BOSS processing summary"""
        print("\n" + "="*70)
        print("📊 BOSS RESULTS PROCESSING SUMMARY")
        print("="*70)
        print(f"📂 Files processed: {self.boss_stats['files_processed']}")
        print(f"📄 Total rows: {self.boss_stats['total_rows']}")
        print(f"🪟 Bid windows created: {self.boss_stats['bid_windows_created']}")
        print(f"📊 Class availability records: {self.boss_stats['class_availability_created']}")
        print(f"📈 Bid result records: {self.boss_stats['bid_results_created']}")
        print(f"❌ Failed mappings: {self.boss_stats['failed_mappings']}")
        print("="*70)
        
        print("\n📁 OUTPUT FILES:")
        print(f"   - new_bid_window.csv ({self.boss_stats['bid_windows_created']} records)")
        print(f"   - new_class_availability.csv ({self.boss_stats['class_availability_created']} records)")
        print(f"   - new_bid_result.csv ({self.boss_stats['bid_results_created']} records)")
        if self.boss_stats['failed_mappings'] > 0:
            print(f"   - failed_boss_results_mapping.csv ({self.boss_stats['failed_mappings']} records)")
        print(f"   - boss_result_log.txt (processing log)")
        print("="*70)

    def run_phase3_boss_processing(self):
        """
        Orchestrates the entire bidding data processing workflow with a robust, linear order of operations.
        """
        try:
            print("🚀 Starting Enhanced Phase 3: Final, Robust Workflow")
            print("============================================================")

            logger.info("🛠️ Pre-Phase 3: Updating cache with newly created records from previous phases...")
            
            # Update professors cache
            new_profs_path = os.path.join(self.verify_dir, 'new_professors.csv')
            prof_cache_path = os.path.join(self.cache_dir, 'professors_cache.pkl')
            
            if os.path.exists(new_profs_path) and os.path.exists(prof_cache_path):
                new_profs_df = pd.read_csv(new_profs_path)
                if not new_profs_df.empty:
                    # Drop columns from new_profs_df that are not in the main professors table
                    # This prevents errors if new_professors.csv has extra columns
                    prof_cache_df = pd.read_pickle(prof_cache_path)
                    new_profs_df = new_profs_df[[col for col in new_profs_df.columns if col in prof_cache_df.columns]]
                    
                    combined_profs_df = pd.concat([prof_cache_df, new_profs_df], ignore_index=True).drop_duplicates(subset=['id'])
                    combined_profs_df.to_pickle(prof_cache_path)
                    logger.info(f"   ✅ Updated professors_cache.pkl with {len(new_profs_df)} new records.")

            # Update classes cache (similar logic)
            new_classes_path = os.path.join(self.output_base, 'new_classes.csv')
            class_cache_path = os.path.join(self.cache_dir, 'classes_cache.pkl')

            if os.path.exists(new_classes_path) and os.path.exists(class_cache_path):
                new_classes_df = pd.read_csv(new_classes_path)
                if not new_classes_df.empty:
                    class_cache_df = pd.read_pickle(class_cache_path)
                    combined_classes_df = pd.concat([class_cache_df, new_classes_df], ignore_index=True).drop_duplicates(subset=['id'])
                    combined_classes_df.to_pickle(class_cache_path)
                    logger.info(f"   ✅ Updated classes_cache.pkl with {len(new_classes_df)} new records.")

            self.setup_boss_processing()

            # --- Step 1: Load ALL required data from cache and input files ---
            self.log_boss_activity("🔄 Loading all required data caches with freshness check and combination...")
            # Use the method that combines new data from CSVs before validation
            if not self.load_or_cache_data_with_freshness_check(): return False
            
            self.log_boss_activity("🔄 Loading all raw input files...")
            if not self.load_raw_data(): return False
            self.overall_boss_results_df = self.load_overall_boss_results() # Load this once

            # --- Step 2: Process base entities to establish a stable state ---
            self.log_boss_activity("🔄 Processing base entities (Courses, Classes, Timings)...")
            self.process_acad_terms()
            self.process_professors()
            self.process_courses() 
            self.process_classes()  # This now correctly handles create vs. update
            self.process_timings()  # This now receives stable, non-duplicate class_ids

            # --- Step 3: Sequential catch-up processing for all windows ---
            self.log_boss_activity("🔄 Starting sequential catch-up processing...")

            # Determine current live window and processing range
            now = datetime.now()
            current_window_index = None
            bidding_schedule_for_term = BIDDING_SCHEDULES.get(START_AY_TERM, [])

            if not bidding_schedule_for_term:
                self.log_boss_activity("❌ No bidding schedule found for current term")
                return False

            # Find current live window (first future window)
            for i, (results_date, window_name, folder_suffix) in enumerate(bidding_schedule_for_term):
                if now < results_date:
                    current_window_index = i
                    break

            if current_window_index is None:
                # Past all windows, use last window as current
                current_window_index = len(bidding_schedule_for_term) - 1

            current_window_name = bidding_schedule_for_term[current_window_index][1]
            self.log_boss_activity(f"🎯 Current live window: {current_window_name}")

            # Processing range: from start to current window (inclusive)
            processing_range = bidding_schedule_for_term[:current_window_index + 1]
            self.log_boss_activity(f"📅 Processing {len(processing_range)} windows chronologically")

            # Load data sources once
            if not self.load_raw_data():
                self.log_boss_activity("⚠️ Could not load raw_data.xlsx")
                return False

            self.overall_boss_results_df = self.load_overall_boss_results()
            if self.overall_boss_results_df is None:
                self.log_boss_activity("⚠️ Could not load overallBossResults.xlsx")

            # Load existing bid result keys for deduplication
            existing_bid_result_keys = set()
            cache_file = os.path.join(self.cache_dir, 'bid_result_cache.pkl')
            if os.path.exists(cache_file):
                try:
                    existing_df = pd.read_pickle(cache_file)
                    if not existing_df.empty:
                        for _, record in existing_df.iterrows():
                            key = (record['bid_window_id'], record['class_id'])
                            existing_bid_result_keys.add(key)
                        self.log_boss_activity(f"✅ Pre-loaded {len(existing_bid_result_keys)} existing bid result keys")
                except Exception as e:
                    self.log_boss_activity(f"⚠️ Could not pre-load bid_result_cache: {e}")

            # Sequential processing loop
            for window_index, (results_date, window_name, folder_suffix) in enumerate(processing_range):
                self.log_boss_activity(f"🔄 Processing window {window_index + 1}/{len(processing_range)}: {window_name}")
                
                # Parse window name to get round and window number
                round_str, window_num = self.parse_bidding_window(window_name)
                if not round_str or not window_num:
                    self.log_boss_activity(f"⚠️ Could not parse window: {window_name}")
                    continue
                
                acad_term_id = ACAD_TERM_ID
                window_key = (acad_term_id, round_str, window_num)
                
                # A. Find or create BidWindow record
                bid_window_id = self.bid_window_cache.get(window_key)
                if not bid_window_id:
                    bid_window_id = self.bid_window_id_counter
                    new_bid_window = {
                        'id': bid_window_id,
                        'acad_term_id': acad_term_id,
                        'round': round_str,
                        'window': window_num
                    }
                    self.new_bid_windows.append(new_bid_window)
                    self.bid_window_cache[window_key] = bid_window_id
                    self.bid_window_id_counter += 1
                    self.boss_stats['bid_windows_created'] += 1
                    self.log_boss_activity(f"✅ Created bid window {bid_window_id}: {window_name}")
                
                # B. Process ClassAvailability (check if scrape data exists in raw_data.xlsx)
                window_data_in_raw = self.standalone_data[
                    self.standalone_data['bidding_window'] == window_name
                ] if hasattr(self, 'standalone_data') and self.standalone_data is not None else pd.DataFrame()
                
                if not window_data_in_raw.empty:
                    self.log_boss_activity(f"📊 Processing ClassAvailability for {window_name} from raw_data.xlsx")
                    
                    # Load existing availability keys for deduplication
                    existing_availability_keys = set()
                    avail_cache_file = os.path.join(self.cache_dir, 'class_availability_cache.pkl')
                    if os.path.exists(avail_cache_file):
                        try:
                            avail_df = pd.read_pickle(avail_cache_file)
                            if not avail_df.empty:
                                for _, record in avail_df.iterrows():
                                    key = (record['class_id'], record['bid_window_id'])
                                    existing_availability_keys.add(key)
                        except Exception as e:
                            pass
                    
                    for _, row in window_data_in_raw.iterrows():
                        course_code = row.get('course_code')
                        section = row.get('section')
                        class_boss_id = row.get('class_boss_id')
                        
                        if pd.isna(course_code) or pd.isna(section) or pd.isna(class_boss_id):
                            continue
                        
                        class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)
                        if not class_ids:
                            continue
                        
                        # Extract availability values
                        total_val = int(row.get('total')) if pd.notna(row.get('total')) else 0
                        current_enrolled_val = int(row.get('current_enrolled')) if pd.notna(row.get('current_enrolled')) else 0
                        reserved_val = int(row.get('reserved')) if pd.notna(row.get('reserved')) else 0
                        available_val = int(row.get('available')) if pd.notna(row.get('available')) else 0
                        
                        for class_id in class_ids:
                            availability_key = (class_id, bid_window_id)
                            if availability_key not in existing_availability_keys:
                                availability_record = {
                                    'class_id': class_id,
                                    'bid_window_id': bid_window_id,
                                    'total': total_val,
                                    'current_enrolled': current_enrolled_val,
                                    'reserved': reserved_val,
                                    'available': available_val
                                }
                                self.new_class_availability.append(availability_record)
                                existing_availability_keys.add(availability_key)
                                self.boss_stats['class_availability_created'] += 1
                else:
                    self.log_boss_activity(f"⏭️ Skipping ClassAvailability for {window_name} (no scrape data)")
                
                # C. Process BidResult based on window type
                is_current_live = (window_index == current_window_index)
                
                if is_current_live:
                    # Current live window: create placeholder BidResult from raw_data.xlsx
                    self.log_boss_activity(f"📈 Processing placeholder BidResult for current live window: {window_name}")
                    
                    if not window_data_in_raw.empty:
                        for _, row in window_data_in_raw.iterrows():
                            course_code = row.get('course_code')
                            section = row.get('section')
                            class_boss_id = row.get('class_boss_id')
                            
                            if pd.isna(class_boss_id):
                                continue
                            
                            class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)
                            if not class_ids:
                                continue
                            
                            def safe_int(val): return int(val) if pd.notna(val) else None
                            
                            total_val = safe_int(row.get('total'))
                            enrolled_val = safe_int(row.get('current_enrolled'))
                            
                            for class_id in class_ids:
                                bid_result_key = (bid_window_id, class_id)
                                if bid_result_key not in existing_bid_result_keys:
                                    # Create placeholder record (median/min as None)
                                    result_data = {
                                        'bid_window_id': bid_window_id,
                                        'class_id': class_id,
                                        'vacancy': total_val,
                                        'opening_vacancy': safe_int(row.get('opening_vacancy')),
                                        'before_process_vacancy': total_val - enrolled_val if total_val is not None and enrolled_val is not None else None,
                                        'dice': safe_int(row.get('d_i_c_e') or row.get('dice')),
                                        'after_process_vacancy': safe_int(row.get('after_process_vacancy')),
                                        'enrolled_students': enrolled_val,
                                        'median': None,  # Placeholder
                                        'min': None      # Placeholder
                                    }
                                    self.new_bid_result.append(result_data)
                                    existing_bid_result_keys.add(bid_result_key)
                                    self.boss_stats['bid_results_created'] += 1
                else:
                    # Historical window: process from overallBossResults.xlsx
                    self.log_boss_activity(f"📈 Processing historical BidResult for {window_name} from overallBossResults.xlsx")
                    
                    if self.overall_boss_results_df is not None and not self.overall_boss_results_df.empty:
                        # Filter overall results for this specific window
                        overall_df = self.overall_boss_results_df.copy()
                        
                        # Parse bidding window column
                        bidding_window_col = None
                        for col in overall_df.columns:
                            if 'bidding window' in col.lower() or 'bidding_window' in col.lower():
                                bidding_window_col = col
                                break
                        
                        if bidding_window_col:
                            # Parse and filter for current window
                            parsed_windows = overall_df[bidding_window_col].apply(self.parse_bidding_window)
                            overall_df['round'] = parsed_windows.apply(lambda x: x[0] if isinstance(x, tuple) else None)
                            overall_df['window'] = parsed_windows.apply(lambda x: x[1] if isinstance(x, tuple) else None)
                            
                            overall_df.dropna(subset=['round', 'window'], inplace=True)
                            overall_df['round'] = overall_df['round'].astype(str)
                            overall_df['window'] = pd.to_numeric(overall_df['window']).astype(int)
                            
                            window_filtered_df = overall_df[
                                (overall_df['round'] == str(round_str)) &
                                (overall_df['window'] == int(window_num))
                            ]
                            
                            if not window_filtered_df.empty:
                                for _, row in window_filtered_df.iterrows():
                                    # Extract course and section info
                                    course_code = self._get_column_value(row, ['Course Code', 'course_code', 'Course_Code'])
                                    section = self._get_column_value(row, ['Section', 'section'])
                                    
                                    if pd.isna(course_code) or pd.isna(section):
                                        continue
                                    
                                    # Find class_boss_id from raw data
                                    class_boss_id = self._find_class_boss_id_from_course_section(course_code, section, acad_term_id)
                                    if not class_boss_id:
                                        continue
                                    
                                    class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)
                                    if not class_ids:
                                        continue
                                    
                                    # Extract bid data
                                    median_bid = None
                                    min_bid = None
                                    for col in row.index:
                                        col_lower = str(col).lower()
                                        if 'median' in col_lower and 'bid' in col_lower:
                                            median_bid = row[col]
                                        elif 'min' in col_lower and 'bid' in col_lower:
                                            min_bid = row[col]
                                    
                                    def safe_int(val): return int(val) if pd.notna(val) else None
                                    def safe_float(val): return float(val) if pd.notna(val) else None
                                    
                                    for class_id in class_ids:
                                        result_data = {
                                            'bid_window_id': bid_window_id,
                                            'class_id': class_id,
                                            'vacancy': safe_int(self._get_column_value(row, ['Vacancy', 'vacancy'])),
                                            'opening_vacancy': safe_int(self._get_column_value(row, ['Opening Vacancy', 'opening_vacancy', 'Opening_Vacancy'])),
                                            'before_process_vacancy': safe_int(self._get_column_value(row, ['Before Process Vacancy', 'before_process_vacancy', 'Before_Process_Vacancy'])),
                                            'dice': safe_int(self._get_column_value(row, ['D.I.C.E', 'dice', 'd_i_c_e', 'DICE'])),
                                            'after_process_vacancy': safe_int(self._get_column_value(row, ['After Process Vacancy', 'after_process_vacancy', 'After_Process_Vacancy'])),
                                            'enrolled_students': safe_int(self._get_column_value(row, ['Enrolled Students', 'enrolled_students', 'Enrolled_Students'])),
                                            'median': safe_float(median_bid),
                                            'min': safe_float(min_bid)
                                        }
                                        
                                        # Check if record exists (UPDATE vs CREATE)
                                        bid_result_key = (bid_window_id, class_id)
                                        if bid_result_key in existing_bid_result_keys:
                                            self.update_bid_result.append(result_data)
                                        else:
                                            self.new_bid_result.append(result_data)
                                            existing_bid_result_keys.add(bid_result_key)
                                            self.boss_stats['bid_results_created'] += 1
                            else:
                                self.log_boss_activity(f"⚠️ No data found in overallBossResults for {window_name}")
                        else:
                            self.log_boss_activity(f"⚠️ No bidding window column found in overallBossResults.xlsx")

            self.log_boss_activity("✅ Sequential catch-up processing completed")

            # --- Step 5: Save all generated files ---
            self.save_outputs()
            self.save_boss_outputs()
            
            # --- Step 6: Final Summary ---
            self.print_boss_summary()
            
            self.log_boss_activity("📝 ✅ Enhanced Phase 3 completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced Phase 3 failed catastrophically: {e}")
            traceback.print_exc()
            return False

    def load_faculties_cache(self):
        """Load faculties from database cache for mapping"""
        try:
            cache_file = os.path.join(self.cache_dir, 'faculties_cache.pkl')
            
            # Try loading from cache file first
            if os.path.exists(cache_file):
                try:
                    faculties_df = pd.read_pickle(cache_file)
                    if not faculties_df.empty:
                        self.faculties_cache = {}
                        self.faculty_acronym_to_id = {}
                        
                        for _, row in faculties_df.iterrows():
                            faculty_id = row['id']
                            acronym = row['acronym'].upper()
                            
                            self.faculties_cache[faculty_id] = row.to_dict()
                            self.faculty_acronym_to_id[acronym] = faculty_id
                        
                        logger.info(f"📚 Loaded {len(self.faculties_cache)} faculties from cache")
                        return True
                    else:
                        logger.warning("⚠️ Faculty cache file exists but is empty")
                except Exception as e:
                    logger.warning(f"⚠️ Error reading faculty cache file: {e}")
            
            # If cache doesn't exist or failed, try database
            if self.connection:
                try:
                    query = "SELECT * FROM faculties"
                    faculties_df = pd.read_sql_query(query, self.connection)
                    if not faculties_df.empty:
                        # Save to cache for future use
                        faculties_df.to_pickle(cache_file)
                        
                        # Load into memory
                        self.faculties_cache = {}
                        self.faculty_acronym_to_id = {}
                        
                        for _, row in faculties_df.iterrows():
                            faculty_id = row['id']
                            acronym = row['acronym'].upper()
                            
                            self.faculties_cache[faculty_id] = row.to_dict()
                            self.faculty_acronym_to_id[acronym] = faculty_id
                        
                        logger.info(f"📚 Downloaded and cached {len(self.faculties_cache)} faculties from database")
                        return True
                    else:
                        logger.warning("⚠️ Database faculties table is empty")
                except Exception as e:
                    logger.warning(f"⚠️ Error downloading faculties from database: {e}")
            
            # Fallback: create basic mapping from known data
            logger.warning("⚠️ Using fallback faculty mapping")
            self.faculties_cache = {}
            self.faculty_acronym_to_id = {
                'LKCSB': 1,   # Lee Kong Chian School of Business
                'YPHSL': 2,   # Yong Pung How School of Law
                'SOE': 3,     # School of Economics
                'SCIS': 4,    # School of Computing and Information Systems
                'SOSS': 5,    # School of Social Sciences
                'SOA': 6,     # School of Accountancy
                'CIS': 7,     # College of Integrative Studies
                'CEC': 8,      # Center for English Communication
                'C4SR': 9,      # Centre for Social Responsibility
                'OCS': 10,      # Dato’ Kho Hui Meng Career Centre
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error in load_faculties_cache: {e}")
            return False

    def map_courses_to_faculties_from_boss(self):
        """Map courses to faculties using course code prefix patterns from existing courses"""
        logger.info("🎓 Starting automated faculty mapping from course code patterns...")
        
        # Load faculties cache first
        if not self.load_faculties_cache():
            logger.error("❌ Failed to load faculties cache")
            return False
        
        # Build prefix-to-faculty mapping from existing courses
        prefix_faculty_mapping = defaultdict(set)  # prefix -> set of faculty_ids
        
        logger.info("📋 Analyzing existing course patterns...")
        
        # Analyze existing courses in cache to build prefix patterns
        for course_code, course_data in self.courses_cache.items():
            faculty_id = course_data.get('belong_to_faculty')
            if faculty_id:
                # Extract prefix (characters before numbers)
                # Handle patterns like "COR-COMM567A" -> "COR-COMM"
                import re
                prefix_match = re.match(r'^([A-Z-]+)', course_code.upper())
                if prefix_match:
                    prefix = prefix_match.group(1)
                    prefix_faculty_mapping[prefix].add(faculty_id)
        
        logger.info(f"📊 Found {len(prefix_faculty_mapping)} unique course prefixes in existing courses")
        
        # Log the patterns found
        for prefix, faculty_ids in prefix_faculty_mapping.items():
            faculty_names = []
            for fid in faculty_ids:
                if fid in self.faculties_cache:
                    faculty_names.append(self.faculties_cache[fid]['acronym'])
            logger.info(f"   {prefix}: {len(faculty_ids)} faculties ({', '.join(faculty_names)})")
        
        # Apply automatic mapping to new courses
        mapped_count = 0
        course_faculty_mappings = {}
        
        for course_info in self.courses_needing_faculty[:]:  # Copy list to modify during iteration
            course_code = course_info['course_code']
            
            # Extract prefix from new course code
            import re
            prefix_match = re.match(r'^([A-Z-]+)', course_code.upper())
            if not prefix_match:
                continue
            
            prefix = prefix_match.group(1)
            
            # Check if this prefix has exactly 1 unique faculty in existing courses
            if prefix in prefix_faculty_mapping:
                faculty_ids = prefix_faculty_mapping[prefix]
                
                if len(faculty_ids) == 1:
                    # Only 1 faculty found - auto-assign
                    faculty_id = list(faculty_ids)[0]
                    course_faculty_mappings[course_code] = faculty_id
                    mapped_count += 1
                    
                    faculty_name = self.faculties_cache[faculty_id]['acronym'] if faculty_id in self.faculties_cache else str(faculty_id)
                    logger.info(f"✅ Auto-mapped {course_code}: prefix '{prefix}' → {faculty_name}")
                else:
                    # Multiple faculties found - leave for manual assignment
                    faculty_names = [self.faculties_cache[fid]['acronym'] for fid in faculty_ids if fid in self.faculties_cache]
                    logger.info(f"⚠️ {course_code}: prefix '{prefix}' maps to {len(faculty_ids)} faculties ({', '.join(faculty_names)}) - manual review needed")
            else:
                # No existing pattern found
                logger.info(f"🆕 {course_code}: new prefix '{prefix}' - manual review needed")
        
        # Apply mappings to courses
        if course_faculty_mappings:
            self._apply_faculty_mappings_to_courses(course_faculty_mappings)
        
        logger.info(f"✅ Pattern-based faculty mapping completed:")
        logger.info(f"   • {mapped_count} courses auto-mapped based on prefix patterns")
        logger.info(f"   • {len(self.courses_needing_faculty)} courses still need manual review")
        
        return True

    def _apply_faculty_mappings_to_courses(self, course_faculty_mappings):
        """Apply faculty mappings to new courses and update courses needing faculty"""
        logger.info(f"🔄 Applying faculty mappings to {len(course_faculty_mappings)} courses...")
        
        mapped_count = 0
        
        # Update new_courses
        for course in self.new_courses:
            course_code = course['code']
            if course_code in course_faculty_mappings:
                course['belong_to_faculty'] = course_faculty_mappings[course_code]
                mapped_count += 1
        
        # Update courses_cache
        for course_code, faculty_id in course_faculty_mappings.items():
            if course_code in self.courses_cache:
                self.courses_cache[course_code]['belong_to_faculty'] = faculty_id
        
        # Remove mapped courses from courses_needing_faculty
        original_needing_count = len(self.courses_needing_faculty)
        self.courses_needing_faculty = [
            course_info for course_info in self.courses_needing_faculty
            if course_info['course_code'] not in course_faculty_mappings
        ]
        
        removed_count = original_needing_count - len(self.courses_needing_faculty)
        
        logger.info(f"✅ Applied faculty mappings:")
        logger.info(f"   • {mapped_count} courses updated with faculty")
        logger.info(f"   • {removed_count} courses removed from manual review queue")
        logger.info(f"   • {len(self.courses_needing_faculty)} courses still need manual review")

    def extract_acad_term_from_path(self, file_path: str) -> Optional[str]:
        r"""Extract acad_term_id from file path as fallback
        Examples:
        'script_input\classTimingsFull\2021-22_T1' -> 'AY202122T1'
        'script_input\classTimingsFull\2022-23_T3A' -> 'AY202223T3A'
        """
        # Extract the term folder name
        path_parts = file_path.replace('/', '\\').split('\\')
        
        for part in path_parts:
            # Look for pattern like "2021-22_T1"
            match = re.match(r'(\d{4})-(\d{2})_T(\w+)', part)
            if match:
                year_start = match.group(1)
                year_end = match.group(2)
                term = match.group(3)
                return f"AY{year_start}{year_end}T{term}"
        
        return None

    def get_last_filepath_by_course(self, course_code):
        """Direct filepath lookup for course code - bypasses record_key linking"""
        print(f"🔍 DEBUG: Looking for course {course_code} using direct method")
        
        # Check if we have standalone data with filepath column
        if hasattr(self, 'standalone_data') and self.standalone_data is not None:
            if 'filepath' in self.standalone_data.columns:
                print(f"✅ DEBUG: Found filepath column in standalone_data")
                
                course_records = self.standalone_data[
                    self.standalone_data['course_code'].str.upper() == course_code.upper()
                ].copy()
                
                print(f"📊 DEBUG: Found {len(course_records)} records for {course_code}")
                
                if not course_records.empty:
                    # Get the most recent record (last row)
                    last_record = course_records.iloc[-1]
                    filepath = last_record.get('filepath')
                    
                    print(f"📁 DEBUG: Last record filepath: {filepath}")
                    
                    if pd.notna(filepath):
                        print(f"✅ Found filepath for {course_code}: {filepath}")
                        return filepath
                    else:
                        print(f"❌ DEBUG: Filepath is NaN for {course_code}")
            else:
                print(f"❌ DEBUG: No 'filepath' column in standalone_data")
                print(f"Available columns: {list(self.standalone_data.columns)}")
        
        # Fallback: check multiple_data if standalone doesn't have filepath
        if hasattr(self, 'multiple_data') and self.multiple_data is not None:
            if 'filepath' in self.multiple_data.columns and 'course_code' in self.multiple_data.columns:
                print(f"✅ DEBUG: Checking multiple_data as fallback")
                
                course_records = self.multiple_data[
                    self.multiple_data['course_code'].str.upper() == course_code.upper()
                ].copy()
                
                if not course_records.empty:
                    last_record = course_records.iloc[-1]
                    filepath = last_record.get('filepath')
                    
                    if pd.notna(filepath):
                        print(f"✅ Found filepath in multiple_data for {course_code}: {filepath}")
                        return filepath
        
        print(f"❌ DEBUG: No filepath found for {course_code}")
        return None

    def close_connection(self):
        """Explicitly close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("🔒 Database connection closed")

    def check_cache_freshness(self) -> bool:
        """
        Implements granular, window-aware cache freshness logic.
        The cache is STALE if its modification time is before the results date 
        of the previous bidding window.
        """
        logger.info("🔍 Checking cache freshness with window-aware logic...")
        now = datetime.now()

        # a. Identify the current active bidding window from the schedule.
        # The current round is the first one whose results_date is in the future.
        current_window_index = -1
        if not self.bidding_schedule:
            logger.info("✅ No bidding schedule found. Assuming cache is fresh.")
            return True

        for i, (results_date, _, _) in enumerate(self.bidding_schedule):
            if now < results_date:
                current_window_index = i
                break
                
        # b. If no future window is found, or we are before/in the first window,
        # there is no "previous window" to check against. The cache is fresh.
        if current_window_index <= 0:
            logger.info("✅ Not in an active bidding period or before the second round. Cache is considered fresh.")
            return True

        # c. Get the results_date of the previous bidding window.
        previous_window_info = self.bidding_schedule[current_window_index - 1]
        previous_round_results_date = previous_window_info[0]
        
        logger.info(f"ℹ️ Rule: Cache must be newer than the previous window's results date: {previous_round_results_date.strftime('%Y-%m-%d %H:%M')}")

        # d. Check if the cache is fresh by comparing its modification time.
        cache_files_to_check = [
            os.path.join(self.cache_dir, 'professors_cache.pkl'),
            os.path.join(self.cache_dir, 'courses_cache.pkl'),
            os.path.join(self.cache_dir, 'classes_cache.pkl'),
        ]
        oldest_cache_time = None

        for path in cache_files_to_check:
            if not os.path.exists(path):
                logger.warning(f"⚠️ Critical cache file not found: {path}. A full download is required.")
                return False
                
            mod_time_dt = datetime.fromtimestamp(os.path.getmtime(path))
            if oldest_cache_time is None or mod_time_dt < oldest_cache_time:
                oldest_cache_time = mod_time_dt

        if oldest_cache_time is None:
            # This case should be prevented by the check above, but is included for safety.
            logger.warning("⚠️ No cache files exist. A full download is required.")
            return False

        logger.info(f"ℹ️ Oldest relevant cache file was last modified on: {oldest_cache_time.strftime('%Y-%m-%d %H:%M')}")

        # The cache is fresh if its oldest part was created on or after the previous window's results.
        if oldest_cache_time >= previous_round_results_date:
            logger.info("✅ Cache is FRESH.")
            return True
        else:
            logger.warning("❌ Cache is STALE. A full download is required.")
            return False

    def _get_bidding_round_info_for_term(self, ay_term, now):
        """Get current bidding round info for a term"""
        if START_AY_TERM in ay_term:
            for results_date, _, folder_suffix in self.bidding_schedule:
                if now < results_date:
                    return f"{ay_term}_{folder_suffix}"
        return None

    def load_or_cache_data_with_freshness_check(self):
        """
        Load data with a freshness check, combine it with new CSV files,
        and handle caching using robust, type-safe logic.
        """
        # Part 1: Check cache freshness and download new data if necessary.
        if not self.check_cache_freshness():
            logger.info("🔄 Cache is stale, downloading fresh data from the database...")
            if not self.connect_database():
                return False
            
            try:
                self._download_and_cache_data()
                logger.info("✅ Successfully downloaded fresh data from the database.")
            except Exception as e:
                logger.error(f"❌ Failed to download fresh data: {e}")
                return False
        
        # Part 2: Load the core data from the cache (either freshly downloaded or existing).
        if not self.load_or_cache_data():
            return False
        
        # Part 3: Combine the loaded cache with new data from local CSV files.
        # This logic is integrated from the improved _combine_with_new_files method.
        logger.info("🔄 Combining database cache with new CSV files...")
        
        # Combine new_classes.csv with existing_classes_cache
        new_classes_path = os.path.join(self.output_base, 'new_classes.csv')
        if os.path.exists(new_classes_path):
            try:
                new_classes_df = pd.read_csv(new_classes_path)
                if not new_classes_df.empty:
                    new_classes_list = new_classes_df.to_dict('records')
                    
                    # Safely initialize the cache if it doesn't exist to prevent errors.
                    if not hasattr(self, 'existing_classes_cache'):
                        self.existing_classes_cache = []
                    
                    # Add new classes, checking for duplicates with precise, multi-field logic.
                    added_count = 0
                    for new_class in new_classes_list:
                        exists = False
                        for existing_class in self.existing_classes_cache:
                            # Precise check including course, section, term, and professor.
                            if (existing_class['course_id'] == new_class['course_id'] and
                                str(existing_class['section']) == str(new_class['section']) and
                                existing_class['acad_term_id'] == new_class['acad_term_id'] and
                                existing_class.get('professor_id') == new_class.get('professor_id')):
                                exists = True
                                break
                        
                        if not exists:
                            self.existing_classes_cache.append(new_class)
                            added_count += 1
                    
                    if added_count > 0:
                        logger.info(f"✅ Added {added_count} new, unique classes to the existing cache.")
            except Exception as e:
                logger.warning(f"⚠️ Could not combine new_classes.csv: {e}")
        
        # Update bid_window_cache with new_bid_window.csv
        new_bid_window_path = os.path.join(self.output_base, 'new_bid_window.csv')
        if os.path.exists(new_bid_window_path):
            try:
                new_bid_window_df = pd.read_csv(new_bid_window_path)
                if not new_bid_window_df.empty:
                    added_count = 0
                    for _, row in new_bid_window_df.iterrows():
                        # Use explicit type casting for robust key creation.
                        window_key = (row['acad_term_id'], str(row['round']), int(row['window']))
                        if window_key not in self.bid_window_cache:
                            self.bid_window_cache[window_key] = row['id']
                            added_count += 1
                    if added_count > 0:
                        logger.info(f"✅ Added {added_count} new bid windows to the cache.")
            except Exception as e:
                logger.warning(f"⚠️ Could not combine new_bid_window.csv: {e}")
        
        # Update courses_cache with new_courses.csv from multiple locations
        new_courses_paths = [
            os.path.join(self.output_base, 'new_courses.csv'),
            os.path.join(self.verify_dir, 'new_courses.csv')
        ]
        
        for path in new_courses_paths:
            if os.path.exists(path):
                try:
                    new_courses_df = pd.read_csv(path)
                    if not new_courses_df.empty:
                        added_count = 0
                        for _, row in new_courses_df.iterrows():
                            if row['code'] not in self.courses_cache:
                                self.courses_cache[row['code']] = row.to_dict()
                                added_count += 1
                        if added_count > 0:
                            logger.info(f"✅ Added {added_count} new courses from {path}.")
                except Exception as e:
                    logger.warning(f"⚠️ Could not combine {path}: {e}")
        
        return True

    def check_record_exists_in_cache(self, table_name, record_data, key_fields):
        """Check if record exists in database cache and if it needs updates"""
        try:
            cache_file = os.path.join(self.cache_dir, f'{table_name}_cache.pkl')
            if not os.path.exists(cache_file):
                return False, None
            
            df = pd.read_pickle(cache_file)
            if df.empty:
                return False, None
            
            # Build query mask
            mask = True
            for field in key_fields:
                if field in df.columns and field in record_data:
                    mask = mask & (df[field] == record_data[field])
            
            matching_records = df[mask]
            if matching_records.empty:
                return False, None
            
            # Return first match for update comparison
            return True, matching_records.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error checking cache for {table_name}: {e}")
            return False, None

    def needs_update(self, existing_record, new_record_or_row, field_mapping_or_fields):
        """
        Check if existing record needs updates based on field mapping or field list.
        Handles both dictionary-to-dictionary and row-to-record comparisons.
        """
        # Handle different parameter types
        if isinstance(field_mapping_or_fields, dict):
            # field_mapping case: {db_field: raw_field}
            field_mapping = field_mapping_or_fields
            compare_fields = []
            for db_field, raw_field in field_mapping.items():
                old_value = existing_record.get(db_field)
                new_value = new_record_or_row.get(raw_field)
                
                # Type-specific comparison
                if db_field == 'credit_units':
                    new_value = float(new_value) if pd.notna(new_value) else None
                    old_value = float(old_value) if pd.notna(old_value) else None
                else:
                    if pd.isna(new_value):
                        new_value = None
                    else:
                        new_value = str(new_value).strip()
                    
                    if pd.isna(old_value):
                        old_value = None
                    else:
                        old_value = str(old_value).strip() if old_value is not None else None
                
                # Check for actual change
                if new_value != old_value:
                    # Don't overwrite existing data with empty data
                    if new_value is None or new_value == '':
                        if old_value is not None and old_value != '':
                            continue
                    return True
            return False
        else:
            # field list case: direct field comparison
            compare_fields = field_mapping_or_fields
            for field in compare_fields:
                existing_value = existing_record.get(field)
                new_value = new_record_or_row.get(field)
                
                # Handle different data types
                if pd.isna(existing_value) and pd.isna(new_value):
                    continue
                if pd.isna(existing_value) or pd.isna(new_value):
                    return True
                
                # FIXED: Proper string comparison for course_outline_url and other fields
                if str(existing_value).strip() != str(new_value).strip():
                    return True
            
            return False

    def load_overall_boss_results(self):
        """Load overall BOSS results from script_input/overallBossResults/"""
        logger.info("📊 Loading overall BOSS results...")
        
        now = datetime.now()
        current_round_info = self._get_bidding_round_info_for_term(START_AY_TERM, now)
        
        if not current_round_info:
            logger.info("⏭️ Not in active bidding period - skipping overall results")
            return None
        
        # Determine which overall results file to load based on current round
        overall_results_dir = 'script_input/overallBossResults'
        if not os.path.exists(overall_results_dir):
            logger.warning(f"⚠️ Overall results directory not found: {overall_results_dir}")
            return None
        
        # Look for the appropriate Excel file
        results_file = os.path.join(overall_results_dir, f"{START_AY_TERM}.xlsx")
        if not os.path.exists(results_file):
            logger.warning(f"⚠️ Overall results file not found: {results_file}")
            return None
        
        try:
            # Load the Excel file
            df = pd.read_excel(results_file)
            logger.info(f"✅ Loaded {len(df)} overall BOSS results from {results_file}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading overall results: {e}")
            return None

    def determine_previous_round_for_overall_results(self, current_round_info):
        """Determine which round's results to create based on current round"""
        if not current_round_info:
            return None, None
        
        # Extract current round suffix
        current_suffix = current_round_info.split('_')[-1]
        
        # Mapping of current round to previous round results
        round_mapping = {
            'R1AW1': ('1', 1),     # When in R1A W1, create R1 W1 results
            'R1AW2': ('1A', 1),    # When in R1A W2, create R1A W1 results
            'R1AW3': ('1A', 2),    # When in R1A W3, create R1A W2 results
            'R1BW1': ('1A', 3),    # When in R1B W1, create R1A W3 results
            'R1BW2': ('1B', 1),    # When in R1B W2, create R1B W1 results
            'R1CW1': ('1B', 2),    # When in R1C W1, create R1B W2 results
            'R1CW2': ('1C', 1),    # When in R1C W2, create R1C W1 results
            'R1CW3': ('1C', 2),    # When in R1C W3, create R1C W2 results
            'R1FW1': ('1C', 3),    # When in R1F W1, create R1C W3 results
            'R1FW2': ('1F', 1),    # When in R1F W2, create R1F W1 results
            'R1FW3': ('1F', 2),    # When in R1F W3, create R1F W2 results
            'R1FW4': ('1F', 3),    # When in R1F W4, create R1F W3 results
            'R2W1': ('1F', 4),     # When in R2 W1, create R1F W4 results
            'R2W2': ('2', 1),      # When in R2 W2, create R2 W1 results
            'R2W3': ('2', 2),      # When in R2 W3, create R2 W2 results
            'R2AW1': ('2', 3),     # When in R2A W1, create R2 W3 results
            'R2AW2': ('2A', 1),    # When in R2A W2, create R2A W1 results
            'R2AW3': ('2A', 2),    # When in R2A W3, create R2A W2 results
        }
        
        return round_mapping.get(current_suffix, (None, None))

    def process_overall_boss_results(self):
        """
        Process overall BOSS results from the dedicated Excel file.
        - It parses the 'Bidding Window' column to extract round and window numbers.
        - For existing bid_result records, it creates an UPDATE record with the latest median/min values.
        - If a record doesn't exist, it creates a NEW record as a fallback.
        - It correctly logs any failed class lookups.
        """
        logger.info("📈 Processing overall BOSS results (for updates)...")
        
        overall_df = self.load_overall_boss_results()
        if overall_df is None or overall_df.empty:
            logger.warning("⚠️ No overall BOSS results file found or file is empty. Skipping.")
            return True

        # --- FIX for 'Bidding Window' column ---
        # Standardize column names for robustness BUT preserve the actual column names for data access
        standardized_columns = {}
        for col in overall_df.columns:
            standardized = str(col).lower().replace(' ', '_').replace('.', '')
            standardized_columns[standardized] = col
        
        # Check for the bidding window column
        bidding_window_col = None
        for std_name, orig_name in standardized_columns.items():
            if 'bidding_window' in std_name or 'bidding window' in orig_name.lower():
                bidding_window_col = orig_name
                break
        
        if not bidding_window_col:
            logger.error("❌ 'overallBossResults' file is missing the 'Bidding Window' column. Cannot process.")
            return False

        # Apply the parsing function to create 'round' and 'window' columns from the 'bidding_window' string
        try:
            parsed_windows = overall_df[bidding_window_col].apply(self.parse_bidding_window)
            overall_df['round'] = parsed_windows.apply(lambda x: x[0] if isinstance(x, tuple) else None)
            overall_df['window'] = parsed_windows.apply(lambda x: x[1] if isinstance(x, tuple) else None)
        except Exception as e:
            logger.error(f"❌ Failed to parse 'bidding_window' column: {e}")
            return False

        # Drop rows where parsing might have failed (e.g., unexpected format)
        overall_df.dropna(subset=['round', 'window'], inplace=True)
        if overall_df.empty:
            logger.warning("⚠️ Could not parse any valid round/window from the 'Bidding Window' column. Skipping.")
            return True
        
        # ==================================================================
        # === NEW LOGIC TO PROCESS ONLY THE PREVIOUS ROUND/WINDOW        ===
        # ==================================================================
        now = datetime.now()
        # Use the global START_AY_TERM to find the current active round based on the system time and schedule
        current_round_info = self._get_bidding_round_info_for_term(START_AY_TERM, now)

        if not current_round_info:
            logger.info("ℹ️ Not in an active bidding period. Skipping overall results processing.")
            return True

        # Determine the target round/window that we SHOULD be processing from the Excel file
        target_round, target_window = self.determine_previous_round_for_overall_results(current_round_info)

        if not target_round or not target_window:
            logger.info(f"ℹ️ Current active window ('{current_round_info}') does not require processing of previous results. Skipping.")
            return True
            
        logger.info(f"🎯 Current active window is '{current_round_info}'. Targeting results for Round [{target_round}] Window [{target_window}].")
        
        # Filter the DataFrame to only include rows for the target round and window
        overall_df['round'] = overall_df['round'].astype(str)
        overall_df['window'] = pd.to_numeric(overall_df['window']).astype(int)
        
        original_rows = len(overall_df)
        overall_df = overall_df[
            (overall_df['round'] == str(target_round)) &
            (overall_df['window'] == int(target_window))
        ]
        
        if overall_df.empty:
            logger.warning(f"⚠️ No data found in 'overallBossResults.xlsx' for the target Round [{target_round}] Window [{target_window}]. (Checked {original_rows} rows). Skipping.")
            return True
        
        logger.info(f"✅ Filtered 'overallBossResults' to {len(overall_df)} rows for the target round and window.")

        if not hasattr(self, 'update_bid_result'):
            self.update_bid_result = []
            
        overall_df['round'] = overall_df['round'].astype(str)
        overall_df['window'] = pd.to_numeric(overall_df['window']).astype(int)
        
        grouped_results = overall_df.groupby(['round', 'window'])
        
        new_records_count = 0
        updated_records_count = 0
        failed_count = 0

        for (current_round, current_window), group in grouped_results:
            logger.info(f"📊 Processing results for Round {current_round} Window {current_window}")
            
            acad_term_id = "AY202526T1"
            window_key = (acad_term_id, str(current_round), int(current_window))
            bid_window_id = self.bid_window_cache.get(window_key)

            if not bid_window_id:
                logger.warning(f"⚠️ Could not find bid_window_id for {window_key}, creating it now.")
                new_bid_window = {
                    'id': self.bid_window_id_counter, 'acad_term_id': acad_term_id,
                    'round': str(current_round), 'window': int(current_window)
                }
                self.new_bid_windows.append(new_bid_window)
                self.bid_window_cache[window_key] = self.bid_window_id_counter
                bid_window_id = self.bid_window_id_counter
                self.boss_stats['bid_windows_created'] += 1
                self.bid_window_id_counter += 1

            # Track rows with bid data for debugging
            rows_with_bid_data = 0

            for idx, row in group.iterrows():
                try:
                    # FIXED: Use Course Code + Section instead of class_boss_id
                    course_code = self._get_column_value(row, ['Course Code', 'course_code', 'Course_Code'])
                    section = self._get_column_value(row, ['Section', 'section'])
                    
                    if pd.isna(course_code) or pd.isna(section):
                        continue

                    # Find class_boss_id from the raw data using course_code + section
                    class_boss_id = self._find_class_boss_id_from_course_section(course_code, section, acad_term_id)
                    
                    if not class_boss_id:
                        failed_count += 1
                        self.failed_mappings.append({
                            'course_code': course_code, 'section': section, 'acad_term_id': acad_term_id,
                            'round': current_round, 'window': current_window, 'reason': 'class_boss_id_not_found',
                            'bidding_window_str': row.get(bidding_window_col, '')
                        })
                        continue

                    class_ids = self.find_all_class_ids(acad_term_id, class_boss_id)
                    
                    if not class_ids:
                        failed_count += 1
                        self.failed_mappings.append({
                            'course_code': course_code, 'section': section, 'acad_term_id': acad_term_id,
                            'round': current_round, 'window': current_window, 'reason': 'class_not_found',
                            'bidding_window_str': row.get(bidding_window_col, '')
                        })
                        continue

                    # Find the correct column names (case-insensitive)
                    median_bid = None
                    min_bid = None
                    
                    # Map standardized names to actual column names
                    for col in row.index:
                        col_lower = str(col).lower()
                        if 'median' in col_lower and 'bid' in col_lower:
                            median_bid = row[col]
                        elif 'min' in col_lower and 'bid' in col_lower:
                            min_bid = row[col]

                    # DEBUG: Log first few rows with bid data
                    if pd.notna(median_bid) or pd.notna(min_bid):
                        rows_with_bid_data += 1
                        if rows_with_bid_data <= 3:
                            logger.info(f"🔍 DEBUG: {course_code}-{section} has median={median_bid}, min={min_bid}")

                    for class_id in class_ids:
                        def safe_int(val): return int(val) if pd.notna(val) else None
                        def safe_float(val): return float(val) if pd.notna(val) else None

                        # Map all the column names properly
                        result_data = {
                            'bid_window_id': bid_window_id, 
                            'class_id': class_id,
                            'vacancy': safe_int(self._get_column_value(row, ['Vacancy', 'vacancy'])),
                            'opening_vacancy': safe_int(self._get_column_value(row, ['Opening Vacancy', 'opening_vacancy', 'Opening_Vacancy'])),
                            'before_process_vacancy': safe_int(self._get_column_value(row, ['Before Process Vacancy', 'before_process_vacancy', 'Before_Process_Vacancy'])),
                            'dice': safe_int(self._get_column_value(row, ['D.I.C.E', 'dice', 'd_i_c_e', 'DICE'])),
                            'after_process_vacancy': safe_int(self._get_column_value(row, ['After Process Vacancy', 'after_process_vacancy', 'After_Process_Vacancy'])),
                            'enrolled_students': safe_int(self._get_column_value(row, ['Enrolled Students', 'enrolled_students', 'Enrolled_Students'])),
                            'median': safe_float(median_bid),
                            'min': safe_float(min_bid)
                        }
                        
                        exists, existing_record = self.check_record_exists_in_cache(
                            'bid_result',
                            {'bid_window_id': bid_window_id, 'class_id': class_id},
                            ['bid_window_id', 'class_id']
                        )

                        if exists:
                            # Always update when processing overall results (they have the final bid data)
                            self.update_bid_result.append(result_data)
                            updated_records_count += 1
                            if (pd.notna(median_bid) or pd.notna(min_bid)) and updated_records_count <= 5:
                                logger.info(f"📊 UPDATE: {course_code}-{section} with median={median_bid}, min={min_bid}")
                        else:
                            self.new_bid_result.append(result_data)
                            self.boss_stats['bid_results_created'] += 1
                            new_records_count += 1
                            
                except Exception as e:
                    logger.error(f"Error processing row for {row.get('Course Code', 'unknown')}-{row.get('Section', 'unknown')}: {e}")
                    continue

            logger.info(f"✅ Round {current_round} Window {current_window}: {rows_with_bid_data} rows had bid data")

        self.boss_stats['failed_mappings'] += failed_count
        logger.info("✅ Overall Results Processing Complete.")
        logger.info(f"  - Records to CREATE: {new_records_count}")
        logger.info(f"  - Records to UPDATE: {updated_records_count}")
        logger.info(f"  - Failed Mappings: {failed_count}")
        
        return True

    def _get_column_value(self, row, possible_names):
        """Helper method to get column value by trying multiple possible column names"""
        for name in possible_names:
            if name in row.index:
                return row[name]
        return None

    def _find_class_boss_id_from_course_section(self, course_code, section, acad_term_id):
        """Find class_boss_id from course_code + section + acad_term_id"""
        if not hasattr(self, 'standalone_data') or self.standalone_data is None:
            return None
        
        # Look up in standalone_data
        matches = self.standalone_data[
            (self.standalone_data['course_code'] == course_code) &
            (self.standalone_data['section'].astype(str) == str(section).strip()) &
            (self.standalone_data['acad_term_id'] == acad_term_id)
        ]
        
        if not matches.empty:
            return matches.iloc[0].get('class_boss_id')
        
        return None
     
    def _parse_boss_aliases(self, boss_aliases_val: any) -> list[str]:
        """
        Robustly parses the boss_aliases value from various formats into a clean list of strings.

        This function correctly handles:
        - None, pd.isna(), or other "empty" values.
        - A standard Python list.
        - A NumPy array.
        - A raw PostgreSQL array string (e.g., '{"item1","item2"}').
        - A JSON-formatted string array (e.g., '["item1", "item2"]').

        Returns:
            A clean Python list of strings. Returns an empty list for any invalid or empty input.
        """
        # Return an empty list for any "empty" or None-like value.
        if boss_aliases_val is None:
            return []
        
        # Handle arrays/lists before using pd.isna
        if hasattr(boss_aliases_val, '__len__') and not isinstance(boss_aliases_val, str):
            # It's already an array/list, so check if it's empty
            if len(boss_aliases_val) == 0:
                return []
            # If it's a non-empty array, process it
            if isinstance(boss_aliases_val, list):
                return [str(item).strip() for item in boss_aliases_val if item and str(item).strip()]
            elif hasattr(boss_aliases_val, 'tolist'):
                # NumPy array
                return [str(item).strip() for item in boss_aliases_val.tolist() if item and str(item).strip()]
            else:
                # Other iterable
                return [str(item).strip() for item in boss_aliases_val if item and str(item).strip()]
        
        # Now safe to use pd.isna for non-array values
        try:
            if pd.isna(boss_aliases_val):
                return []
        except:
            # If pd.isna fails for any reason, continue processing
            pass

        # Handle standard Python list.
        if isinstance(boss_aliases_val, list):
            return [str(item).strip() for item in boss_aliases_val if item and str(item).strip()]

        # Handle NumPy array by checking for the .tolist() method.
        if hasattr(boss_aliases_val, 'tolist'):
            return [str(item).strip() for item in boss_aliases_val.tolist() if item and str(item).strip()]

        # Handle various string formats.
        if isinstance(boss_aliases_val, str):
            aliases_str = boss_aliases_val.strip()
            
            if not aliases_str:
                return []
                
            # Case 1: PostgreSQL array format '{"item1","item2"}'
            if aliases_str.startswith('{') and aliases_str.endswith('}'):
                content = aliases_str[1:-1]
                # Split by comma, then strip whitespace and quotes from each item.
                return [item.strip().strip('"') for item in content.split(',') if item.strip()]

            # Case 2: JSON array format '["item1", "item2"]'
            if aliases_str.startswith('[') and aliases_str.endswith(']'):
                try:
                    parsed_list = json.loads(aliases_str)
                    if isinstance(parsed_list, list):
                        return [str(item).strip() for item in parsed_list if item and str(item).strip()]
                except (json.JSONDecodeError, TypeError):
                    # If JSON is malformed, fall back to treating it as a plain string.
                    pass

            # Case 3: A single alias provided as a plain string.
            return [aliases_str]

        # Fallback for other iterable types like tuples or sets.
        if hasattr(boss_aliases_val, '__iter__'):
            return [str(item).strip() for item in boss_aliases_val if item and str(item).strip()]
            
        return []

    def _extract_unique_professors(self) -> Tuple[set, dict]:
        """Extracts unique professor names and their variations from the raw data."""
        unique_professors = set()
        professor_variations = defaultdict(set)

        for _, row in self.multiple_data.iterrows():
            prof_name_raw = row.get('professor_name')
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue
            
            prof_name = str(prof_name_raw).strip()
            if not prof_name or prof_name.lower() in ['nan', 'tba', 'to be announced']:
                continue
            
            split_professors = self._split_professor_names(prof_name)
            for individual in split_professors:
                clean_prof = individual.strip()
                if clean_prof:
                    unique_professors.add(clean_prof)
                    if ', ' in clean_prof:
                        parts = clean_prof.split(', ')
                        if len(parts) == 2:
                            base_name = parts[0].strip()
                            extension = parts[1].strip()
                            if len(extension.split()) == 1:
                                professor_variations[clean_prof].add(base_name)
                                professor_variations[clean_prof].add(clean_prof)
                                if base_name in professor_variations:
                                    professor_variations[base_name].add(clean_prof)
                    else:
                        professor_variations[clean_prof].add(clean_prof)
        
        return unique_professors, professor_variations

    def _normalize_professors_batch(self, names_to_process: list) -> dict:
        """
        Normalizes a list of professor names using the pre-configured LLM client,
        with a rule-based fallback.
        """
        normalized_map = {}
        if not names_to_process:
            return normalized_map

        # --- LLM Pathway ---
        try:
            # Check if the client was successfully initialized in __init__
            if not self.llm_client:
                raise ValueError("LLM client not configured. Check API key.")

            total_batches = (len(names_to_process) + self.llm_batch_size - 1) // self.llm_batch_size
            logger.info(f"🧪 Normalizing {len(names_to_process)} names in {total_batches} batches using '{self.llm_model_name}'...")

            for i in range(0, len(names_to_process), self.llm_batch_size):
                batch_names = names_to_process[i:i + self.llm_batch_size]
                logger.info(f"  -> Processing batch {i//self.llm_batch_size + 1} of {total_batches} ({len(batch_names)} names)...")
                
                # Use the new client API
                response = self.llm_client.models.generate_content(
                    model=self.llm_model_name,
                    contents=f"{self.llm_prompt}\n\n{json.dumps(batch_names)}",
                    config=genai.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                # Robustly find the JSON block within the response text
                response_text = response.text
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if not match:
                    raise ValueError("LLM response did not contain a valid JSON array.")
                
                json_text = match.group(0)
                surnames = json.loads(json_text)
                
                if not isinstance(surnames, list) or len(surnames) != len(batch_names):
                    raise ValueError(f"LLM returned malformed data for batch {i//self.llm_batch_size + 1}.")

                for original_name, surname in zip(batch_names, surnames):
                    name_str = str(original_name).strip().replace("'", "'")
                    name_str = re.sub(r'\s*\(.*\)\s*', ' ', name_str).strip()
                    words = name_str.split()
                    words_no_initials = [word for word in words if not (len(word) == 1 and word.isalpha()) and not (len(word) == 2 and word.endswith('.'))]
                    boss_name = ' '.join(words_no_initials).upper()

                    name_parts = re.split(r'([ ,])', original_name)
                    afterclass_parts = []
                    surname_found = False
                    for part in name_parts:
                        if not surname_found and part.strip(" ,").upper() == surname.upper():
                            afterclass_parts.append(part.upper())
                            surname_found = True
                        else:
                            afterclass_parts.append(part.capitalize())
                    afterclass_name = "".join(afterclass_parts)
                    normalized_map[original_name] = (boss_name, afterclass_name)
                
                time.sleep(6)
            
            logger.info("✅ Batch normalization completed using Gemini LLM.")

        # --- Fallback Pathway ---
        except Exception as e:
            logger.warning(f"⚠️ LLM normalization failed ({e}). Falling back to rule-based method.")
            normalized_map.clear() # Ensure map is empty before filling
            for name in names_to_process:
                normalized_map[name] = self._normalize_professor_name_fallback(name)
        
        return normalized_map
    
if __name__ == "__main__":
    builder = TableBuilder()
    
    # Establish database connection with pooling (no transaction started here)
    if not builder.connect_database():
        sys.exit(1)

    try:
        # --- Phase 1 ---
        print("--- Running Phase 1: Professors and Courses ---")
        if not builder.run_phase1_professors_and_courses():
            raise Exception("Phase 1 failed.")
        print("✅ Phase 1 completed successfully.")

        # --- Phase 2 ---
        print("\n--- Running Phase 2: Classes and Timings ---")
        if not builder.run_phase2_remaining_tables():
            raise Exception("Phase 2 failed.")
        print("✅ Phase 2 completed successfully.")
        
        # --- Faculty Assignment (Interactive Step) ---
        print("\n--- Running Faculty Assignment ---")
        builder.assign_course_faculties()
        print("✅ Faculty assignment completed.")
        
        # --- Phase 3 ---
        print("\n--- Running Phase 3: Bidding Data ---")
        if not builder.run_phase3_boss_processing():
            raise Exception("Phase 3 failed.")
        print("✅ Phase 3 completed successfully.")

        # --- Phase 4: Save CSV Checkpoints (always before DB upload) ---
        print("\n--- Saving CSV Checkpoints ---")
        builder.save_outputs()
        print("✅ CSV checkpoints saved.")
        
        # --- Phase 5: Upload to Database with Transaction ---
        print("\n--- Uploading CSVs to Database ---")
        upload_success = builder.upload_csvs_to_database()
        
        if not upload_success:
            print("❌ Database upload failed. CSVs are preserved - you can retry upload.")
            sys.exit(1)
        
        print("\n🎉 All phases completed successfully!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Close database connection
        if hasattr(builder, 'db_connection') and builder.db_connection:
            builder.db_connection.close()
            print("🔒 Database connection closed.")
        
        # Close database manager if it exists
        if hasattr(builder, 'db_manager') and builder.db_manager:
            builder.db_manager.close_all_connections()
            print("🔒 Database connections pooled and closed.")