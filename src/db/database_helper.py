import os
import pandas as pd
from datetime import datetime
from src.utils.cache_resolver import get_table_builder_cache_files, merge_bid_window_csv_into_cache
from src.utils.alias_parser import parse_boss_aliases
import traceback

class DatabaseHelper:
    @staticmethod
    def _get_cache_dir(obj):
        """Get cache_dir from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'config'):
            return obj.config.cache_dir
        return obj.cache_dir

    @staticmethod
    def _get_logger(obj):
        """Get logger from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'config'):
            return obj.logger
        return obj._logger

    @staticmethod
    def _get_db_connection(obj):
        """Get db_connection from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'config'):
            return obj.db_connection
        return obj.db_connection

    @staticmethod
    def _get_bidding_schedule(obj):
        """Get bidding_schedule from TableBuilder or ProcessorContext."""
        # Return obj.bidding_schedule if it exists (TableBuilder sets this to the filtered list)
        if hasattr(obj, 'bidding_schedule'):
            return obj.bidding_schedule
        # Fallback to config's bidding_schedules dict filtered by start_ay_term
        if hasattr(obj, 'config'):
            bidding_schedules = obj.config.bidding_schedules or {}
            start_term = getattr(obj.config, 'start_ay_term', None)
            if start_term:
                return bidding_schedules.get(start_term, [])
            return list(bidding_schedules.values())[0] if bidding_schedules else []
        return obj.bidding_schedule

    @staticmethod
    def _get_output_base(obj):
        """Get output_base from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'config'):
            return obj.config.output_base
        return obj.output_base

    @staticmethod
    def _get_verify_dir(obj):
        """Get verify_dir from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'config'):
            return obj.config.verify_dir
        return obj.verify_dir

    @staticmethod
    def _get_existing_classes_cache(obj):
        """Get existing_classes_cache from TableBuilder or ProcessorContext."""
        if hasattr(obj, 'context'):
            return obj.context.existing_classes_cache
        return obj.existing_classes_cache

    @staticmethod
    def create_connection(db_adapter, logger):
        """Connect to PostgreSQL database using psycopg2."""
        try:
            connection = db_adapter.connect()
            return connection
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def load_or_cache_data(obj):
        """Load data from cache or database"""
        logger = DatabaseHelper._get_logger(obj)

        # Try loading from cache first
        if DatabaseHelper._load_from_cache(obj):
            logger.info("✅ Loaded data from cache")
            return True

        # Connect to database and download
        db_connection = DatabaseHelper._get_db_connection(obj)
        if not db_connection:
            return False

        try:
            DatabaseHelper._download_and_cache_data(obj)
            logger.info("✅ Downloaded and cached data from database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download data: {e}")
            return False

    @staticmethod
    def _download_and_cache_data(obj):
        """Download data from database tables and cache them locally."""
        logger = DatabaseHelper._get_logger(obj)
        db_connection = DatabaseHelper._get_db_connection(obj)
        cache_dir = DatabaseHelper._get_cache_dir(obj)

        try:
            tables_to_cache = [
                "professors", "courses", "acad_term", "faculties",
                "classes", "class_timing", "class_exam_timing",
                "class_availability", "bid_window", "bid_result", "bid_prediction"
            ]

            for table_name in tables_to_cache:
                logger.info(f"⬇️ Caching table: {table_name}")
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, db_connection)
                df.to_pickle(os.path.join(cache_dir, f'{table_name}_cache.pkl'))

            logger.info("✅ Downloaded all tables from database and cached locally")
            DatabaseHelper._load_from_cache(obj)

        except Exception as e:
            logger.error(f"❌ Failed to download and cache data for table '{table_name}': {e}")
            raise

    @staticmethod
    def _load_from_cache(obj) -> bool:
        """
        Load cached data from files, with robust validation of the professor lookup against the database cache.
        Professor validation only runs during Phase 1 (TableBuilder).

        Context-agnostic: works with TableBuilder (Phase 1/2/3) or ProcessorContext (Phase 3 BOSS).
        """
        try:
            logger = DatabaseHelper._get_logger(obj)
            cache_dir = DatabaseHelper._get_cache_dir(obj)

            cache_files = get_table_builder_cache_files(cache_dir)

            if not all(os.path.exists(f) for f in cache_files.values()):
                logger.warning("⚠️ Not all cache files found. Need to download from database.")
                return False

            # Load professors data first
            professors_df = pd.read_pickle(cache_files['professors'])

            # Check if this is Phase 1 by checking for _phase2_mode attribute
            # Phase 1: TableBuilder without _phase2_mode set (obj._phase2_mode = False or undefined)
            # Phase 2/3: TableBuilder with _phase2_mode=True OR ProcessorContext
            is_table_builder = hasattr(obj, '_phase2_mode')
            is_phase1 = is_table_builder and not obj._phase2_mode

            # Caches are always directly on the object (TableBuilder) or ProcessorContext
            professors_cache = obj.professors_cache
            courses_cache = obj.courses_cache
            acad_term_cache = obj.acad_term_cache
            faculties_cache = obj.faculties_cache
            bid_window_cache = obj.bid_window_cache
            faculty_acronym_to_id = obj.faculty_acronym_to_id
            professor_lookup = getattr(obj, 'professor_lookup', {})

            if is_phase1 and is_table_builder:
                # --- Professor Lookup Synchronization (Phase 1 only - TableBuilder) ---
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
                    aliases_list = parse_boss_aliases(row.get('boss_aliases'))
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

                professor_lookup = validated_professor_lookup

                logger.info("✅ Phase 1 Professor lookup synchronization complete:")
                logger.info(f"  - Entries validated: {len(validated_professor_lookup)}")
                logger.info(f"  - Corrected entries: {csv_entries_corrected}")
                logger.info(f"  - Added DB entries: {csv_entries_added}")
                logger.info(f"  - Removed invalid entries: {csv_entries_removed}")

                # Save corrected lookup back to file
                corrected_lookup_data = sorted(list(professor_lookup.values()), key=lambda x: x['boss_name'])
                for item in corrected_lookup_data:
                    item['method'] = 'validated'

                corrected_df = pd.DataFrame(corrected_lookup_data)
                corrected_df.to_csv(lookup_file, index=False, columns=['boss_name', 'afterclass_name', 'database_id', 'method'])
                logger.info(f"💾 Updated '{lookup_file}' with synchronized data.")

                # Build professors_cache for lookups
                professors_cache.clear()
                for lookup_data in professor_lookup.values():
                    db_id = lookup_data['database_id']
                    boss_name_key = lookup_data['boss_name'].upper()
                    if db_id in database_professors:
                        professors_cache[boss_name_key] = database_professors[db_id]
            else:
                # Phase 2/3 or ProcessorContext: Simple loading without validation
                if is_table_builder:
                    logger.info("🔄 Phase 2/3: Loading professor data without validation...")
                else:
                    logger.info("🔄 Loading professor data (ProcessorContext)...")

                professors_cache.clear()
                for _, row in professors_df.iterrows():
                    # Simple loading for non-Phase 1
                    professor_data = row.to_dict()
                    professor_name = str(row.get('name', '')).strip().upper()
                    if professor_name:
                        professors_cache[professor_name] = professor_data

            # --- Load Remaining Caches (all phases) ---
            courses_df = pd.read_pickle(cache_files['courses'])
            for _, row in courses_df.iterrows(): courses_cache[row['code']] = row.to_dict()

            acad_term_df = pd.read_pickle(cache_files['acad_term'])
            for _, row in acad_term_df.iterrows(): acad_term_cache[row['id']] = row.to_dict()

            faculties_df = pd.read_pickle(cache_files['faculties'])
            for _, row in faculties_df.iterrows():
                faculties_cache[row['id']] = row.to_dict()
                faculty_acronym_to_id[row['acronym'].upper()] = row['id']

            # Load classes_cache and assign to existing_classes_cache for TBA class lookup
            classes_df = pd.read_pickle(cache_files['classes'])
            loaded_existing_classes_cache = []
            for _, row in classes_df.iterrows():
                loaded_existing_classes_cache.append(row.to_dict())

            # Assign to both TableBuilder and context for compatibility
            if is_table_builder:
                obj.existing_classes_cache = loaded_existing_classes_cache
            if hasattr(obj, 'context'):
                obj.context.existing_classes_cache = loaded_existing_classes_cache

            bid_window_df = pd.read_pickle(cache_files['bid_window'])
            if not bid_window_df.empty:
                bid_window_id_counter = bid_window_df['id'].max() + 1
                for _, row in bid_window_df.iterrows():
                    bid_window_cache[(row['acad_term_id'], row['round'], row['window'])] = row['id']
            else:
                bid_window_id_counter = 1

            # Set bid_window_id_counter back on obj if TableBuilder
            if is_table_builder:
                obj.bid_window_id_counter = bid_window_id_counter

            logger.info("✅ All cache files loaded successfully.")
            if is_phase1 and is_table_builder:
                logger.info(f"  - Professor lookup entries: {len(professor_lookup)} entries")
            logger.info(f"  - Professors cache: {len(professors_cache)} entries")
            return True

        except Exception as e:
            logger.error(f"❌ Cache loading error: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def check_cache_freshness(obj) -> bool:
        """
        Implements granular, window-aware cache freshness logic.
        The cache is STALE if its modification time is before the results date
        of the previous bidding window.
        """
        logger = DatabaseHelper._get_logger(obj)
        bidding_schedule = DatabaseHelper._get_bidding_schedule(obj)
        cache_dir = DatabaseHelper._get_cache_dir(obj)

        logger.info("🔍 Checking cache freshness with window-aware logic...")
        now = datetime.now()

        # a. Identify the current active bidding window from the schedule.
        # The current round is the first one whose results_date is in the future.
        current_window_index = -1
        if not bidding_schedule:
            logger.info("✅ No bidding schedule found. Assuming cache is fresh.")
            return True

        for i, (results_date, _, _) in enumerate(bidding_schedule):
            if now < results_date:
                current_window_index = i
                break

        # b. If no future window is found, or we are before/in the first window,
        # there is no "previous window" to check against. The cache is fresh.
        if current_window_index <= 0:
            logger.info("✅ Not in an active bidding period or before the second round. Cache is considered fresh.")
            return True

        # c. Get the results_date of the previous bidding window.
        previous_window_info = bidding_schedule[current_window_index - 1]
        previous_round_results_date = previous_window_info[0]

        logger.info(f"ℹ️ Rule: Cache must be newer than the previous window's results date: {previous_round_results_date.strftime('%Y-%m-%d %H:%M')}")

        # d. Check if the cache is fresh by comparing its modification time.
        cache_files_to_check = [
            os.path.join(cache_dir, 'professors_cache.pkl'),
            os.path.join(cache_dir, 'courses_cache.pkl'),
            os.path.join(cache_dir, 'classes_cache.pkl'),
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

    @staticmethod
    def check_record_exists_in_cache(obj, table_name, record_data, key_fields):
        """Check if record exists in database cache and if it needs updates"""
        try:
            cache_dir = DatabaseHelper._get_cache_dir(obj)
            logger = DatabaseHelper._get_logger(obj)
            cache_file = os.path.join(cache_dir, f'{table_name}_cache.pkl')
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

    @staticmethod
    def _combine_with_new_files(obj, output_base, logger) -> None:
        """
        Combine database cache with new CSV files.
        Handles: new_classes.csv, new_bid_window.csv, new_courses.csv
        """
        # Combine new_classes.csv with existing_classes_cache
        new_classes_path = os.path.join(output_base, 'new_classes.csv')
        if os.path.exists(new_classes_path):
            try:
                new_classes_df = pd.read_csv(new_classes_path)
                if not new_classes_df.empty:
                    new_classes_list = new_classes_df.to_dict('records')
                    existing_classes_cache = DatabaseHelper._get_existing_classes_cache(obj)

                    # Safely initialize the cache if it doesn't exist
                    if existing_classes_cache is None:
                        existing_classes_cache = []
                        if hasattr(obj, 'context'):
                            obj.context.existing_classes_cache = existing_classes_cache
                        else:
                            obj.existing_classes_cache = existing_classes_cache

                    # Add new classes, checking for duplicates
                    added_count = 0
                    for new_class in new_classes_list:
                        exists = False
                        for existing_class in existing_classes_cache:
                            if (existing_class['course_id'] == new_class['course_id'] and
                                str(existing_class['section']) == str(new_class['section']) and
                                existing_class['acad_term_id'] == new_class['acad_term_id'] and
                                existing_class.get('professor_id') == new_class.get('professor_id')):
                                exists = True
                                break
                        if not exists:
                            existing_classes_cache.append(new_class)
                            added_count += 1

                    if added_count > 0:
                        logger.info(f"✅ Added {added_count} new, unique classes to the existing cache.")
            except Exception as e:
                logger.warning(f"⚠️ Could not combine new_classes.csv: {e}")

        # Update bid_window_cache with new_bid_window.csv
        new_bid_window_path = os.path.join(output_base, 'new_bid_window.csv')
        if os.path.exists(new_bid_window_path):
            # Get bid_window_cache from obj
            bid_window_cache = None
            if hasattr(obj, 'bid_window_cache'):
                bid_window_cache = obj.bid_window_cache
            elif hasattr(obj, 'context') and hasattr(obj.context, 'bid_window_cache'):
                bid_window_cache = obj.context.bid_window_cache

            if bid_window_cache is not None:
                added_count = merge_bid_window_csv_into_cache(
                    bid_window_cache,
                    new_bid_window_path,
                    logger=logger,
                )
                if added_count > 0:
                    logger.info(f"✅ Added {added_count} new bid windows to the cache.")

        # Update courses_cache with new_courses.csv from multiple locations
        courses_cache = None
        new_courses_list = None
        if hasattr(obj, 'courses_cache'):
            courses_cache = obj.courses_cache
        elif hasattr(obj, 'context') and hasattr(obj.context, 'courses_cache'):
            courses_cache = obj.context.courses_cache

        # Also get new_courses list for database insertion if available
        if hasattr(obj, 'new_courses'):
            new_courses_list = obj.new_courses
        elif hasattr(obj, 'context') and hasattr(obj.context, 'new_courses'):
            new_courses_list = obj.context.new_courses

        if courses_cache is not None:
            new_courses_paths = [
                os.path.join(output_base, 'new_courses.csv'),
            ]
            # Also check verify_dir if available
            verify_dir = DatabaseHelper._get_verify_dir(obj)
            if verify_dir:
                new_courses_paths.append(os.path.join(verify_dir, 'new_courses.csv'))

            for path in new_courses_paths:
                if os.path.exists(path):
                    try:
                        new_courses_df = pd.read_csv(path)
                        if not new_courses_df.empty:
                            added_count = 0
                            for _, row in new_courses_df.iterrows():
                                if row['code'] not in courses_cache:
                                    course_dict = row.to_dict()
                                    # Clean belong_to_faculty: convert NaN/empty to None
                                    if pd.isna(course_dict.get('belong_to_faculty')) or course_dict.get('belong_to_faculty') == '':
                                        course_dict['belong_to_faculty'] = None
                                    courses_cache[row['code']] = course_dict
                                    added_count += 1
                                    # Also add to new_courses list for database insertion ONLY if not already present
                                    if new_courses_list is not None:
                                        # Check if course with same ID is already in new_courses_list
                                        course_id = course_dict.get('id')
                                        if not any(c.get('id') == course_id for c in new_courses_list):
                                            new_courses_list.append(course_dict)
                            if added_count > 0:
                                logger.info(f"✅ Added {added_count} new courses from {path}.")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not combine {path}: {e}")

    @staticmethod
    def load_or_cache_data_with_freshness_check(obj) -> bool:
        """
        Load data with a freshness check, combine it with new CSV files.
        Context-agnostic: works with TableBuilder or ProcessorContext.

        Part 1: Check cache freshness and download if stale
        Part 2: Load core data from cache
        Part 3: Combine with new CSV files
        """
        logger = DatabaseHelper._get_logger(obj)
        output_base = DatabaseHelper._get_output_base(obj)

        # Part 1: Check cache freshness
        if not DatabaseHelper.check_cache_freshness(obj):
            logger.info("🔄 Cache is stale, downloading fresh data from the database...")

            # Get db_adapter and create connection
            db_adapter = None
            if hasattr(obj, 'db_adapter'):
                db_adapter = obj.db_adapter
            elif hasattr(obj, '_db_adapter'):
                db_adapter = obj._db_adapter
            elif hasattr(obj, 'context') and hasattr(obj.context, 'db_adapter'):
                db_adapter = obj.context.db_adapter
            elif hasattr(obj, 'context') and hasattr(obj.context, '_db_adapter'):
                db_adapter = obj.context._db_adapter

            if db_adapter is None:
                logger.error("❌ No db_adapter found")
                return False

            db_connection = DatabaseHelper.create_connection(db_adapter, logger)
            if not db_connection:
                return False

            # Set db_connection on obj
            if hasattr(obj, 'db_connection'):
                obj.db_connection = db_connection
            elif hasattr(obj, 'context'):
                obj.context.db_connection = db_connection

            try:
                DatabaseHelper._download_and_cache_data(obj)
                logger.info("✅ Successfully downloaded fresh data from the database.")
            except Exception as e:
                logger.error(f"❌ Failed to download fresh data: {e}")
                return False

        # Part 2: Load the core data from the cache (either freshly downloaded or existing)
        if not DatabaseHelper.load_or_cache_data(obj):
            return False

        # Part 3: Combine the loaded cache with new data from local CSV files
        logger.info("🔄 Combining database cache with new CSV files...")
        DatabaseHelper._combine_with_new_files(obj, output_base, logger)

        return True

    @staticmethod
    def load_raw_data(obj) -> bool:
        """
        Load raw data from Excel file.
        Context-agnostic: works with TableBuilder or ProcessorContext.
        """
        logger = DatabaseHelper._get_logger(obj)

        # Get input_file from config
        if hasattr(obj, 'config'):
            input_file = obj.config.input_file
        else:
            input_file = obj.input_file

        try:
            logger.info(f"📂 Loading raw data from {input_file}")

            standalone_df = pd.read_excel(input_file, sheet_name='standalone')
            multiple_df = pd.read_excel(input_file, sheet_name='multiple')

            # Set on context or directly on object
            if hasattr(obj, 'context'):
                obj.context.standalone_data = standalone_df
                obj.context.multiple_data = multiple_df
            else:
                obj.standalone_data = standalone_df
                obj.multiple_data = multiple_df

            logger.info(f"✅ Loaded {len(standalone_df)} standalone and {len(multiple_df)} multiple records.")

            # Log available bidding windows for debugging
            if 'bidding_window' in standalone_df.columns:
                available_windows = standalone_df['bidding_window'].dropna().unique()
                logger.info(f"📊 Available bidding windows in data: {sorted(available_windows)}")

            # Log available academic terms for debugging
            if 'acad_term_id' in standalone_df.columns:
                available_terms = standalone_df['acad_term_id'].dropna().unique()
                logger.info(f"📊 Available academic terms in data: {sorted(available_terms)}")

            # Create optimized lookup for multiple_data (stored directly on obj for TableBuilder)
            from collections import defaultdict
            multiple_lookup = defaultdict(list)
            for _, row in multiple_df.iterrows():
                key = row.get('record_key')
                if pd.notna(key):
                    multiple_lookup[key].append(row)

            if hasattr(obj, 'context'):
                obj.context.multiple_lookup = multiple_lookup
            else:
                obj.multiple_lookup = multiple_lookup

            logger.info(f"✅ Created optimized lookup for {len(multiple_lookup)} record keys from unfiltered data.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load raw data: {e}")
            return False

    @staticmethod
    def upsert_df(connection, df, table_name, index_elements, logger=None):
        """
        Perform an UPSERT for a dataframe using psycopg2 execute_values.
        """
        if df.empty:
            return

        from psycopg2.extras import execute_values

        cols = df.columns.tolist()
        update_cols = [col for col in cols if col not in index_elements]
        if not update_cols:
            if logger is not None:
                logger.warning(f"No update columns for upserting into {table_name}. Skipping.")
            return

        sql_stub = f'''
            INSERT INTO "{table_name}" ({', '.join(f'"{c}"' for c in cols)})
            VALUES %s
            ON CONFLICT ({', '.join(f'"{c}"' for c in index_elements)})
            DO UPDATE SET {', '.join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])};
        '''

        cursor = connection.cursor()
        try:
            values = [tuple(row) for row in df.to_numpy()]
            execute_values(cursor, sql_stub, values, page_size=1000)
            if logger is not None:
                logger.info(f"Queued {len(df)} records for upsert into {table_name}.")
        finally:
            cursor.close()

    @staticmethod
    def merge_csv_into_cache(cache_path: str, csv_path: str, key_field: str = 'id') -> int:
        """
        Merge new CSV records into existing cache pickle.
        Returns number of new records merged. Does nothing if CSV doesn't exist.
        """
        if not os.path.exists(csv_path):
            return 0

        new_df = pd.read_csv(csv_path)
        if new_df.empty:
            return 0

        if os.path.exists(cache_path):
            cache_df = pd.read_pickle(cache_path)
            # Only keep columns that exist in both
            new_df = new_df[[col for col in new_df.columns if col in cache_df.columns]]
            combined = pd.concat([cache_df, new_df], ignore_index=True).drop_duplicates(subset=[key_field])
        else:
            combined = new_df

        combined.to_pickle(cache_path)
        return len(new_df)
