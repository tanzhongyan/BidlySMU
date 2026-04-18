"""
BOSSProcessor - handles BOSS (Business Object Scheduling System) processing logic.
Extracted from table_builder.py run_phase3_boss_processing and related methods.
"""
import os
import pandas as pd
from datetime import datetime
from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext
from src.pipeline.processors.acad_term_processor import AcadTermProcessor
from src.pipeline.processors.professor_processor import ProfessorProcessor
from src.pipeline.processors.course_processor import CourseProcessor
from src.pipeline.processors.class_processor import ClassProcessor
from src.pipeline.processors.timing_processor import TimingProcessor
from src.utils.schedule_resolver import parse_window_name
from src.utils.class_id_resolver import find_all_class_ids
from src.utils.cache_resolver import safe_int, safe_float
from src.db.database_helper import DatabaseHelper


class BOSSProcessor(AbstractProcessor):
    """Processes BOSS (Bidding) data - vacancy, enrollment, and bid results."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        pass

    def _do_process(self) -> None:
        """Execute BOSS processing logic."""
        self._logger.info("🚀 Starting Enhanced Phase 3: BOSS Processing")
        self._logger.info("=" * 50)

        # Pre-Phase 3: Update caches with newly created records
        self._update_caches_from_previous_phases()

        # Initialize BOSS processing
        self._setup_boss_processing()

        # Step 1: Load all required data
        if not self._load_all_data():
            return

        # Step 2: Process base entities
        self._process_base_entities()

        # Step 3: Sequential catch-up processing for all windows
        self._process_all_bidding_windows()

        # Step 4: Save all outputs
        self._save_all_outputs()

        # Step 5: Final summary
        self._print_boss_summary()

        self._logger.info("📝 ✅ Enhanced Phase 3 BOSS processing completed successfully!")

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass

    def _update_caches_from_previous_phases(self) -> None:
        """Update caches with newly created records from previous phases."""
        self._logger.info("🛠️ Pre-Phase 3: Updating cache with newly created records...")

        # Update professors cache
        prof_cache_path = os.path.join(self.context.config.cache_dir, 'professors_cache.pkl')
        new_profs_path = os.path.join(self.context.config.verify_dir, 'new_professors.csv')
        count = DatabaseHelper.merge_csv_into_cache(prof_cache_path, new_profs_path)
        if count > 0:
            self._logger.info(f"   ✅ Updated professors_cache.pkl with {count} new records.")

        # Update classes cache
        class_cache_path = os.path.join(self.context.config.cache_dir, 'classes_cache.pkl')
        new_classes_path = os.path.join(self.context.config.output_base, 'new_classes.csv')
        count = DatabaseHelper.merge_csv_into_cache(class_cache_path, new_classes_path)
        if count > 0:
            self._logger.info(f"   ✅ Updated classes_cache.pkl with {count} new records.")

    def _setup_boss_processing(self) -> None:
        """Initialize BOSS results processing with logging and caches."""
        self._logger.info("🔄 Initializing BOSS processing...")

        # Setup logging
        self.context.boss_log_file = os.path.join(self.context.config.output_base, 'boss_result_log.txt')

        try:
            with open(self.context.boss_log_file, 'w') as f:
                f.write(f"BOSS Results Processing Log - {datetime.now().isoformat()}\n")
                f.write("=" * 70 + "\n\n")
            self._logger.info(f"📝 Log file created: {self.context.boss_log_file}")
        except Exception as e:
            self._logger.info(f"⚠️ Warning: Could not create log file {self.context.boss_log_file}: {e}")
            self.context.boss_log_file = None

        # Initialize data storage
        self.context.boss_data = []
        self.context.failed_mappings = []

        # Output collectors
        self.context.new_bid_windows = []
        self.context.new_class_availability = []
        self.context.new_bid_result = []
        self.context.update_bid_result = []

        # Caches for deduplication
        self.context.bid_window_cache = {}

        # Initialize bid_window_id_counter
        self.context.bid_window_id_counter = 1

        # Load existing bid_window data
        self._load_existing_bid_windows()

        # Initialize boss_stats
        self.context.boss_stats = {
            'files_processed': 0,
            'total_rows': 0,
            'bid_windows_created': 0,
            'class_availability_created': 0,
            'bid_results_created': 0,
            'failed_mappings': 0
        }

        self._logger.info("🔄 BOSS processing initialization completed")

    def _load_existing_bid_windows(self) -> None:
        """Load existing bid windows and build cache."""
        try:
            cache_file = os.path.join(self.context.config.cache_dir, 'bid_window_cache.pkl')
            if os.path.exists(cache_file):
                bid_window_df = pd.read_pickle(cache_file)
                if not bid_window_df.empty:
                    max_id = bid_window_df['id'].max()
                    self.context.bid_window_id_counter = max_id + 1

                    for _, row in bid_window_df.iterrows():
                        window_key = (row['acad_term_id'], row['round'], row['window'])
                        self.context.bid_window_cache[window_key] = row['id']

                    self._log_boss_activity(f"✅ Loaded {len(bid_window_df)} existing bid windows, next ID will be {self.context.bid_window_id_counter}")
                else:
                    self._log_boss_activity("⚠️ Bid window cache exists but is empty, starting from ID 1")
            else:
                # Try database if cache doesn't exist
                if self.context.db_connection:
                    try:
                        query = "SELECT * FROM bid_window ORDER BY id"
                        bid_window_df = pd.read_sql_query(query, self.context.db_connection)
                        if not bid_window_df.empty:
                            bid_window_df.to_pickle(cache_file)
                            max_id = bid_window_df['id'].max()
                            self.context.bid_window_id_counter = max_id + 1

                            for _, row in bid_window_df.iterrows():
                                window_key = (row['acad_term_id'], row['round'], row['window'])
                                self.context.bid_window_cache[window_key] = row['id']

                            self._log_boss_activity(f"✅ Downloaded {len(bid_window_df)} bid windows from database")
                        else:
                            self._log_boss_activity("⚠️ Database bid_window table is empty")
                    except Exception as e:
                        self._log_boss_activity(f"⚠️ Could not download bid_window from database: {e}")
                else:
                    self._log_boss_activity("⚠️ No bid window cache found and no database connection")
        except Exception as e:
            self._log_boss_activity(f"⚠️ Error initializing bid_window counter: {e}")
            self.context.bid_window_id_counter = 1

    def _load_all_data(self) -> bool:
        """Load all required data from cache and input files."""
        self._log_boss_activity("🔄 Loading all required data caches with freshness check...")

        # Use DatabaseHelper's method to load with freshness check
        if not DatabaseHelper.load_or_cache_data_with_freshness_check(self.context):
            return False

        self._log_boss_activity("🔄 Loading all raw input files...")
        if not DatabaseHelper.load_raw_data(self.context):
            return False

        # Load overall boss results
        self.context.overall_boss_results_df = self._load_overall_boss_results()

        return True

    def _load_overall_boss_results(self) -> pd.DataFrame:
        """Load overall BOSS results from script_input/overallBossResults/"""
        self._logger.info("📊 Loading overall BOSS results...")

        now = datetime.now()
        current_round_info = self._get_bidding_round_info_for_term(self.context.config.start_ay_term, now)

        if not current_round_info:
            self._logger.info("⏭️ Not in active bidding period - skipping overall results")
            return None

        overall_results_dir = 'script_input/overallBossResults'
        if not os.path.exists(overall_results_dir):
            self._logger.warning(f"⚠️ Overall results directory not found: {overall_results_dir}")
            return None

        results_file = os.path.join(overall_results_dir, f"{self.context.config.start_ay_term}.xlsx")
        if not os.path.exists(results_file):
            self._logger.warning(f"⚠️ Overall results file not found: {results_file}")
            return None

        try:
            df = pd.read_excel(results_file)
            self._logger.info(f"✅ Loaded {len(df)} overall BOSS results from {results_file}")
            return df
        except Exception as e:
            self._logger.error(f"❌ Error loading overall results: {e}")
            return None

    def _get_bidding_round_info_for_term(self, ay_term, now):
        """Get current bidding round info for a term."""
        bidding_schedule = (self.context.config.bidding_schedules or {}).get(self.context.config.start_ay_term, [])
        for results_date, _, folder_suffix in bidding_schedule:
            if now < results_date:
                return f"{ay_term}_{folder_suffix}"
        return None

    def _process_base_entities(self) -> None:
        """Process base entities (AcadTerms, Professors, Courses, Classes, Timings)."""
        self._log_boss_activity("🔄 Processing base entities (AcadTerms, Professors, Courses, Classes, Timings)...")

        # Process in correct dependency order:
        # 1. AcadTerms - no dependencies
        # 2. Professors - no dependencies
        # 3. Courses - depends on faculties (already in cache)
        # 4. Classes - depends on courses, professors, acad_terms
        # 5. Timings - depends on classes

        # 1. Process Academic Terms
        self.context.logger.info("Processing acad_terms...")
        try:
            acad_term_processor = AcadTermProcessor(self.context)
            acad_term_processor.process()
            self.context.logger.info("✅ AcadTerm processing complete.")
        except Exception as e:
            self.context.logger.error(f"❌ Error in AcadTermProcessor: {e}")
            raise

        # 2. Process Professors
        self.context.logger.info("Processing professors...")
        try:
            professor_processor = ProfessorProcessor(self.context)
            professor_processor.process()
            self.context.logger.info("✅ Professor processing complete.")
        except Exception as e:
            self.context.logger.error(f"❌ Error in ProfessorProcessor: {e}")
            raise

        # 3. Process Courses
        self.context.logger.info("Processing courses...")
        try:
            course_processor = CourseProcessor(self.context)
            course_processor.process()
            self.context.logger.info(f"✅ Course processing complete. New: {self.context.stats.get('courses_created', 0)}, Updated: {self.context.stats.get('courses_updated', 0)}")
        except Exception as e:
            self.context.logger.error(f"❌ Error in CourseProcessor: {e}")
            raise

        # 4. Process Classes
        self.context.logger.info("Processing classes...")
        try:
            class_processor = ClassProcessor(self.context)
            class_processor.process()
            self.context.logger.info(f"✅ Class processing complete. New: {self.context.stats.get('classes_created', 0)}, Updated: {len(self.context.update_classes)}")
        except Exception as e:
            self.context.logger.error(f"❌ Error in ClassProcessor: {e}")
            raise

        # 5. Process Timings
        self.context.logger.info("Processing timings...")
        try:
            timing_processor = TimingProcessor(self.context)
            timing_processor.process()
            self.context.logger.info("✅ Timing processing complete.")
        except Exception as e:
            self.context.logger.error(f"❌ Error in TimingProcessor: {e}")
            raise

        self._log_boss_activity("✅ Base entity processing complete.")

    def _process_all_bidding_windows(self) -> None:
        """Sequential catch-up processing for all bidding windows."""
        self._log_boss_activity("🔄 Starting sequential catch-up processing...")

        now = datetime.now()
        bidding_schedule_for_term = (self.context.config.bidding_schedules or {}).get(self.context.config.start_ay_term, [])

        if not bidding_schedule_for_term:
            self._log_boss_activity("❌ No bidding schedule found for current term")
            return

        # Find current live window
        current_window_index = None
        for i, (results_date, window_name, folder_suffix) in enumerate(bidding_schedule_for_term):
            if now < results_date:
                current_window_index = i
                break

        if current_window_index is None:
            current_window_index = len(bidding_schedule_for_term) - 1

        current_window_name = bidding_schedule_for_term[current_window_index][1]
        self._log_boss_activity(f"🎯 Current live window: {current_window_name}")

        # Processing range: from start to current window (inclusive)
        processing_range = bidding_schedule_for_term[:current_window_index + 1]
        self._log_boss_activity(f"📅 Processing {len(processing_range)} windows chronologically")

        # Load existing bid result keys for deduplication
        existing_bid_result_keys = self._load_existing_bid_result_keys()

        # Sequential processing loop
        for window_index, (results_date, window_name, folder_suffix) in enumerate(processing_range):
            self._log_boss_activity(f"🔄 Processing window {window_index + 1}/{len(processing_range)}: {window_name}")

            round_str, window_num = parse_window_name(window_name)
            if not round_str or not window_num:
                self._log_boss_activity(f"⚠️ Could not parse window: {window_name}")
                continue

            acad_term_id = self.context.expected_acad_term_id
            window_key = (acad_term_id, round_str, window_num)

            # Find or create BidWindow
            bid_window_id = self.context.bid_window_cache.get(window_key)
            if not bid_window_id:
                bid_window_id = self.context.bid_window_id_counter
                new_bid_window = {
                    'id': bid_window_id,
                    'acad_term_id': acad_term_id,
                    'round': round_str,
                    'window': window_num
                }
                self.context.new_bid_windows.append(new_bid_window)
                self.context.bid_window_cache[window_key] = bid_window_id
                self.context.bid_window_id_counter += 1
                self.context.boss_stats['bid_windows_created'] += 1
                self._log_boss_activity(f"✅ Created bid window {bid_window_id}: {window_name}")

            # Process ClassAvailability
            self._process_class_availability(window_name, bid_window_id, acad_term_id)

            # Process BidResult
            is_current_live = (window_index == current_window_index)
            self._process_bid_results(window_name, bid_window_id, acad_term_id, existing_bid_result_keys, is_current_live)

        self._log_boss_activity("✅ Sequential catch-up processing completed")

    def _load_existing_bid_result_keys(self) -> set:
        """Load existing bid result keys for deduplication."""
        existing_bid_result_keys = set()
        cache_file = os.path.join(self.context.config.cache_dir, 'bid_result_cache.pkl')
        if os.path.exists(cache_file):
            try:
                existing_df = pd.read_pickle(cache_file)
                if not existing_df.empty:
                    for _, record in existing_df.iterrows():
                        key = (record['bid_window_id'], record['class_id'])
                        existing_bid_result_keys.add(key)
                self._log_boss_activity(f"✅ Pre-loaded {len(existing_bid_result_keys)} existing bid result keys")
            except Exception as e:
                self._log_boss_activity(f"⚠️ Could not pre-load bid_result_cache: {e}")
        return existing_bid_result_keys

    def _process_class_availability(self, window_name: str, bid_window_id: int, acad_term_id: str) -> None:
        """Process class availability for a specific window."""
        if not hasattr(self.context, 'standalone_data') or self.context.standalone_data is None:
            return

        window_data = self.context.standalone_data[
            (self.context.standalone_data['bidding_window'] == window_name) &
            (self.context.standalone_data['acad_term_id'] == acad_term_id)
        ]

        if window_data.empty:
            self._log_boss_activity(f"⏭️ Skipping ClassAvailability for {window_name} (no scrape data)")
            return

        self._log_boss_activity(f"📊 Processing ClassAvailability for {window_name} from raw_data.xlsx")

        # Load existing availability keys
        existing_availability_keys = self._load_existing_availability_keys()

        for _, row in window_data.iterrows():
            course_code = row.get('course_code')
            section = row.get('section')
            class_boss_id = row.get('class_boss_id')

            if pd.isna(course_code) or pd.isna(section) or pd.isna(class_boss_id):
                continue

            class_ids = find_all_class_ids(
                acad_term_id, class_boss_id,
                self.context.new_classes,
                getattr(self.context, 'existing_classes_cache', []),
                self.context.config.output_base
            )
            if not class_ids:
                continue

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
                    self.context.new_class_availability.append(availability_record)
                    existing_availability_keys.add(availability_key)
                    self.context.boss_stats['class_availability_created'] += 1

    def _load_existing_availability_keys(self) -> set:
        """Load existing class availability keys."""
        existing_availability_keys = set()
        avail_cache_file = os.path.join(self.context.config.cache_dir, 'class_availability_cache.pkl')
        if os.path.exists(avail_cache_file):
            try:
                avail_df = pd.read_pickle(avail_cache_file)
                if not avail_df.empty:
                    for _, record in avail_df.iterrows():
                        key = (record['class_id'], record['bid_window_id'])
                        existing_availability_keys.add(key)
            except Exception:
                pass
        return existing_availability_keys

    def _process_bid_results(self, window_name: str, bid_window_id: int, acad_term_id: str,
                            existing_bid_result_keys: set, is_current_live: bool) -> None:
        """Process bid results for a specific window."""
        if not hasattr(self.context, 'standalone_data') or self.context.standalone_data is None:
            return

        window_data = self.context.standalone_data[
            (self.context.standalone_data['bidding_window'] == window_name) &
            (self.context.standalone_data['acad_term_id'] == acad_term_id)
        ]

        if is_current_live:
            # Current live window: create placeholder BidResult
            self._log_boss_activity(f"📈 Processing placeholder BidResult for current live window: {window_name}")

            if not window_data.empty:
                for _, row in window_data.iterrows():
                    class_boss_id = row.get('class_boss_id')
                    if pd.isna(class_boss_id):
                        continue

                    class_ids = find_all_class_ids(
                        acad_term_id, class_boss_id,
                        self.context.new_classes,
                        getattr(self.context, 'existing_classes_cache', []),
                        self.context.config.output_base
                    )
                    if not class_ids:
                        continue

                    total_val = safe_int(row.get('total'))
                    enrolled_val = safe_int(row.get('current_enrolled'))

                    for class_id in class_ids:
                        bid_result_key = (bid_window_id, class_id)
                        if bid_result_key not in existing_bid_result_keys:
                            result_data = {
                                'bid_window_id': bid_window_id,
                                'class_id': class_id,
                                'vacancy': total_val,
                                'opening_vacancy': safe_int(row.get('opening_vacancy')),
                                'before_process_vacancy': total_val - enrolled_val if total_val is not None and enrolled_val is not None else None,
                                'dice': safe_int(row.get('d_i_c_e') or row.get('dice')),
                                'after_process_vacancy': safe_int(row.get('after_process_vacancy')),
                                'enrolled_students': enrolled_val,
                                'median': None,
                                'min': None
                            }
                            self.context.new_bid_result.append(result_data)
                            existing_bid_result_keys.add(bid_result_key)
                            self.context.boss_stats['bid_results_created'] += 1
        else:
            # Historical window: process from overallBossResults.xlsx
            self._log_boss_activity(f"📈 Processing historical BidResult for {window_name} from overallBossResults.xlsx")

            if self.context.overall_boss_results_df is not None and not self.context.overall_boss_results_df.empty:
                self._process_historical_bid_results(bid_window_id, acad_term_id, existing_bid_result_keys, window_name)

    def _process_historical_bid_results(self, bid_window_id: int, acad_term_id: str,
                                        existing_bid_result_keys: set, window_name: str) -> None:
        """Process historical bid results from overallBossResults.xlsx."""
        overall_df = self.context.overall_boss_results_df.copy()

        # Find bidding window column
        bidding_window_col = None
        for col in overall_df.columns:
            if 'bidding window' in col.lower() or 'bidding_window' in col.lower():
                bidding_window_col = col
                break

        if not bidding_window_col:
            self._log_boss_activity(f"⚠️ No bidding window column found in overallBossResults.xlsx")
            return

        # Parse and filter for current window
        parsed_windows = overall_df[bidding_window_col].apply(parse_window_name)
        overall_df['round'] = parsed_windows.apply(lambda x: x[0] if isinstance(x, tuple) else None)
        overall_df['window'] = parsed_windows.apply(lambda x: x[1] if isinstance(x, tuple) else None)

        overall_df.dropna(subset=['round', 'window'], inplace=True)
        overall_df['round'] = overall_df['round'].astype(str)
        overall_df['window'] = pd.to_numeric(overall_df['window']).astype(int)

        round_str, window_num = parse_window_name(window_name)
        window_filtered_df = overall_df[
            (overall_df['round'] == str(round_str)) &
            (overall_df['window'] == int(window_num))
        ]

        if window_filtered_df.empty:
            self._log_boss_activity(f"⚠️ No data found in overallBossResults for {window_name}")
            return

        for _, row in window_filtered_df.iterrows():
            course_code = self._get_column_value(row, ['Course Code', 'course_code', 'Course_Code'])
            section = self._get_column_value(row, ['Section', 'section'])

            if pd.isna(course_code) or pd.isna(section):
                continue

            class_boss_id = self._find_class_boss_id_from_course_section(course_code, section, acad_term_id)
            if not class_boss_id:
                continue

            class_ids = find_all_class_ids(
                acad_term_id, class_boss_id,
                self.context.new_classes,
                getattr(self.context, 'existing_classes_cache', []),
                self.context.config.output_base
            )
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

                bid_result_key = (bid_window_id, class_id)
                if bid_result_key in existing_bid_result_keys:
                    self.context.update_bid_result.append(result_data)
                else:
                    self.context.new_bid_result.append(result_data)
                    existing_bid_result_keys.add(bid_result_key)
                    self.context.boss_stats['bid_results_created'] += 1

    def _get_column_value(self, row, possible_names):
        """Helper to get column value by trying multiple possible names."""
        for name in possible_names:
            if name in row.index:
                return row[name]
        return None

    def _find_class_boss_id_from_course_section(self, course_code, section, acad_term_id):
        """Find class_boss_id from course_code + section + acad_term_id."""
        if not hasattr(self.context, 'standalone_data') or self.context.standalone_data is None:
            return None

        matches = self.context.standalone_data[
            (self.context.standalone_data['course_code'] == course_code) &
            (self.context.standalone_data['section'].astype(str) == str(section).strip()) &
            (self.context.standalone_data['acad_term_id'] == acad_term_id)
        ]

        if not matches.empty:
            return matches.iloc[0].get('class_boss_id')

        return None

    def _save_all_outputs(self) -> None:
        """Save all output files."""
        self._log_boss_activity("💾 Saving BOSS output files...")

        # Save bid windows
        if self.context.new_bid_windows:
            df = pd.DataFrame(self.context.new_bid_windows)
            output_path = os.path.join(self.context.config.output_base, 'new_bid_window.csv')
            df.to_csv(output_path, index=False)
            self._log_boss_activity(f"✅ Saved {len(self.context.new_bid_windows)} bid windows to new_bid_window.csv")

        # Save class availability
        if self.context.new_class_availability:
            df = pd.DataFrame(self.context.new_class_availability)
            output_path = os.path.join(self.context.config.output_base, 'new_class_availability.csv')
            df.to_csv(output_path, index=False)
            self._log_boss_activity(f"✅ Saved {len(self.context.new_class_availability)} availability records")

        # Save bid results
        if self.context.new_bid_result:
            df = pd.DataFrame(self.context.new_bid_result)
            output_path = os.path.join(self.context.config.output_base, 'new_bid_result.csv')
            df.to_csv(output_path, index=False)
            self._log_boss_activity(f"✅ Saved {len(self.context.new_bid_result)} bid results")

        # Save failed mappings
        if self.context.failed_mappings:
            df = pd.DataFrame(self.context.failed_mappings)
            output_path = os.path.join(self.context.config.output_base, 'failed_boss_results_mapping.csv')
            df.to_csv(output_path, index=False)
            self._log_boss_activity(f"⚠️ Saved {len(self.context.failed_mappings)} failed mappings")

        self._log_boss_activity("✅ All BOSS output files saved successfully")

    def _print_boss_summary(self) -> None:
        """Print BOSS processing summary."""
        self._logger.info("\n" + "=" * 70)
        self._logger.info("📊 BOSS RESULTS PROCESSING SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info(f"📂 Files processed: {self.context.boss_stats['files_processed']}")
        self._logger.info(f"📄 Total rows: {self.context.boss_stats['total_rows']}")
        self._logger.info(f"🪟 Bid windows created: {self.context.boss_stats['bid_windows_created']}")
        self._logger.info(f"📊 Class availability records: {self.context.boss_stats['class_availability_created']}")
        self._logger.info(f"📈 Bid result records: {self.context.boss_stats['bid_results_created']}")
        self._logger.info(f"❌ Failed mappings: {self.context.boss_stats['failed_mappings']}")
        self._logger.info("=" * 70)

        self._logger.info("\n📁 OUTPUT FILES:")
        self._logger.info(f"   - new_bid_window.csv ({self.context.boss_stats['bid_windows_created']} records)")
        self._logger.info(f"   - new_class_availability.csv ({self.context.boss_stats['class_availability_created']} records)")
        self._logger.info(f"   - new_bid_result.csv ({self.context.boss_stats['bid_results_created']} records)")
        if self.context.boss_stats['failed_mappings'] > 0:
            self._logger.info(f"   - failed_boss_results_mapping.csv ({self.context.boss_stats['failed_mappings']} records)")
        self._logger.info(f"   - boss_result_log.txt (processing log)")
        self._logger.info("=" * 70)

    def _log_boss_activity(self, message, print_to_stdout=True):
        """Log activity to both file and optionally stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        if hasattr(self.context, 'boss_log_file') and self.context.boss_log_file:
            try:
                with open(self.context.boss_log_file, 'a') as f:
                    f.write(log_message)
            except Exception as e:
                self._logger.info(f"⚠️ Warning: Could not write to log file: {e}")

        if print_to_stdout:
            self._logger.info(f"📝 {message}")