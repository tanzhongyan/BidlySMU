"""
ClassAvailabilityProcessor - handles class availability CREATE logic.
Extracted from table_builder.py process_class_availability method.
"""
import os
from datetime import datetime
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext
from src.utils.schedule_resolver import parse_window_name
from src.utils.schedule_resolver import get_current_live_window_name
from src.utils.class_id_resolver import find_all_class_ids


class ClassAvailabilityProcessor(AbstractProcessor):
    """Processes class availability records from standalone data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        pass

    def _do_process(self) -> None:
        """Execute class availability processing logic."""
        self._logger.info("Processing class availability from raw_data...")

        # === STEP 1: Determine Current Bidding Window ===
        now = datetime.now()
        bidding_schedule_for_term = (self.context.config.bidding_schedules or {}).get(self.context.config.start_ay_term, [])
        current_window_name = get_current_live_window_name(bidding_schedule_for_term, now)

        self._logger.info(f"Processing class availability for current window: '{current_window_name}'")

        # === STEP 2: Filter the data to only current window and current term records ===
        if current_window_name and hasattr(self.context, 'standalone_data') and not self.context.standalone_data.empty:
            if 'bidding_window' in self.context.standalone_data.columns:
                original_count = len(self.context.standalone_data)

                # Filter by bidding window
                current_window_data = self.context.standalone_data[
                    self.context.standalone_data['bidding_window'] == current_window_name
                ].copy()

                # Also filter by current academic term to prevent cross-term contamination
                if 'acad_term_id' in current_window_data.columns:
                    expected_term_id = self.context.expected_acad_term_id

                    before_term_filter = len(current_window_data)
                    current_window_data = current_window_data[
                        current_window_data['acad_term_id'] == expected_term_id
                    ].copy()

                    self._logger.info(f"Filtered data: {original_count} -> {before_term_filter} (window) -> {len(current_window_data)} (window + term)")
                    self._logger.info(f"    Window filter: '{current_window_name}', Term filter: '{expected_term_id}'")
                else:
                    self._logger.info(f"Filtered data from {original_count} to {len(current_window_data)} records for current window: '{current_window_name}'")
            else:
                self._logger.warning("No 'bidding_window' column found - processing all data")
                current_window_data = self.context.standalone_data.copy()
        else:
            self._logger.warning("Could not determine current window or no standalone data - processing all data")
            current_window_data = self.context.standalone_data.copy() if hasattr(self.context, 'standalone_data') else pd.DataFrame()

        # Load existing class availability data to prevent duplicates
        existing_availability_keys = set()
        cache_file = self.context.config.cache_dir + '/class_availability_cache.pkl' if hasattr(self.context.config, 'cache_dir') else None
        if cache_file:
            if os.path.exists(cache_file):
                try:
                    existing_df = pd.read_pickle(cache_file)
                    if not existing_df.empty:
                        for _, record in existing_df.iterrows():
                            key = (record['class_id'], record['bid_window_id'])
                            existing_availability_keys.add(key)
                        self._logger.info(f"Pre-loaded {len(existing_availability_keys)} existing class availability keys from cache.")
                except Exception as e:
                    self._logger.warning(f"Could not pre-load class_availability_cache: {e}")

        # Track keys from current run to prevent duplicates within the same processing
        current_run_keys = set()
        for availability_record in self.context.new_class_availability:
            key = (availability_record['class_id'], availability_record['bid_window_id'])
            current_run_keys.add(key)

        newly_created_count = 0

        # === STEP 3: Process only the filtered current window data ===
        for _, row in current_window_data.iterrows():
            course_code = row.get('course_code')
            section = row.get('section')
            acad_term_id = row.get('acad_term_id')
            bidding_window_str = row.get('bidding_window')

            if pd.isna(course_code) or pd.isna(section) or pd.isna(acad_term_id) or pd.isna(bidding_window_str):
                continue

            round_str, window_num = parse_window_name(bidding_window_str)
            if not all([round_str, window_num]):
                continue

            class_boss_id = row.get('class_boss_id')
            class_ids = find_all_class_ids(
                acad_term_id, class_boss_id,
                self.context.new_classes, self.context.existing_classes_cache
            )

            if not class_ids:
                failed_row = {
                    'course_code': course_code, 'section': section, 'acad_term_id': acad_term_id,
                    'bidding_window_str': bidding_window_str, 'reason': 'class_not_found'
                }
                self.context.failed_mappings.append(failed_row)
                self.context.stats['failed_mappings'] = self.context.stats.get('failed_mappings', 0) + 1
                continue

            window_key = (acad_term_id, round_str, window_num)
            bid_window_id = self.context.bid_window_cache.get(window_key)
            if not bid_window_id:
                self._logger.warning(f"No bid_window_id for {window_key}")
                continue

            # Extract values safely
            total_val = int(row.get('total')) if pd.notna(row.get('total')) else 0
            current_enrolled_val = int(row.get('current_enrolled')) if pd.notna(row.get('current_enrolled')) else 0
            reserved_val = int(row.get('reserved')) if pd.notna(row.get('reserved')) else 0
            available_val = int(row.get('available')) if pd.notna(row.get('available')) else 0

            for class_id in class_ids:
                # Check for existence in both existing data and current run
                availability_key = (class_id, bid_window_id)

                if availability_key in existing_availability_keys or availability_key in current_run_keys:
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

                self.context.new_class_availability.append(availability_record)
                current_run_keys.add(availability_key)
                newly_created_count += 1

        self.context.boss_stats['class_availability_created'] = self.context.boss_stats.get('class_availability_created', 0) + newly_created_count
        self._logger.info(f"Class availability checks complete. Created {newly_created_count} new records.")

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass