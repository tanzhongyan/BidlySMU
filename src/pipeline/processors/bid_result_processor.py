"""
BidResultProcessor - handles bid result CREATE and UPDATE logic.
Extracted from table_builder.py process_bid_results method.
"""
import os
from datetime import datetime
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext
from src.utils.schedule_resolver import parse_window_name
from src.utils.class_id_resolver import find_all_class_ids
from src.utils.cache_resolver import safe_int, safe_float


class BidResultProcessor(AbstractProcessor):
    """Processes bid result records from standalone data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        pass

    def _do_process(self) -> None:
        """Execute bid result processing logic."""
        self._logger.info("Processing bid results from raw_data...")

        # Ensure the list for update records exists
        if not hasattr(self.context, 'update_bid_result'):
            self.context.update_bid_result = []

        # === STEP 1: Determine Current and Previous Bidding Windows ===
        now = datetime.now()
        current_window_name = None

        # Get the bidding schedule for the current term
        bidding_schedule_for_term = (self.context.config.bidding_schedules or {}).get(self.context.config.start_ay_term, [])

        if bidding_schedule_for_term:
            # Find the current window (first future window)
            for i, (results_date, window_name, folder_suffix) in enumerate(bidding_schedule_for_term):
                if now < results_date:
                    current_window_name = window_name
                    break

            # If no future window found, we're past all scheduled windows
            if current_window_name is None and bidding_schedule_for_term:
                current_window_name = bidding_schedule_for_term[-1][1]

        self._logger.info(f"Processing bid results for current window: '{current_window_name}'")

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

                    current_window_data = current_window_data[
                        current_window_data['acad_term_id'] == expected_term_id
                    ].copy()

                    self._logger.info(f"Filtered data: {original_count} -> {len(current_window_data)} (window + term)")
                else:
                    self._logger.info(f"Filtered data from {original_count} to {len(current_window_data)} records for current window: '{current_window_name}'")
            else:
                self._logger.warning("No 'bidding_window' column found - processing all data")
                current_window_data = self.context.standalone_data.copy()
        else:
            self._logger.warning("Could not determine current window or no standalone data - processing all data")
            current_window_data = self.context.standalone_data.copy() if hasattr(self.context, 'standalone_data') else pd.DataFrame()

        # Load existing bid_result data to check for duplicates
        existing_bid_result_keys = set()
        existing_bid_results = {}  # Store full records for update comparison
        cache_file = self.context.config.cache_dir + '/bid_result_cache.pkl' if hasattr(self.context.config, 'cache_dir') else None
        if cache_file:
            if os.path.exists(cache_file):
                try:
                    existing_df = pd.read_pickle(cache_file)
                    if not existing_df.empty:
                        for _, record in existing_df.iterrows():
                            key = (record['bid_window_id'], record['class_id'])
                            existing_bid_result_keys.add(key)
                            existing_bid_results[key] = record.to_dict()
                        self._logger.info(f"Pre-loaded {len(existing_bid_result_keys)} existing bid result keys from cache.")
                except Exception as e:
                    self._logger.warning(f"Could not pre-load bid_result_cache: {e}")

        newly_created_count = 0
        updated_count = 0

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

                round_str, window_num = parse_window_name(bidding_window_str)
                if not all([round_str, window_num]):
                    continue

                class_ids = find_all_class_ids(
                    acad_term_id, class_boss_id,
                    self.context.new_classes, self.context.existing_classes_cache
                )
                if not class_ids:
                    continue

                window_key = (acad_term_id, round_str, window_num)
                bid_window_id = self.context.bid_window_cache.get(window_key)
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
                            break

                # Try all possible column names for min
                min_column_names = ['min', 'Min', 'Min Bid', 'min_bid', 'Min_Bid', 'MIN']
                for col_name in min_column_names:
                    if col_name in row.index:
                        val = row[col_name]
                        if pd.notna(val):
                            min_bid = val
                            break

                has_bid_data = pd.notna(median_bid) or pd.notna(min_bid)

                # Prepare data record
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
                            self.context.update_bid_result.append(result_data)
                            updated_count += 1
                    else:
                        # This is a NEW record
                        self.context.new_bid_result.append(result_data)
                        existing_bid_result_keys.add(bid_result_key)
                        newly_created_count += 1

            except Exception as e:
                self._logger.error(f"Error processing bid result row for {row.get('course_code')}-{row.get('section')}: {e}")

        self.context.boss_stats['bid_results_created'] = self.context.boss_stats.get('bid_results_created', 0) + newly_created_count
        self._logger.info(f"Bid result checks complete. Created: {newly_created_count}, Updated: {updated_count}.")

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass