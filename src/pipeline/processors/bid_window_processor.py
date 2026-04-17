"""
BidWindowProcessor - handles bid window CREATE logic.
Extracted from table_builder.py process_bid_windows method.
"""
from collections import defaultdict
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext
from src.utils.schedule_resolver import parse_window_name


class BidWindowProcessor(AbstractProcessor):
    """Processes bid window records from boss_data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)

    def _load_cache(self) -> None:
        # Bid window cache already loaded into context.bid_window_cache by TableBuilder
        pass

    def _do_process(self) -> None:
        """Execute bid window processing logic."""
        self._logger.info("Processing bid windows from boss_data...")

        if self.context.boss_data is None or len(self.context.boss_data) == 0:
            self._logger.error("No BOSS data loaded")
            return

        # Track all unique bid windows found in data
        found_windows = defaultdict(set)  # acad_term_id -> set of (round, window) tuples

        # Discover all windows that exist in the data
        for _, row in self.context.boss_data.iterrows():
            acad_term_id = row.get('acad_term_id')
            bidding_window_str = row.get('bidding_window')

            if pd.isna(acad_term_id) or pd.isna(bidding_window_str):
                continue

            round_str, window_num = parse_window_name(bidding_window_str)

            if acad_term_id and round_str and window_num:
                found_windows[acad_term_id].add((round_str, window_num))

        # Use the counter that was set from existing data
        bid_window_id = self.context.bid_window_id_counter
        round_order = {'1': 1, '1A': 2, '1B': 3, '1C': 4, '1F': 5, '2': 6, '2A': 7}

        for acad_term_id in sorted(found_windows.keys()):
            windows_for_term = found_windows[acad_term_id]
            sorted_windows = sorted(windows_for_term, key=lambda x: (round_order.get(x[0], 99), x[1]))

            self._logger.info(f"Processing {acad_term_id}: found {len(sorted_windows)} windows")

            for round_str, window_num in sorted_windows:
                window_key = (acad_term_id, round_str, window_num)

                # Skip if already exists in database
                if window_key in self.context.bid_window_cache:
                    self._logger.info(f"Bid window already exists: {acad_term_id} Round {round_str} Window {window_num}")
                    continue

                new_bid_window = {
                    'id': bid_window_id,
                    'acad_term_id': acad_term_id,
                    'round': round_str,
                    'window': window_num
                }

                self.context.new_bid_windows.append(new_bid_window)
                self.context.bid_window_cache[window_key] = bid_window_id
                self.context.boss_stats['bid_windows_created'] = self.context.boss_stats.get('bid_windows_created', 0) + 1

                self._logger.info(f"Created bid_window {bid_window_id}: {acad_term_id} Round {round_str} Window {window_num}")
                bid_window_id += 1

        self.context.bid_window_id_counter = bid_window_id
        self._logger.info(f"Created {self.context.stats.get('bid_windows_created', 0)} bid windows")

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass