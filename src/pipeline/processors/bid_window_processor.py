"""
BidWindowProcessor - handles bid window CREATE logic.
Refactored to pure function pattern with DTO return.
"""
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.config import parse_bidding_window
from src.pipeline.dtos.bid_window_dto import BidWindowDTO


class BidWindowProcessor(AbstractProcessor):
    """Processes bid window records and returns DTOs."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        bid_window_cache: Dict[Tuple[str, str, int], int],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self._raw_data = raw_data
        self._bid_window_cache = bid_window_cache

    def process(self) -> Tuple[List[BidWindowDTO], List[BidWindowDTO]]:
        """Main entry point - returns (new_bid_windows, updated_bid_windows)."""
        self._logger.info("Processing bid windows...")

        # Track all unique bid windows found in data
        found_windows = defaultdict(set)  # acad_term_id -> set of (round, window) tuples

        # Optimize: drop NAs and duplicates FIRST to reduce iteration
        relevant_cols = self._raw_data[['acad_term_id', 'bidding_window']].dropna(
        ).drop_duplicates()

        # Discover all windows using itertuples (much faster than iterrows)
        for row in relevant_cols.itertuples(index=False):
            acad_term_id = row.acad_term_id
            bidding_window_str = row.bidding_window

            round_str, window_num = parse_bidding_window(bidding_window_str, allow_abbrev=True)

            if acad_term_id and round_str and window_num:
                found_windows[acad_term_id].add((round_str, window_num))

        # Determine starting ID for new windows
        max_id = 0
        for bid_window_entry in self._bid_window_cache.values():
            # bid_window_entry may be int (old format) or dict (new format)
            bid_window_id = bid_window_entry.get('id') if isinstance(bid_window_entry, dict) else bid_window_entry
            if isinstance(bid_window_id, int) and bid_window_id > max_id:
                max_id = bid_window_id
        next_bid_window_id = max_id + 1

        # Process each term's windows
        results_new = []
        for acad_term_id in sorted(found_windows.keys()):
            windows_for_term = found_windows[acad_term_id]
            sorted_windows = sorted(
                windows_for_term,
                key=lambda x: (BidWindowDTO.ROUND_ORDER.get(x[0], 99), x[1])
            )

            self._logger.info(f"Processing {acad_term_id}: found {len(sorted_windows)} windows")

            for round_str, window_num in sorted_windows:
                window_key = (acad_term_id, round_str, window_num)

                # Skip if already exists in cache
                if window_key in self._bid_window_cache:
                    self._logger.info(f"Bid window already exists: {acad_term_id} Round {round_str} Window {window_num}")
                    continue

                # Create new BidWindowDTO
                dto = BidWindowDTO(
                    id=next_bid_window_id,
                    acad_term_id=acad_term_id,
                    round=round_str,
                    window=window_num
                )
                results_new.append(dto)
                # Store as dict for consistency with _convert_caches_to_dicts format
                self._bid_window_cache[window_key] = {
                    'id': next_bid_window_id,
                    'acad_term_id': acad_term_id,
                    'round': round_str,
                    'window': window_num
                }

                self._logger.info(f"Created bid_window {next_bid_window_id}: {acad_term_id} Round {round_str} Window {window_num}")
                next_bid_window_id += 1

        self._logger.info(f"Created {len(results_new)} bid windows")
        return results_new, []  # Always empty updated list (bid_window only does CREATE)