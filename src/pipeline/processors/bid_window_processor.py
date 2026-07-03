"""
BidWindowProcessor - handles bid window CREATE logic.
Refactored to pure function pattern with DTO return.
"""
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.processors.abstract_processor import AbstractProcessor
from src.config import parse_bidding_window, CURRENT_WINDOW_NAME, PREVIOUS_WINDOW_NAME, build_window_abbrev, acad_term_id_to_dash_format, ROUND_ORDER, SGT
from src.pipeline.dtos.bid_window_dto import BidWindowDTO, derive_timeline_from_results_at


class BidWindowProcessor(AbstractProcessor):
    """Processes bid window records and returns DTOs with computed timeline dates."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        bid_window_cache: Dict[Tuple[str, str, int], int],
        expected_acad_term_id: Optional[str] = None,
        bidding_schedules: Optional[Dict] = None,
        results_datetime: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self._raw_data = raw_data
        self._bid_window_cache = bid_window_cache
        self._expected_acad_term_id = expected_acad_term_id
        self._bidding_schedules = bidding_schedules or {}
        self._results_datetime = results_datetime  # From RESULTS_DATETIME env var (single window)

    def _compute_timeline(
        self,
        acad_term_id: str,
        round_str: str,
        window_num: int,
    ):
        """
        Compute opens_at, closes_at, and results_at for a bid window.

        Priority:
        1. If bidding_schedules.json entry has 5 elements (new extended format):
           [results_at, title, abbrev, opens_at, closes_at] → use dates directly
        2. If bidding_schedules.json entry has 3 elements (legacy format):
           [results_at, title, abbrev] → derive opens_at/closes_at from results_at
        3. If RESULTS_DATETIME env var is set for this exact window → use it + derive
        4. Otherwise → leave as None
        """
        results_at = None
        opens_at = None
        closes_at = None

        # Look up in bidding_schedules.json (matches by abbrev like "R1AW2")
        abbrev = build_window_abbrev(round_str, window_num)
        dash_term = acad_term_id_to_dash_format(acad_term_id)
        schedule = self._bidding_schedules.get(dash_term, [])
        for entry in schedule:
            if entry[2] == abbrev:
                # Check if entry has extended format (5 elements: results, title, abbrev, opens, closes)
                if len(entry) >= 5 and entry[3] and entry[4]:
                    results_at = entry[0]
                    opens_at = entry[3]
                    closes_at = entry[4]
                else:
                    # Legacy format: derive from results_at
                    results_at = entry[0]
                    opens_at, closes_at = derive_timeline_from_results_at(results_at)
                break

        # RESULTS_DATETIME env var overrides results_at for the current run's window,
        # but preserves opens_at/closes_at from the schedule if already resolved.
        if self._results_datetime:
            current_round, current_window = None, None
            if CURRENT_WINDOW_NAME:
                current_round, current_window = parse_bidding_window(
                    CURRENT_WINDOW_NAME, allow_abbrev=True
                )
            if (current_round == round_str and current_window == window_num):
                try:
                    results_at = datetime.fromisoformat(self._results_datetime)
                    # Only derive opens/closes if schedule didn't provide them
                    if opens_at is None or closes_at is None:
                        opens_at, closes_at = derive_timeline_from_results_at(results_at)
                except (ValueError, TypeError):
                    pass

        # Convert string dates to datetime if needed, then ensure SGT-aware
        # bidding_schedules.json stores wall-clock times in SGT (naive or UTC).
        # PostgreSQL TIMESTAMPTZ requires timezone-aware datetimes — naive
        # datetimes are interpreted as UTC, causing an 8-hour offset.
        # SGT is imported from src.config (single source of truth).

        if isinstance(results_at, str):
            results_at = datetime.fromisoformat(results_at)
        if results_at is not None and results_at.tzinfo is None:
            results_at = results_at.replace(tzinfo=SGT)

        if isinstance(opens_at, str):
            opens_at = datetime.fromisoformat(opens_at)
        if opens_at is not None and opens_at.tzinfo is None:
            opens_at = opens_at.replace(tzinfo=SGT)

        if isinstance(closes_at, str):
            closes_at = datetime.fromisoformat(closes_at)
        if closes_at is not None and closes_at.tzinfo is None:
            closes_at = closes_at.replace(tzinfo=SGT)

        if results_at is None:
            return None, None, None

        return opens_at, closes_at, results_at

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

        # Add explicit window names if defined
        if self._expected_acad_term_id:
            for explicit_window in filter(None, [CURRENT_WINDOW_NAME, PREVIOUS_WINDOW_NAME]):
                round_str, window_num = parse_bidding_window(explicit_window, allow_abbrev=True)
                if round_str and window_num:
                    found_windows[self._expected_acad_term_id].add((round_str, window_num))

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
                key=lambda x: (ROUND_ORDER.get(x[0], 99), x[1])
            )

            self._logger.info(f"Processing {acad_term_id}: found {len(sorted_windows)} windows")

            for round_str, window_num in sorted_windows:
                window_key = (acad_term_id, round_str, window_num)

                # Compute timeline dates for this window
                opens_at, closes_at, results_at = self._compute_timeline(
                    acad_term_id, round_str, window_num
                )

                # Skip if already exists in cache — but still UPDATE if it lacks dates
                if window_key in self._bid_window_cache:
                    existing = self._bid_window_cache[window_key]
                    existing_opens = existing.get('opens_at') if isinstance(existing, dict) else None
                    existing_closes = existing.get('closes_at') if isinstance(existing, dict) else None
                    if not existing_opens and opens_at:
                        self._logger.info(
                            f"Bid window {acad_term_id} Round {round_str} Window {window_num} "
                            f"exists but missing opens_at — will be updated"
                        )
                        existing['opens_at'] = opens_at
                        existing['closes_at'] = closes_at
                        existing['results_at'] = results_at
                    else:
                        self._logger.info(
                            f"Bid window already exists: {acad_term_id} Round {round_str} Window {window_num}"
                        )
                    continue

                # Create new BidWindowDTO with timeline dates
                dto = BidWindowDTO(
                    id=next_bid_window_id,
                    acad_term_id=acad_term_id,
                    round=round_str,
                    window=window_num,
                    opens_at=opens_at,
                    closes_at=closes_at,
                    results_at=results_at,
                )
                results_new.append(dto)
                # Store as dict for consistency with _convert_caches_to_dicts format
                self._bid_window_cache[window_key] = {
                    'id': next_bid_window_id,
                    'acad_term_id': acad_term_id,
                    'round': round_str,
                    'window': window_num,
                    'opens_at': opens_at,
                    'closes_at': closes_at,
                    'results_at': results_at,
                }

                self._logger.info(
                    f"Created bid_window {next_bid_window_id}: {acad_term_id} "
                    f"Round {round_str} Window {window_num}"
                    + (f" opens={opens_at.isoformat()}" if opens_at else "")
                )
                next_bid_window_id += 1

        self._logger.info(f"Created {len(results_new)} bid windows")
        return results_new, []  # Always empty updated list (bid_window only does CREATE)