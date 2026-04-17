"""
Schedule resolution utilities for BOSS bidding rounds.

Provides bidding round info based on current time and academic term.
"""
import re
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import pandas as pd


ScheduleItem = Tuple[datetime, str, str]


def get_bidding_round_info_for_term(
    ay_term: str,
    now: datetime,
    bidding_schedule: dict,
) -> Optional[str]:
    """
    Determines the bidding round folder name for a given academic term based on the current time.

    Args:
        ay_term: Academic year term (e.g., '2024-25_T1').
        now: Current datetime to check against schedule.
        bidding_schedule: Bidding schedule dictionary. REQUIRED - no default.

    Returns:
        Folder name suffix if in a bidding window, None otherwise.
        Format: "{ay_term}_{folder_suffix}"
    """
    schedule = bidding_schedule.get(ay_term)
    if not schedule:
        return None

    for results_date, _, folder_suffix in schedule:
        if now < results_date:
            return f"{ay_term}_{folder_suffix}"

    return None


def get_current_live_window_name(
    bidding_schedule_for_term: Sequence[ScheduleItem],
    now: datetime,
) -> Optional[str]:
    """
    Return the current live window name for a term schedule.

    Behavior mirrors existing pipeline logic:
    - Select the first future window name.
    - If none are future, return the last window name.
    """
    if not bidding_schedule_for_term:
        return None

    for results_date, window_name, _ in bidding_schedule_for_term:
        if now < results_date:
            return window_name

    return bidding_schedule_for_term[-1][1]


def get_processing_range_to_current(
    bidding_schedule_for_term: Sequence[ScheduleItem],
    now: datetime,
) -> List[str]:
    """
    Return the list of window names from start of schedule to current live window (inclusive).
    """
    if not bidding_schedule_for_term:
        return []

    current_live_window = get_current_live_window_name(bidding_schedule_for_term, now)
    if not current_live_window:
        return []

    processing_range: List[str] = []
    for _, window_name, _ in bidding_schedule_for_term:
        processing_range.append(window_name)
        if window_name == current_live_window:
            break

    return processing_range


def parse_bidding_window(
    bidding_window_str: str,
    *,
    allow_abbrev: bool = True,
    allow_generic_fallback: bool = False,
    default_round: Optional[str] = None,
    default_window: Optional[int] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse bidding window text into (round, window).

    Supported formats include:
    - Round 1 Window 1
    - Round 1A Window 2
    - Incoming Exchange Rnd 1C Win 1
    - Incoming Freshmen Rnd 1 Win 4
    - Rnd 1A Win 2
    """
    if bidding_window_str is None or (isinstance(bidding_window_str, float) and pd.isna(bidding_window_str)):
        return default_round, default_window

    window_str = str(bidding_window_str).strip()
    if not window_str:
        return default_round, default_window

    freshmen_match = re.search(r'Incoming\s+Freshmen\s+Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
    if freshmen_match:
        raw_round = freshmen_match.group(1)
        mapped_round = "1F" if raw_round == "1" else f"{raw_round}F"
        return mapped_round, int(freshmen_match.group(2))

    exchange_match = re.search(r'Incoming\s+Exchange\s+Rnd\s+(\w+)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
    if exchange_match:
        return exchange_match.group(1), int(exchange_match.group(2))

    round_window_match = re.search(r'Round\s+(\d[A-C]?|\d+F?)\s+Window\s+(\d+)', window_str, re.IGNORECASE)
    if round_window_match:
        return round_window_match.group(1), int(round_window_match.group(2))

    if allow_abbrev:
        abbrev_match = re.search(r'Rnd\s+(\d[A-C]?|\d+F?)\s+Win\s+(\d+)', window_str, re.IGNORECASE)
        if abbrev_match:
            return abbrev_match.group(1), int(abbrev_match.group(2))

    if allow_generic_fallback:
        fallback_round_match = re.search(r'(\d[A-C]?|\d+F?)', window_str)
        if fallback_round_match:
            fallback_window_match = re.search(r'Window\s+(\d+)|Win\s+(\d+)', window_str, re.IGNORECASE)
            if fallback_window_match:
                window_num = int(fallback_window_match.group(1) or fallback_window_match.group(2))
                return fallback_round_match.group(1), window_num
            return fallback_round_match.group(1), 1

    return default_round, default_window


def parse_window_name(window_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a window display name into (round, window).

    Kept as a semantic alias for existing predictor callsites.
    """
    return parse_bidding_window(window_name, allow_abbrev=True, default_round=None, default_window=None)
