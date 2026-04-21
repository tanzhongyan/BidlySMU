"""
Bidding window parsing and schedule utilities.

Provides bidding round info based on current time and academic term.
Refactored from schedule_resolver.py - focuses on parsing and schedule resolution.
"""
import re
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd


ScheduleItem = Tuple[datetime, str, str]
T = TypeVar('T')


def _first_future_item(
    schedule: Sequence[ScheduleItem],
    now: datetime,
    mapper: Callable[[ScheduleItem], T],
    fallback: T,
) -> T:
    """
    Return mapper(item) for the first schedule item where now < results_date,
    or `fallback` if no item matches.
    """
    for results_date, *rest in schedule:
        if now < results_date:
            return mapper((results_date, *rest))
    return fallback


def acad_term_id_to_dash(acad_term_id: str) -> str:
    """Convert ACAD_TERM_ID format to BOSS schedule key format."""
    start_year = acad_term_id[2:6]
    end_year = acad_term_id[6:8]
    term = acad_term_id[8:]
    return f"{start_year}-{end_year}_{term}"


def get_bidding_round_info_for_term(
    ay_term: str,
    now: datetime,
    bidding_schedule: dict,
) -> Optional[str]:
    """
    Determines the bidding round folder name for a given academic term based on the current time.
    """
    schedule = bidding_schedule.get(ay_term)
    if not schedule:
        return None
    suffix = _first_future_item(schedule, now, lambda item: item[2], None)
    return f"{ay_term}_{suffix}" if suffix else None


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
    return _first_future_item(
        bidding_schedule_for_term,
        now,
        lambda item: item[1],
        bidding_schedule_for_term[-1][1],
    )


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


class BidSchedule:
    """Convenience wrapper for schedule-based operations."""

    def __init__(self, schedule_for_term: Sequence[ScheduleItem]):
        self._schedule = schedule_for_term

    def current_live_window_name(self, now: datetime) -> Optional[str]:
        return get_current_live_window_name(self._schedule, now)

    def processing_range_to_current(self, now: datetime) -> List[str]:
        return get_processing_range_to_current(self._schedule, now)


# ---------------------------------------------------------------------------
# Bidding window text parsing — registry pattern (OCP-compliant)
# ---------------------------------------------------------------------------

class _BiddingWindowFormat:
    """A single bidding window parsing format with regex pattern and mapper."""

    def __init__(self, pattern: str, mapper: Callable[[Any], Tuple[str, int]]):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._mapper = mapper

    def try_parse(self, text: str) -> Optional[Tuple[str, int]]:
        m = self._pattern.search(text)
        return self._mapper(m) if m else None


_FORMATS: List[_BiddingWindowFormat] = [
    _BiddingWindowFormat(
        r'Incoming\s+Freshmen\s+Rnd\s+(\w+)\s+Win\s+(\d+)',
        lambda m: ("1F" if m.group(1) == "1" else f"{m.group(1)}F", int(m.group(2))),
    ),
    _BiddingWindowFormat(
        r'Incoming\s+Exchange\s+Rnd\s+(\w+)\s+Win\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
    _BiddingWindowFormat(
        r'Round\s+(\d[A-C]?|\d+F?)\s+Window\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
]
_FORMATS_WITH_ABBREV = _FORMATS + [
    _BiddingWindowFormat(
        r'Rnd\s+(\d[A-C]?|\d+F?)\s+Win\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
]


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
    if bidding_window_str is None or (isinstance(bidding_window_str, float) and bidding_window_str != bidding_window_str):
        return default_round, default_window

    window_str = str(bidding_window_str).strip()
    if not window_str:
        return default_round, default_window

    for fmt in (_FORMATS_WITH_ABBREV if allow_abbrev else _FORMATS):
        result = fmt.try_parse(window_str)
        if result:
            return result

    if allow_generic_fallback:
        fallback_round_match = re.search(r'(\d[A-C]?|\d+F?)', window_str)
        if fallback_round_match:
            fallback_window_match = re.search(r'Window\s+(\d+)|Win\s+(\d+)', window_str, re.IGNORECASE)
            if fallback_window_match:
                window_num = int(fallback_window_match.group(1) or fallback_window_match.group(2))
                return fallback_round_match.group(1), window_num
            return fallback_round_match.group(1), 1

    return default_round, default_window
