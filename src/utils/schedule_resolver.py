"""
Schedule resolution utilities for BOSS bidding rounds.

Provides bidding round info based on current time and academic term.
"""
from datetime import datetime
from typing import Optional


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
