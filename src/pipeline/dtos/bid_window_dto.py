# BidWindowDTO - Data Transfer Object for bid window records
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.config import ROUND_ORDER, SGT


def derive_timeline_from_results_at(
    results_at: Optional[datetime],
) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Derive opensAt and closesAt from resultsAt using SMU BOSS business rules.

    - closesAt = same calendar day as resultsAt, 10:00 SGT
    - opensAt  = 2 calendar days before closesAt, 10:00 SGT

    This is a heuristic fallback used only when Trumba calendar data is
    unavailable. Actual schedules vary (back-to-back windows open at 17:00,
    Incoming Freshmen can open 4 days before close).

    Returns (opens_at, closes_at) as timezone-aware datetimes or (None, None).
    """
    if results_at is None:
        return None, None

    # Ensure results_at is SGT-aware
    if results_at.tzinfo is None:
        results_sgt = results_at.replace(tzinfo=SGT)
    else:
        results_sgt = results_at.astimezone(SGT)

    # closesAt: same calendar day, 10:00 SGT
    closes_sgt = datetime(
        results_sgt.year, results_sgt.month, results_sgt.day, 10, 0, 0, tzinfo=SGT
    )

    # opensAt: 2 calendar days before close, 10:00 SGT
    # (updated to match current SMU BOSS schedule from Trumba;
    #  note: back-to-back windows open at 17:00 same day — this
    #  heuristic is a last-resort fallback when Trumba is unavailable)
    open_date = closes_sgt - timedelta(days=2)
    opens_sgt = datetime(
        open_date.year, open_date.month, open_date.day, 10, 0, 0, tzinfo=SGT
    )

    return opens_sgt, closes_sgt


@dataclass
class BidWindowDTO:
    """DTO representing a bid window record."""

    COLUMNS = {
        'id': 'id',
        'acad_term_id': 'acad_term_id',
        'round': 'round',
        'window': 'window',
        'opens_at': 'opens_at',
        'closes_at': 'closes_at',
        'results_at': 'results_at',
    }

    # Class constant for round ordering (imported from centralized config)
    ROUND_ORDER = ROUND_ORDER

    id: int  # Note: int, not string (matches current implementation)
    acad_term_id: str  # BOSS format: 'AY202526T1'
    round: str  # '1', '1A', '1B', '1C', '1F', '2', '2A'
    window: int  # 1, 2, 3, etc.
    opens_at: Optional[datetime] = None  # Window opens (bidding starts) in SGT
    closes_at: Optional[datetime] = None  # Window closes (bidding ends) in SGT
    results_at: Optional[datetime] = None  # Results release datetime in SGT

    @classmethod
    def from_dict(cls, item: dict) -> 'BidWindowDTO':
        """Create a BidWindowDTO from a dictionary (e.g., from database cache)."""
        return cls(
            id=int(item.get('id', 0)),
            acad_term_id=str(item.get('acad_term_id', '')),
            round=str(item.get('round', '')),
            window=int(item.get('window', 0)),
            opens_at=item.get('opens_at'),
            closes_at=item.get('closes_at'),
            results_at=item.get('results_at'),
        )

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}