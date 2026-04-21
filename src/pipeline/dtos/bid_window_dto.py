# BidWindowDTO - Data Transfer Object for bid window records
from dataclasses import dataclass


@dataclass
class BidWindowDTO:
    """DTO representing a bid window record."""

    COLUMNS = {
        'id': 'id',
        'acad_term_id': 'acad_term_id',
        'round': 'round',
        'window': 'window'
    }

    # Class constant for round ordering (extracted from hardcoded dict)
    ROUND_ORDER = {'1': 1, '1A': 2, '1B': 3, '1C': 4, '1F': 5, '2': 6, '2A': 7}

    id: int  # Note: int, not string (matches current implementation)
    acad_term_id: str  # BOSS format: 'AY202526T1'
    round: str  # '1', '1A', '1B', '1C', '1F', '2', '2A'
    window: int  # 1, 2, 3, etc.

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}