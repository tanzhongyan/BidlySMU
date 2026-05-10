# AcadTermDTO - Data Transfer Object for academic term records
from dataclasses import dataclass
from typing import Optional
from datetime import date


@dataclass
class AcadTermDTO:
    """DTO representing an academic term record."""

    # Column name mapping for DB and CSV (same format per Prisma schema)
    COLUMNS = {
        'id': 'id',
        'acad_year_start': 'acad_year_start',
        'acad_year_end': 'acad_year_end',
        'term': 'term',
        'boss_id': 'boss_id',
        'start_dt': 'start_dt',
        'end_dt': 'end_dt'
    }

    id: str  # e.g., 'AY202526T1'
    acad_year_start: int  # e.g., 2025
    acad_year_end: int  # e.g., 2026
    term: str  # e.g., '1' (no T prefix)
    boss_id: Optional[int]  # e.g., 1001
    start_dt: Optional[date]
    end_dt: Optional[date]

    def to_csv_row(self) -> dict:
        """Convert to CSV row."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}

    def to_db_row(self) -> dict:
        """Convert to database row."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}
