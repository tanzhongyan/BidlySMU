"""
ClassAvailabilityDTO - Data Transfer Object for class availability records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassAvailabilityDTO:
    """DTO representing a class availability record."""

    COLUMNS = {
        'class_id': 'class_id',
        'bid_window_id': 'bid_window_id',
        'total': 'total',
        'current_enrolled': 'current_enrolled',
        'reserved': 'reserved',
        'available': 'available'
    }

    class_id: str
    bid_window_id: int
    total: int = 0
    current_enrolled: int = 0
    reserved: int = 0
    available: int = 0

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'class_id': self.class_id,
            'bid_window_id': self.bid_window_id,
            'total': self.total,
            'current_enrolled': self.current_enrolled,
            'reserved': self.reserved,
            'available': self.available
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {
            'class_id': self.class_id,
            'bid_window_id': self.bid_window_id,
            'total': self.total,
            'current_enrolled': self.current_enrolled,
            'reserved': self.reserved,
            'available': self.available
        }

    @staticmethod
    def from_row(row: dict, class_id: str, bid_window_id: int) -> 'ClassAvailabilityDTO':
        """Factory for CREATE."""
        import pandas as pd

        def safe_int(val, default=0):
            if pd.isna(val):
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        return ClassAvailabilityDTO(
            class_id=class_id,
            bid_window_id=bid_window_id,
            total=safe_int(row.get('total')),
            current_enrolled=safe_int(row.get('current_enrolled')),
            reserved=safe_int(row.get('reserved')),
            available=safe_int(row.get('available'))
        )