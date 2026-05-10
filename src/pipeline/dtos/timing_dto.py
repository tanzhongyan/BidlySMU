"""
TimingDTO - Data Transfer Objects for class and exam timing records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass
class ClassTimingDTO:
    """DTO representing a class timing record."""

    COLUMNS = {
        'id': 'id',
        'class_id': 'class_id',
        'start_date': 'start_date',
        'end_date': 'end_date',
        'day_of_week': 'day_of_week',
        'start_time': 'start_time',
        'end_time': 'end_time',
        'venue': 'venue'
    }

    id: str
    class_id: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    day_of_week: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    venue: Optional[str] = None

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'id': self.id,
            'class_id': self.class_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'day_of_week': self.day_of_week,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'venue': self.venue
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        # Note: id is excluded - database auto-generates via sequence
        return {
            'class_id': self.class_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'day_of_week': self.day_of_week,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'venue': self.venue
        }

    @staticmethod
    def from_row(row: dict, class_id: str) -> 'ClassTimingDTO':
        """Factory for CREATE - generates UUID."""
        import pandas as pd

        def safe_val(val, default=None):
            if pd.isna(val):
                return default
            return val

        return ClassTimingDTO(
            id=str(uuid.uuid4()),
            class_id=class_id,
            start_date=safe_val(row.get('start_date')),
            end_date=safe_val(row.get('end_date')),
            day_of_week=safe_val(row.get('day_of_week')),
            start_time=safe_val(row.get('start_time')),
            end_time=safe_val(row.get('end_time')),
            venue=safe_val(row.get('venue'), '')
        )


@dataclass
class ClassExamTimingDTO:
    """DTO representing a class exam timing record."""

    COLUMNS = {
        'id': 'id',
        'class_id': 'class_id',
        'date': 'date',
        'day_of_week': 'day_of_week',
        'start_time': 'start_time',
        'end_time': 'end_time',
        'venue': 'venue'
    }

    id: str
    class_id: str
    date: Optional[str] = None
    day_of_week: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    venue: Optional[str] = None

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'id': self.id,
            'class_id': self.class_id,
            'date': self.date,
            'day_of_week': self.day_of_week,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'venue': self.venue
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        # Note: id is excluded - database auto-generates via sequence
        return {
            'class_id': self.class_id,
            'date': self.date,
            'day_of_week': self.day_of_week,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'venue': self.venue
        }

    @staticmethod
    def from_row(row: dict, class_id: str) -> 'ClassExamTimingDTO':
        """Factory for CREATE - generates UUID."""
        import pandas as pd

        def safe_val(val, default=None):
            if pd.isna(val):
                return default
            return val

        return ClassExamTimingDTO(
            id=str(uuid.uuid4()),
            class_id=class_id,
            date=safe_val(row.get('date')),
            day_of_week=safe_val(row.get('day_of_week')),
            start_time=str(safe_val(row.get('start_time'), '')),
            end_time=str(safe_val(row.get('end_time'), '')),
            venue=safe_val(row.get('venue'))
        )