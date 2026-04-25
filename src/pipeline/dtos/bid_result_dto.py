"""
BidResultDTO - Data Transfer Object for bid result records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class BidResultDTO:
    """DTO representing a bid result record."""

    COLUMNS = {
        'bid_window_id': 'bid_window_id',
        'class_id': 'class_id',
        'vacancy': 'vacancy',
        'opening_vacancy': 'opening_vacancy',
        'before_process_vacancy': 'before_process_vacancy',
        'dice': 'dice',
        'after_process_vacancy': 'after_process_vacancy',
        'enrolled_students': 'enrolled_students',
        'median': 'median',
        'min': 'min'
    }

    bid_window_id: int
    class_id: str
    vacancy: Optional[int] = None
    opening_vacancy: Optional[int] = None
    before_process_vacancy: Optional[int] = None
    dice: Optional[int] = None
    after_process_vacancy: Optional[int] = None
    enrolled_students: Optional[int] = None
    median: Optional[float] = None
    min: Optional[float] = None

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'bid_window_id': self.bid_window_id,
            'class_id': self.class_id,
            'vacancy': self.vacancy,
            'opening_vacancy': self.opening_vacancy,
            'before_process_vacancy': self.before_process_vacancy,
            'dice': self.dice,
            'after_process_vacancy': self.after_process_vacancy,
            'enrolled_students': self.enrolled_students,
            'median': self.median,
            'min': self.min
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {
            'bid_window_id': self.bid_window_id,
            'class_id': self.class_id,
            'vacancy': self.vacancy,
            'opening_vacancy': self.opening_vacancy,
            'before_process_vacancy': self.before_process_vacancy,
            'dice': self.dice,
            'after_process_vacancy': self.after_process_vacancy,
            'enrolled_students': self.enrolled_students,
            'median': self.median,
            'min': self.min
        }

    @staticmethod
    def from_row(
        row: dict,
        class_id: str,
        bid_window_id: int,
        vacancy: Optional[int] = None,
        opening_vacancy: Optional[int] = None,
        before_process_vacancy: Optional[int] = None,
        dice: Optional[int] = None,
        after_process_vacancy: Optional[int] = None,
        enrolled_students: Optional[int] = None,
        median: Optional[float] = None,
        min_bid: Optional[float] = None
    ) -> 'BidResultDTO':
        """Factory for CREATE."""
        return BidResultDTO(
            bid_window_id=bid_window_id,
            class_id=class_id,
            vacancy=vacancy,
            opening_vacancy=opening_vacancy,
            before_process_vacancy=before_process_vacancy,
            dice=dice,
            after_process_vacancy=after_process_vacancy,
            enrolled_students=enrolled_students,
            median=median,
            min=min_bid
        )