"""
ClassAvailabilityProcessor - handles class availability CREATE logic.
Refactored to pure function pattern with explicit parameters.
Processes ONLY current window t(N) from raw_data.xlsx.
"""
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.config import CURRENT_WINDOW_NAME, parse_bidding_window
from src.pipeline.dtos.class_availability_dto import ClassAvailabilityDTO
from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.bid_window_dto import BidWindowDTO


class ClassAvailabilityProcessor:
    """Processes class availability records from standalone data."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        class_lookup: Dict[Tuple, 'ClassDTO'],
        bid_window_lookup: Dict[Tuple, 'BidWindowDTO'],
        existing_availability_keys: Set[Tuple] = None,
        bidding_schedule: List[Tuple] = None,
        expected_acad_term_id: str = None,
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._class_lookup = class_lookup
        self._bid_window_lookup = bid_window_lookup
        self._existing_availability_keys = existing_availability_keys or set()
        self._bidding_schedule = bidding_schedule or []
        self._expected_acad_term_id = expected_acad_term_id
        self._logger = logger
        self._new_availability: List['ClassAvailabilityDTO'] = []

    def process(self) -> List['ClassAvailabilityDTO']:
        """Execute class availability processing logic. Returns new availability records."""
        self._process_all_rows()
        return self._new_availability

    def _process_all_rows(self) -> None:
        """Process all rows in raw_data for current window only."""
        current_window_name = CURRENT_WINDOW_NAME
        self._logger.info(f"Processing class availability for current window: '{current_window_name}'")

        if not current_window_name:
            self._logger.info("No current window found - skipping availability processing")
            return

        current_window_data = self._filter_to_current_window(current_window_name)
        self._logger.info(f"Filtered to {len(current_window_data)} records for current window")

        self._process_window_data(current_window_data)

    def _filter_to_current_window(self, current_window_name: str) -> pd.DataFrame:
        """Filter raw_data to only records for the current window and term."""
        if self._raw_data.empty:
            return self._raw_data

        if 'bidding_window' not in self._raw_data.columns:
            return self._raw_data

        original_count = len(self._raw_data)
        filtered = self._raw_data[self._raw_data['bidding_window'] == current_window_name].copy()

        if 'acad_term_id' in filtered.columns and self._expected_acad_term_id:
            before_term_filter = len(filtered)
            filtered = filtered[filtered['acad_term_id'] == self._expected_acad_term_id]
            self._logger.info(f"Filtered data: {original_count} -> {before_term_filter} (window) -> {len(filtered)} (window + term)")

        return filtered

    def _process_window_data(self, window_data: pd.DataFrame) -> None:
        """Process all rows in the filtered window data."""
        newly_created_count = 0

        for _, row in window_data.iterrows():
            course_code = row.get('course_code')
            section = row.get('section')
            acad_term_id = row.get('acad_term_id')
            bidding_window_str = row.get('bidding_window')

            if pd.isna(course_code) or pd.isna(section) or pd.isna(acad_term_id) or pd.isna(bidding_window_str):
                continue

            round_str, window_num = parse_bidding_window(bidding_window_str, allow_abbrev=True)
            if not all([round_str, window_num]):
                continue

            class_boss_id = row.get('class_boss_id')
            class_ids = self._find_all_class_ids(acad_term_id, class_boss_id)

            if not class_ids:
                continue

            window_key = (acad_term_id, round_str, window_num)
            bid_window_dto = self._bid_window_lookup.get(window_key)
            if not bid_window_dto:
                continue

            total_val = int(row.get('total')) if pd.notna(row.get('total')) else 0
            current_enrolled_val = int(row.get('current_enrolled')) if pd.notna(row.get('current_enrolled')) else 0
            reserved_val = int(row.get('reserved')) if pd.notna(row.get('reserved')) else 0
            available_val = int(row.get('available')) if pd.notna(row.get('available')) else 0

            for class_id in class_ids:
                availability_key = (class_id, bid_window_dto.id)

                if availability_key in self._existing_availability_keys:
                    continue

                availability_dto = ClassAvailabilityDTO(
                    class_id=class_id,
                    bid_window_id=bid_window_dto.id,
                    total=total_val,
                    current_enrolled=current_enrolled_val,
                    reserved=reserved_val,
                    available=available_val
                )
                self._new_availability.append(availability_dto)
                newly_created_count += 1

        self._logger.info(f"Class availability checks complete. Created {newly_created_count} new records.")

    def _find_all_class_ids(self, acad_term_id: str, class_boss_id) -> List[str]:
        """Find all class IDs for a given acad_term_id and boss_id."""
        class_ids = []
        for (term_id, boss_id, professor_id), class_dto in self._class_lookup.items():
            if term_id == acad_term_id and boss_id == class_boss_id:
                class_ids.append(class_dto.id)
        return class_ids

    