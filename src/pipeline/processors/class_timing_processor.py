"""
ClassTimingProcessor - handles class timing CREATE logic.
Refactored to pure function pattern with explicit parameters.
"""
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.pipeline.dtos.timing_dto import ClassTimingDTO
from src.pipeline.dtos.class_dto import ClassDTO


class ClassTimingProcessor:
    """Processes class timing records from multiple data."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        class_lookup: Dict[Tuple, 'ClassDTO'],
        record_key_to_class_ids: Dict[str, List[str]] = None,
        existing_class_timing_keys: Set[Tuple] = None,
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._class_lookup = class_lookup
        self._record_key_to_class_ids = record_key_to_class_ids or {}
        self._existing_class_timing_keys = existing_class_timing_keys or set()
        self._logger = logger
        self._new_timings: List['ClassTimingDTO'] = []

    def process(self) -> List['ClassTimingDTO']:
        """Execute timing processing logic. Returns new timings."""
        self._process_all_rows()
        return self._new_timings

    def _process_all_rows(self) -> None:
        """Process all rows in raw_data."""
        self._logger.info("Processing class timings...")

        for _, row in self._raw_data.iterrows():
            timing_type = row.get('type', 'CLASS')
            if timing_type != 'CLASS':
                continue

            record_key = row.get('record_key')
            class_ids = self._find_class_ids(record_key)

            for class_id in class_ids:
                self._process_class_timing(row, class_id)

        self._logger.info(f"Created {len(self._new_timings)} new class timings (after deduplication).")

    def _find_class_ids(self, record_key: str) -> List[str]:
        """Find class IDs for a record_key using the pre-built mapping.

        The mapping is built during class processing in PipelineCoordinator.
        This is the correct, efficient lookup - O(1) instead of O(n).
        """
        if not record_key or pd.isna(record_key):
            return []

        return self._record_key_to_class_ids.get(record_key, [])

    def _process_class_timing(self, row: dict, class_id: str) -> None:
        """Process a single class timing record."""
        timing_key = (
            class_id,
            '' if pd.isna(row.get('day_of_week')) else str(row.get('day_of_week')),
            '' if pd.isna(row.get('start_time')) else str(row.get('start_time')),
            '' if pd.isna(row.get('end_time')) else str(row.get('end_time')),
            '' if pd.isna(row.get('venue')) else str(row.get('venue'))
        )

        if timing_key in self._existing_class_timing_keys:
            return

        timing_dto = ClassTimingDTO.from_row(row, class_id)
        self._new_timings.append(timing_dto)

    