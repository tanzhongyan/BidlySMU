"""
ClassExamTimingProcessor - handles class exam timing CREATE logic.
Refactored to pure function pattern with explicit parameters.
"""
from typing import Dict, List, Optional, Set
import pandas as pd

from src.pipeline.dtos.timing_dto import ClassExamTimingDTO
from src.pipeline.dtos.class_dto import ClassDTO


class ClassExamTimingProcessor:
    """Processes class exam timing records from multiple data."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        class_lookup: Dict[tuple, 'ClassDTO'],
        record_key_to_class_ids: Dict[str, List[str]] = None,
        processed_exam_class_ids: Set[str] = None,
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._class_lookup = class_lookup
        self._record_key_to_class_ids = record_key_to_class_ids or {}
        self._processed_exam_class_ids = processed_exam_class_ids or set()
        self._logger = logger
        self._new_exam_timings: List['ClassExamTimingDTO'] = []

    def process(self) -> List['ClassExamTimingDTO']:
        """Execute exam timing processing logic. Returns new exam timings."""
        self._process_all_rows()
        return self._new_exam_timings

    def _process_all_rows(self) -> None:
        """Process all rows in raw_data."""
        self._logger.info("Processing class exam timings...")

        for _, row in self._raw_data.iterrows():
            timing_type = row.get('type', 'CLASS')
            if timing_type != 'EXAM':
                continue

            record_key = row.get('record_key')
            class_ids = self._find_class_ids(record_key)

            for class_id in class_ids:
                self._process_exam_timing(row, class_id)

        self._logger.info(f"Created {len(self._new_exam_timings)} new exam timings (after deduplication).")

    def _find_class_ids(self, record_key: str) -> List[str]:
        """Find class IDs for a record_key using the pre-built mapping.

        The mapping is built during class processing in PipelineCoordinator.
        This is the correct, efficient lookup - O(1) instead of O(n).
        """
        if not record_key or pd.isna(record_key):
            return []

        return self._record_key_to_class_ids.get(record_key, [])

    def _process_exam_timing(self, row: dict, class_id: str) -> None:
        """Process a single exam timing record."""
        if class_id in self._processed_exam_class_ids:
            return

        exam_dto = ClassExamTimingDTO.from_row(row, class_id)
        self._new_exam_timings.append(exam_dto)

    