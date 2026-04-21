"""
Abstract base class for all table processors.
Enforces standard process() interface across all processors.
"""
from abc import ABC, abstractmethod
from math import isnan
from typing import Any, List, Optional
import logging
import os
import pandas as pd


class AbstractProcessor(ABC):
    """Base class for processors with standard process() interface."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional injected logger.

        Args:
            logger: Optional logger instance. If not provided, a module-named
                   logger is created automatically (standard Python pattern).
        """
        self._logger = logger or logging.getLogger(self.__class__.__module__)

    @abstractmethod
    def process(self):  # type: (...) -> Any
        """Process and return results as DTOs. Must be implemented by subclasses."""
        pass

    @staticmethod
    def safe_int(val: Any) -> Optional[int]:
        """Safely convert to int. Returns None if val is NaN/None."""
        if val is None:
            return None
        if isinstance(val, float):
            try:
                if isnan(val):
                    return None
            except (TypeError, ValueError):
                pass
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def safe_float(val: Any) -> Optional[float]:
        """Safely convert to float. Returns None if val is NaN/None."""
        if val is None:
            return None
        if isinstance(val, float):
            try:
                if isnan(val):
                    return None
            except (TypeError, ValueError):
                pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def find_all_class_ids(
        self,
        acad_term_id,
        class_boss_id,
        new_classes: List[dict],
        existing_classes_cache: List[dict],
        output_base: Optional[str] = None,
    ) -> List[int]:
        """
        Finds all class_ids for a given acad_term_id and class_boss_id.
        Returns ALL class records for multi-professor classes.

        Args:
            acad_term_id: Academic term ID
            class_boss_id: Class boss ID
            new_classes: List of newly created class dictionaries
            existing_classes_cache: List of existing class dictionaries from cache
            output_base: Optional path to check for new_classes.csv file

        Returns:
            List of class IDs
        """
        if pd.isna(acad_term_id) or pd.isna(class_boss_id):
            return []

        found_class_ids = []

        if new_classes:
            for class_obj in new_classes:
                if (class_obj.get('acad_term_id') == acad_term_id and
                    str(class_obj.get('boss_id')) == str(class_boss_id)):
                    found_class_ids.append(class_obj['id'])

        if existing_classes_cache:
            for class_obj in existing_classes_cache:
                if (class_obj.get('acad_term_id') == acad_term_id and
                    str(class_obj.get('boss_id')) == str(class_boss_id)):
                    found_class_ids.append(class_obj['id'])

        if output_base:
            try:
                new_classes_path = os.path.join(output_base, 'new_classes.csv')
                if os.path.exists(new_classes_path):
                    df = pd.read_csv(new_classes_path)
                    matching_classes = df[
                        (df['acad_term_id'] == acad_term_id) &
                        (df['boss_id'].astype(str) == str(class_boss_id))
                    ]
                    for _, row in matching_classes.iterrows():
                        class_id = row['id']
                        if class_id not in found_class_ids:
                            found_class_ids.append(class_id)
            except Exception:
                pass

        unique_class_ids = []
        seen = set()
        for class_id in found_class_ids:
            if class_id not in seen:
                unique_class_ids.append(class_id)
                seen.add(class_id)

        return unique_class_ids
