"""
Abstract base class for all table processors.
Enforces standard process() interface across all processors.
"""
from abc import ABC, abstractmethod
from math import isnan
from typing import Any, Optional
import logging


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
