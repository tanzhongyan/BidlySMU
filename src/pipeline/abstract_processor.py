"""
Abstract base class for all table processors.
Enforces standard process() interface across all processors.
"""
from abc import ABC, abstractmethod
from typing import Optional
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
    def process(self):
        """Process and return results as DTOs."""
        pass