"""
Abstract base class for all table processors.
Enforces standard process() interface across all processors.
"""
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """Base class for processors with standard process() interface."""

    @abstractmethod
    def process(self):
        """Process and return results as DTOs."""
        pass