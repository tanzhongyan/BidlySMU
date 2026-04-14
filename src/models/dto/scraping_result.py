"""
DTOs for scraping operations and results.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class ErrorType(Enum):
    """Types of scraping errors."""
    STALE_ELEMENT = "stale_element"
    TIMEOUT = "timeout"
    NAVIGATION = "navigation"
    PARSE = "parse"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScraperError:
    """
    Immutable DTO representing a scraping error with metadata for debugging.

    Usage:
        error = ScraperError(
            timestamp=datetime.now(),
            url_attempted="https://boss.intranet.smu.edu.sg/ClassDetails.aspx",
            screenshot_path="/path/to/screenshot.png",
            error_type=ErrorType.TIMEOUT,
            message="Element not found within 30 seconds"
        )
    """
    timestamp: datetime
    url_attempted: str
    screenshot_path: Optional[str]  # Crucial for debugging SMU portal changes remotely
    error_type: str
    message: str

    @classmethod
    def create(
        cls,
        url_attempted: str,
        error_type: ErrorType,
        message: str,
        screenshot_path: Optional[str] = None,
    ) -> "ScraperError":
        """Factory method to create ScraperError with current timestamp."""
        return cls(
            timestamp=datetime.now(),
            url_attempted=url_attempted,
            screenshot_path=screenshot_path,
            error_type=error_type.value,
            message=message,
        )


@dataclass
class ScrapingResult:
    """
    Result of a scraping operation with error tracking.

    Usage:
        result = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=150,
            errors=[error1, error2],
            stopped_early=True,
            stop_reason="300 consecutive empty records"
        )
    """
    ay_term: str
    round_folder: str
    files_saved: int = 0
    errors: List[ScraperError] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.files_saved > 0 and len(self.errors) == 0

    @property
    def duration_seconds(self) -> float:
        """Calculate duration of scraping operation."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def add_error(self, error: ScraperError) -> None:
        """Add an error to the result."""
        self.errors.append(error)

    def finalize(self, stopped_early: bool = False, stop_reason: Optional[str] = None) -> None:
        """Mark operation as complete."""
        self.stopped_early = stopped_early
        self.stop_reason = stop_reason
        self.end_time = datetime.now()