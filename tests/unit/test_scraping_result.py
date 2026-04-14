"""
Unit tests for ScrapingResult DTOs.
"""
import pytest
from datetime import datetime

from src.models.dto.scraping_result import ScrapingResult, ScraperError, ErrorType


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_error_types_exist(self):
        """ErrorType enum should have expected values."""
        assert ErrorType.STALE_ELEMENT.value == "stale_element"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.NAVIGATION.value == "navigation"
        assert ErrorType.PARSE.value == "parse"
        assert ErrorType.AUTHENTICATION.value == "authentication"
        assert ErrorType.UNKNOWN.value == "unknown"


class TestScraperError:
    """Tests for ScraperError."""

    def test_create_sets_timestamp(self):
        """ScraperError.create() should set timestamp to now."""
        error = ScraperError.create(
            url_attempted="https://example.com",
            error_type=ErrorType.TIMEOUT,
            message="Timeout occurred"
        )
        assert error.timestamp is not None
        assert isinstance(error.timestamp, datetime)

    def test_create_sets_url(self):
        """ScraperError.create() should set url_attempted."""
        error = ScraperError.create(
            url_attempted="https://boss.example.com",
            error_type=ErrorType.NAVIGATION,
            message="Navigation failed"
        )
        assert error.url_attempted == "https://boss.example.com"

    def test_create_sets_error_type_as_string(self):
        """ScraperError.create() should convert error_type to string value."""
        error = ScraperError.create(
            url_attempted="https://example.com",
            error_type=ErrorType.AUTHENTICATION,
            message="Auth failed"
        )
        assert error.error_type == "authentication"

    def test_create_with_screenshot_path(self):
        """ScraperError.create() should accept optional screenshot_path."""
        error = ScraperError.create(
            url_attempted="https://example.com",
            error_type=ErrorType.PARSE,
            message="Parse error",
            screenshot_path="/path/to/screenshot.png"
        )
        assert error.screenshot_path == "/path/to/screenshot.png"


class TestScrapingResult:
    """Tests for ScrapingResult."""

    def test_default_values(self):
        """ScrapingResult should have correct defaults."""
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1")
        assert result.files_saved == 0
        assert result.errors == []
        assert result.stopped_early is False
        assert result.stop_reason is None

    def test_is_success_true_when_files_saved_and_no_errors(self):
        """is_success should be True when files_saved > 0 and no errors."""
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1", files_saved=10)
        assert result.is_success is True

    def test_is_success_false_when_no_files_saved(self):
        """is_success should be False when files_saved is 0."""
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1", files_saved=0)
        assert result.is_success is False

    def test_is_success_false_when_has_errors(self):
        """is_success should be False when errors list is not empty."""
        error = ScraperError.create(
            url_attempted="https://example.com",
            error_type=ErrorType.TIMEOUT,
            message="Timeout"
        )
        result = ScrapingResult(
            ay_term="2025-26_T1",
            round_folder="R1W1",
            files_saved=10,
            errors=[error]
        )
        assert result.is_success is False

    def test_duration_seconds_calculates_correctly(self):
        """duration_seconds should calculate time between start and end."""
        start = datetime(2025, 1, 1, 10, 0, 0)
        end = datetime(2025, 1, 1, 10, 0, 10)
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1")
        result.start_time = start
        result.end_time = end
        assert result.duration_seconds == 10.0

    def test_add_error_appends_to_errors(self):
        """add_error should append error to errors list."""
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1")
        error = ScraperError.create(
            url_attempted="https://example.com",
            error_type=ErrorType.UNKNOWN,
            message="Unknown error"
        )
        result.add_error(error)
        assert len(result.errors) == 1
        assert result.errors[0] is error

    def test_finalize_sets_stopped_early_and_reason(self):
        """finalize should set stopped_early and stop_reason."""
        result = ScrapingResult(ay_term="2025-26_T1", round_folder="R1W1")
        result.finalize(stopped_early=True, stop_reason="300 empty records")
        assert result.stopped_early is True
        assert result.stop_reason == "300 empty records"
