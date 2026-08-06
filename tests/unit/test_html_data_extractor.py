"""
Unit tests for HTMLDataExtractor.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.scraper.html_data_extractor import HTMLDataExtractor, ExtractionResult


class TestHTMLDataExtractor:
    """Tests for HTMLDataExtractor."""

    def test_requires_no_config(self):
        """HTMLDataExtractor should not require config."""
        extractor = HTMLDataExtractor()
        assert extractor is not None

    def test_initializes_with_driver(self, mock_webdriver):
        """HTMLDataExtractor should accept driver in constructor."""
        extractor = HTMLDataExtractor(driver=mock_webdriver)
        assert extractor._driver is mock_webdriver


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self):
        """ExtractionResult should have correct defaults."""
        result = ExtractionResult()
        assert result.files_processed == 0
        assert result.files_successful == 0
        assert result.errors == []
        assert result.standalone_records == []
        assert result.multiple_records == []

    def test_tracks_files_processed(self):
        """ExtractionResult should track files_processed."""
        result = ExtractionResult(files_processed=10)
        assert result.files_processed == 10

    def test_tracks_files_successful(self):
        """ExtractionResult should track files_successful."""
        result = ExtractionResult(files_successful=8)
        assert result.files_successful == 8

    def test_tracks_errors(self):
        """ExtractionResult should track errors list."""
        error = {"file": "test.html", "error": "Parse failed"}
        result = ExtractionResult(errors=[error])
        assert len(result.errors) == 1
        assert result.errors[0]["file"] == "test.html"


class TestHTMLDataExtractorBiddingWindow:
    """Tests for HTMLDataExtractor bidding window extraction.

    BIDDING_SCHEDULES is pinned to a known value so these tests are
    deterministic — src.config now loads it from Supabase Storage at import.
    """

    @pytest.fixture(autouse=True)
    def _pin_bidding_schedules(self):
        """Pin BIDDING_SCHEDULES to avoid network/Supabase dependency."""
        with patch('src.config.BIDDING_SCHEDULES', {'AY202526T3A': []}):
            yield

    def test_extract_r2w1_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W1."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("AY202526T1_R2W1")
        assert result == 'Round 2 Window 1'

    def test_extract_r2w2_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W2."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("AY202526T1_R2W2")
        assert result == 'Round 2 Window 2'

    def test_extract_r2w3_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W3."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("AY202526T1_R2W3")
        assert result == 'Round 2 Window 3'

    def test_extract_r2w4_from_folder(self):
        """_extract_bidding_window_from_folder should handle R2W4."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("AY202526T1_R2W4")
        assert result == 'Round 2 Window 4'

    def test_extract_abbrev_only_folder(self):
        """_extract_bidding_window_from_folder should handle abbrev-only 'R2W1' (new convention)."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("R2W1")
        assert result == 'Round 2 Window 1'

    def test_extract_unknown_returns_original(self):
        """_extract_bidding_window_from_folder should return original for unknown."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder("UNKNOWN")
        assert result == "UNKNOWN"

    def test_legacy_prefixed_folder_still_resolves(self):
        """Legacy '2026-27_T1_R1FW4' folders still resolve to the clean full name."""
        extractor = HTMLDataExtractor()
        result = extractor._extract_bidding_window_from_folder('2026-27_T1_R1FW4')
        # The suffix after the last "_" is the abbrev (R1FW4) -> 'Round 1F Window 4'.
        assert result == 'Round 1F Window 4'


class TestHTMLDataExtractorRun:
    """Tests for HTMLDataExtractor.run() method."""

    def test_run_returns_bool(self, mock_webdriver):
        """scrape() should return a boolean."""
        extractor = HTMLDataExtractor(driver=mock_webdriver)

        with patch.object(extractor, '_process_all_files'):
            with patch.object(extractor, '_save_to_excel'):
                result = extractor.scrape(output_path='test.xlsx')
                assert isinstance(result, bool)

    def test_run_sets_up_driver_if_none(self, mock_webdriver):
        """scrape() should set up driver if _driver is None."""
        extractor = HTMLDataExtractor()
        assert extractor._driver is None

        with patch('src.scraper.html_data_extractor.ChromeDriverFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory.create.return_value = mock_webdriver
            mock_factory_class.return_value = mock_factory

            with patch.object(extractor, '_process_all_files'):
                with patch.object(extractor, '_save_to_excel'):
                    extractor.scrape(output_path='test.xlsx')
                    mock_factory.create.assert_called_once()


class TestCleanEncoding:
    """Tests for HTMLDataExtractor._clean_encoding mojibake handling.

    Mojibake sequences must be CONVERTED to the intended character, never
    deleted (the old `re.sub(r'â€[^\\w]', '', ...)` dropped apostrophes and
    quotes from course names).
    """

    def test_converts_apostrophe_mojibake(self):
        """â€™ (U+2019 apostrophe) should become a straight apostrophe."""
        assert HTMLDataExtractor._clean_encoding("DON\u00e2\u20ac\u2122T") == "DON'T"

    def test_converts_double_quote_mojibake(self):
        """â€œ/â€" should become double quotes, not be deleted."""
        assert HTMLDataExtractor._clean_encoding(
            "\u00e2\u20ac\u0153quoted\u00e2\u20ac\u0022") == '"quoted"'

    def test_converts_latin1_control_mojibake(self):
        """latin-1 control-char mojibake (â + 0x80 + 0x9D) should decode to a quote."""
        assert HTMLDataExtractor._clean_encoding("\u00e2\x80\x9d") == "\u201d"

    def test_converts_dash_mojibake(self):
        """â€“/â€” should become en/em dashes."""
        assert HTMLDataExtractor._clean_encoding("a\u00e2\u20ac\u201cb") == "a\u2013b"
        assert HTMLDataExtractor._clean_encoding("a\u00e2\u20ac\u201db") == "a\u2014b"

    def test_keeps_plain_text_unchanged(self):
        """Plain text should pass through unchanged (no lossy decode)."""
        assert HTMLDataExtractor._clean_encoding("  BOSS 101 ") == "BOSS 101"
        assert HTMLDataExtractor._clean_encoding("caf\u00e9") == "caf\u00e9"


class TestParseAcadTerm:
    """Tests for HTMLDataExtractor._parse_acad_term (config-helper refactor)."""

    def test_parses_term_3a(self):
        """Should parse '2025-26 Term 3A' into components + acad_term_id."""
        extractor = HTMLDataExtractor()
        result = extractor._parse_acad_term("2025-26 Term 3A")
        assert result == (2025, 2026, 'T3A', 'AY202526T3A')

    def test_parses_term_1(self):
        """Should parse '2025-26 Term 1'."""
        extractor = HTMLDataExtractor()
        result = extractor._parse_acad_term("2025-26 Term 1 - Regular Academic Session")
        assert result == (2025, 2026, 'T1', 'AY202526T1')

    def test_parses_term_3b(self):
        """Should parse '2024-25 Term 3B'."""
        extractor = HTMLDataExtractor()
        result = extractor._parse_acad_term("2024-25 Term 3B")
        assert result == (2024, 2025, 'T3B', 'AY202425T3B')

    def test_returns_none_for_unparseable(self):
        """Should return all-None for unparseable text."""
        extractor = HTMLDataExtractor()
        result = extractor._parse_acad_term("garbage")
        assert result == (None, None, None, None)

    def test_convert_date_to_timestamp(self):
        """Should keep the DB format 'YYYY-MM-DD 00:00:00.000 +0800'."""
        extractor = HTMLDataExtractor()
        assert extractor._convert_date_to_timestamp("09-Jul-2025") == "2025-07-09 00:00:00.000 +0800"
        assert extractor._convert_date_to_timestamp("not a date") is None
