"""
Unit tests for parse_bidding_window() - centralized bidding window text parser.

This function is used by BidWindowProcessor, BidResultProcessor, and
BidPredictionProcessor. It converts text like "Round 1 Window 1" into
(round, window) tuples.
"""
import pytest
import math

from src.config import parse_bidding_window


# ============================================================================
# Standard Formats
# ============================================================================

class TestStandardFormats:
    """Tests for standard bidding window text formats."""

    @pytest.mark.parametrize("text,expected_round,expected_window", [
        ("Round 1 Window 1", "1", 1),
        ("Round 1 Window 2", "1", 2),
        ("Round 2 Window 1", "2", 1),
        ("Round 2 Window 3", "2", 3),
        ("Round 1A Window 1", "1A", 1),
        ("Round 1A Window 2", "1A", 2),
        ("Round 1B Window 1", "1B", 1),
        ("Round 1C Window 1", "1C", 1),
        ("Round 1F Window 1", "1F", 1),
        ("Round 2A Window 1", "2A", 1),
    ])
    def test_standard_round_window(self, text, expected_round, expected_window):
        """Standard 'Round X Window Y' format should parse correctly."""
        round_val, window_val = parse_bidding_window(text)
        assert round_val == expected_round
        assert window_val == expected_window


# ============================================================================
# Abbreviated Formats
# ============================================================================

class TestAbbreviatedFormats:
    """Tests for abbreviated bidding window text formats."""

    @pytest.mark.parametrize("text,expected_round,expected_window", [
        ("Rnd 1 Win 1", "1", 1),
        ("Rnd 1A Win 2", "1A", 2),
        ("Rnd 2 Win 3", "2", 3),
        ("Rnd 1C Win 1", "1C", 1),
    ])
    def test_abbreviated_round_window(self, text, expected_round, expected_window):
        """Abbreviated 'Rnd X Win Y' format should parse correctly."""
        round_val, window_val = parse_bidding_window(text)
        assert round_val == expected_round
        assert window_val == expected_window

    def test_abbreviated_disabled(self):
        """With allow_abbrev=False, abbreviated format should not match."""
        round_val, window_val = parse_bidding_window(
            "Rnd 1 Win 1",
            allow_abbrev=False
        )
        assert round_val is None
        assert window_val is None


# ============================================================================
# Incoming Freshmen / Exchange Formats
# ============================================================================

class TestIncomingFormats:
    """Tests for Incoming Freshmen and Exchange window formats."""

    @pytest.mark.parametrize("text,expected_round,expected_window", [
        ("Incoming Freshmen Rnd 1 Win 4", "1F", 4),
        ("Incoming Freshmen Rnd 1 Win 1", "1F", 1),
        ("Incoming Exchange Rnd 1 Win 1", "1", 1),
        ("Incoming Exchange Rnd 1C Win 1", "1C", 1),
        ("Incoming Exchange Rnd 2 Win 3", "2", 3),
    ])
    def test_incoming_formats(self, text, expected_round, expected_window):
        """Incoming Freshmen/Exchange formats should parse correctly."""
        round_val, window_val = parse_bidding_window(text)
        assert round_val == expected_round
        assert window_val == expected_window

    def test_incoming_freshmen_round2_not_f(self):
        """Incoming Freshmen with round != 1 should still append F suffix.

        Per the implementation: if m.group(1) == "1" → "1F", else f"{group}F"
        """
        round_val, window_val = parse_bidding_window("Incoming Freshmen Rnd 2 Win 1")
        assert round_val == "2F"
        assert window_val == 1


# ============================================================================
# Generic Fallback
# ============================================================================

class TestGenericFallback:
    """Tests for generic fallback parsing mode."""

    def test_generic_fallback_enabled(self):
        """With allow_generic_fallback=True, partial matches should work."""
        round_val, window_val = parse_bidding_window(
            "Some 1A Window 5 text",
            allow_generic_fallback=True
        )
        assert round_val == "1A"
        assert window_val == 5

    def test_generic_fallback_no_window(self):
        """Generic fallback without 'Window' should default window to 1."""
        round_val, window_val = parse_bidding_window(
            "Some 2A text",
            allow_generic_fallback=True
        )
        assert round_val == "2A"
        assert window_val == 1

    def test_generic_fallback_disabled(self):
        """With allow_generic_fallback=False, unparseable text should return defaults."""
        round_val, window_val = parse_bidding_window(
            "Some 1A Window 5 text",
            allow_generic_fallback=False
        )
        assert round_val is None
        assert window_val is None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and invalid inputs."""

    def test_none_input(self):
        """None input should return default values."""
        round_val, window_val = parse_bidding_window(None)
        assert round_val is None
        assert window_val is None

    def test_nan_input(self):
        """NaN float input should return default values."""
        round_val, window_val = parse_bidding_window(float('nan'))
        assert round_val is None
        assert window_val is None

    def test_empty_string(self):
        """Empty string should return default values."""
        round_val, window_val = parse_bidding_window("")
        assert round_val is None
        assert window_val is None

    def test_whitespace_only(self):
        """Whitespace-only string should return default values."""
        round_val, window_val = parse_bidding_window("   ")
        assert round_val is None
        assert window_val is None

    def test_unparseable_text(self):
        """Completely unparseable text should return default values."""
        round_val, window_val = parse_bidding_window("xyz")
        assert round_val is None
        assert window_val is None

    def test_custom_defaults(self):
        """Custom default_round and default_window should be used when parsing fails."""
        round_val, window_val = parse_bidding_window(
            "xyz",
            default_round="2",
            default_window=5
        )
        assert round_val == "2"
        assert window_val == 5

    def test_case_insensitive(self):
        """Parsing should be case-insensitive."""
        round_val, window_val = parse_bidding_window("round 1 window 1")
        assert round_val == "1"
        assert window_val == 1

    def test_incoming_freshmen_case_insensitive(self):
        """Incoming Freshmen parsing should be case-insensitive."""
        round_val, window_val = parse_bidding_window("incoming freshmen rnd 1 win 4")
        assert round_val == "1F"
        assert window_val == 4

    def test_numeric_only_round(self):
        """Numeric-only round should work correctly."""
        round_val, window_val = parse_bidding_window("Round 1 Window 1")
        assert round_val == "1"
        assert isinstance(round_val, str)

    def test_window_is_integer(self):
        """Window number should be an integer."""
        round_val, window_val = parse_bidding_window("Round 1 Window 5")
        assert isinstance(window_val, int)
        assert window_val == 5

    def test_standard_format_takes_priority_over_abbrev(self):
        """Standard 'Round X Window Y' should match before abbreviated format."""
        round_val, window_val = parse_bidding_window("Round 1 Window 1")
        assert round_val == "1"
        assert window_val == 1

    def test_incoming_freshmen_takes_priority_over_round(self):
        """'Incoming Freshmen Rnd X Win Y' should match before 'Rnd X Win Y'."""
        round_val, window_val = parse_bidding_window("Incoming Freshmen Rnd 1 Win 4")
        assert round_val == "1F"
        assert window_val == 4


# ============================================================================
# BidWindowFormat Internal Class
# ============================================================================

class TestBiddingWindowFormat:
    """Tests for the _BiddingWindowFormat internal class."""

    def test_format_try_parse_returns_none_on_no_match(self):
        """try_parse should return None when pattern doesn't match."""
        from src.config import _BiddingWindowFormat
        fmt = _BiddingWindowFormat(
            r'Round\s+(\d+)\s+Window\s+(\d+)',
            lambda m: (m.group(1), int(m.group(2)))
        )
        result = fmt.try_parse("No match here")
        assert result is None

    def test_format_try_parse_returns_tuple_on_match(self):
        """try_parse should return (round, window) on match."""
        from src.config import _BiddingWindowFormat
        fmt = _BiddingWindowFormat(
            r'Round\s+(\d+)\s+Window\s+(\d+)',
            lambda m: (m.group(1), int(m.group(2)))
        )
        result = fmt.try_parse("Round 3 Window 7")
        assert result == ("3", 7)
