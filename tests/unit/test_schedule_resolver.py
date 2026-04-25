"""
Unit tests for schedule_resolver utilities.
"""
from src.config import parse_bidding_window


class TestParseBiddingWindow:
    """Tests for parse_bidding_window()."""

    # ----- None and empty input tests -----

    def test_returns_defaults_for_none(self):
        """Should return defaults when input is None."""
        result = parse_bidding_window(None, default_round="R1", default_window=1)
        assert result == ("R1", 1)

    def test_returns_defaults_for_nan(self):
        """Should return defaults when input is NaN."""
        result = parse_bidding_window(float('nan'), default_round="R2", default_window=3)
        assert result == ("R2", 3)

    def test_returns_defaults_for_empty_string(self):
        """Should return defaults when input is empty string."""
        result = parse_bidding_window("", default_round="R3", default_window=5)
        assert result == ("R3", 5)

    def test_returns_defaults_for_whitespace_only(self):
        """Should return defaults when input is whitespace only."""
        result = parse_bidding_window("   ", default_round="R1", default_window=2)
        assert result == ("R1", 2)

    # ----- Full "Round X Window Y" format tests -----

    def test_parses_round_1_window_1(self):
        """Should parse 'Round 1 Window 1'."""
        result = parse_bidding_window("Round 1 Window 1")
        assert result == ("1", 1)

    def test_parses_round_1_window_2(self):
        """Should parse 'Round 1 Window 2'."""
        result = parse_bidding_window("Round 1 Window 2")
        assert result == ("1", 2)

    def test_parses_round_1a_window_2(self):
        """Should parse 'Round 1A Window 2'."""
        result = parse_bidding_window("Round 1A Window 2")
        assert result == ("1A", 2)

    def test_parses_round_1b_window_3(self):
        """Should parse 'Round 1B Window 3'."""
        result = parse_bidding_window("Round 1B Window 3")
        assert result == ("1B", 3)

    def test_parses_round_1c_window_4(self):
        """Should parse 'Round 1C Window 4'."""
        result = parse_bidding_window("Round 1C Window 4")
        assert result == ("1C", 4)

    def test_parses_round_2_window_1(self):
        """Should parse 'Round 2 Window 1'."""
        result = parse_bidding_window("Round 2 Window 1")
        assert result == ("2", 1)

    def test_parses_round_with_subletter_and_window(self):
        """Should parse 'Round 2A Window 5'."""
        result = parse_bidding_window("Round 2A Window 5")
        assert result == ("2A", 5)

    # ----- Incoming Freshmen format tests -----

    def test_parses_incoming_freshmen_round_1_window_4(self):
        """Should parse 'Incoming Freshmen Rnd 1 Win 4'."""
        result = parse_bidding_window("Incoming Freshmen Rnd 1 Win 4")
        assert result == ("1F", 4)

    def test_parses_incoming_freshmen_round_2_window_1(self):
        """Should parse 'Incoming Freshmen Rnd 2 Win 1'."""
        result = parse_bidding_window("Incoming Freshmen Rnd 2 Win 1")
        assert result == ("2F", 1)

    def test_incoming_freshmen_round_1_maps_to_1f(self):
        """Incoming Freshmen Round 1 should map to 1F."""
        result = parse_bidding_window("Incoming Freshmen Rnd 1 Win 2")
        assert result == ("1F", 2)

    # ----- Incoming Exchange format tests -----

    def test_parses_incoming_exchange(self):
        """Should parse 'Incoming Exchange Rnd 1C Win 1'."""
        result = parse_bidding_window("Incoming Exchange Rnd 1C Win 1")
        assert result == ("1C", 1)

    def test_parses_incoming_exchange_round_2_window_3(self):
        """Should parse 'Incoming Exchange Rnd 2 Win 3'."""
        result = parse_bidding_window("Incoming Exchange Rnd 2 Win 3")
        assert result == ("2", 3)

    # ----- Abbreviated format tests (Rnd X Win Y) -----

    def test_parses_abbreviated_rnd_win_format(self):
        """Should parse abbreviated 'Rnd 1A Win 2' format."""
        result = parse_bidding_window("Rnd 1A Win 2")
        assert result == ("1A", 2)

    def test_parses_abbreviated_round_win(self):
        """Should parse 'Rnd 1 Win 3'."""
        result = parse_bidding_window("Rnd 1 Win 3")
        assert result == ("1", 3)

    # ----- Case insensitivity tests -----

    def test_case_insensitive_round_window(self):
        """Should be case insensitive for 'Round X Window Y'."""
        result = parse_bidding_window("round 1 window 2")
        assert result == ("1", 2)

    def test_case_insensitive_abbreviated(self):
        """Should be case insensitive for abbreviated format."""
        result = parse_bidding_window("RND 1A WIN 3")
        assert result == ("1A", 3)

    def test_case_insensitive_incoming_freshmen(self):
        """Should be case insensitive for Incoming Freshmen."""
        result = parse_bidding_window("INCOMING FRESHMEN RND 1 WIN 4")
        assert result == ("1F", 4)

    # ----- No abbreviation allowed tests -----

    def test_returns_defaults_when_abbreviation_not_allowed(self):
        """Should return defaults when abbreviation not allowed and input is abbreviated."""
        result = parse_bidding_window("Rnd 1 Win 2", allow_abbrev=False)
        assert result == (None, None)

    def test_parses_full_format_when_abbreviation_not_allowed(self):
        """Should still parse full format even when abbreviation not allowed."""
        result = parse_bidding_window("Round 1 Window 2", allow_abbrev=False)
        assert result == ("1", 2)

    # ----- Generic fallback tests -----

    def test_generic_fallback_extracts_first_number_as_round(self):
        """Generic fallback finds first number as round, second as window."""
        result = parse_bidding_window("Window 5 Round 3", allow_generic_fallback=True)
        assert result == ("5", 5)  # First number (5) is round

    def test_generic_fallback_without_window_uses_default_1(self):
        """Should default window to 1 when only round found."""
        result = parse_bidding_window("Round 2", allow_generic_fallback=True)
        assert result == ("2", 1)

    def test_no_generic_fallback_returns_defaults(self):
        """Should return defaults when no pattern matches and no fallback."""
        result = parse_bidding_window("Something random", allow_generic_fallback=False)
        assert result == (None, None)

    # ----- Custom default values tests -----

    def test_custom_default_values(self):
        """Should return custom default values when no match."""
        result = parse_bidding_window("garbage", default_round="DEFAULT_R", default_window=99)
        assert result == ("DEFAULT_R", 99)

    # ----- With spaces around tests -----

    def test_handles_extra_spaces(self):
        """Should handle extra spaces in input."""
        result = parse_bidding_window("  Round   1A   Window   3  ")
        assert result == ("1A", 3)
