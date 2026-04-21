"""
Unit tests for schedule_resolver utilities.
"""
from datetime import datetime

import pytest

from src.parser.bidding_window_parser import (
    get_bidding_round_info_for_term,
    get_current_live_window_name,
    get_processing_range_to_current,
    parse_bidding_window,
)


class TestGetBiddingRoundInfoForTerm:
    """Tests for get_bidding_round_info_for_term()."""

    def test_returns_none_when_term_not_in_schedule(self):
        """Should return None when ay_term not in bidding_schedule."""
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T9',
            now=datetime(2025, 8, 1),
            bidding_schedule={}
        )
        assert result is None

    def test_returns_none_when_all_dates_passed(self):
        """Should return None when now is after all schedule dates."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
                (datetime(2025, 7, 2, 14, 0), "Round 1 Window 2", "R1W2"),
            ]
        }
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 10),
            bidding_schedule=schedule
        )
        assert result is None

    def test_returns_next_round_folder(self):
        """Should return folder suffix for next upcoming round."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
                (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
                (datetime(2025, 7, 10, 14, 0), "Round 2 Window 1", "R2W1"),
            ]
        }
        # Now is between R1W1 (7/1) and R1W2 (7/5), so R1W2 is next
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 3),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T1_R1W2'

    def test_returns_first_round_when_before_all(self):
        """Should return first round when now is before any date."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 10, 14, 0), "Round 1 Window 1", "R1W1"),
                (datetime(2025, 7, 15, 14, 0), "Round 1 Window 2", "R1W2"),
            ]
        }
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 1),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T1_R1W1'

    def test_returns_none_when_now_equals_date(self):
        """Should return None when now equals a schedule date (edge case)."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 10, 14, 0), "Round 1 Window 1", "R1W1"),
            ]
        }
        # now is exactly at the schedule date
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 10, 14, 0),
            bidding_schedule=schedule
        )
        assert result is None


class TestGetCurrentLiveWindowName:
    def test_returns_first_future_window_name(self):
        schedule_for_term = [
            (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
            (datetime(2025, 7, 10, 14, 0), "Round 2 Window 1", "R2W1"),
        ]
        result = get_current_live_window_name(schedule_for_term, datetime(2025, 7, 3, 12, 0))
        assert result == "Round 1 Window 2"

    def test_returns_last_window_when_all_passed(self):
        schedule_for_term = [
            (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
        ]
        result = get_current_live_window_name(schedule_for_term, datetime(2025, 7, 6, 12, 0))
        assert result == "Round 1 Window 2"

    def test_returns_none_on_empty_schedule(self):
        assert get_current_live_window_name([], datetime(2025, 7, 1)) is None


class TestGetProcessingRangeToCurrent:
    def test_returns_range_up_to_current_window(self):
        schedule_for_term = [
            (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
            (datetime(2025, 7, 10, 14, 0), "Round 2 Window 1", "R2W1"),
        ]
        result = get_processing_range_to_current(schedule_for_term, datetime(2025, 7, 3, 12, 0))
        assert result == ["Round 1 Window 1", "Round 1 Window 2"]

    def test_returns_all_windows_when_all_passed(self):
        schedule_for_term = [
            (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
        ]
        result = get_processing_range_to_current(schedule_for_term, datetime(2025, 7, 6, 12, 0))
        assert result == ["Round 1 Window 1", "Round 1 Window 2"]

    def test_returns_empty_on_empty_schedule(self):
        assert get_processing_range_to_current([], datetime(2025, 7, 1)) == []


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


class TestGetBiddingRoundInfoForTermAdditional:
    """Additional tests for get_bidding_round_info_for_term()."""

    def test_returns_folder_suffix_with_correct_format(self):
        """Should return folder suffix in correct format 'ay_term_suffix'."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            ]
        }
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 6, 1),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T1_R1W1'

    def test_multiple_terms_uses_correct_term(self):
        """Should use the specified ay_term from schedule."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            ],
            '2025-26_T2': [
                (datetime(2025, 8, 1, 14, 0), "Round 1 Window 1", "R1W1"),
            ]
        }
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T2',
            now=datetime(2025, 7, 15),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T2_R1W1'

    def test_returns_none_for_empty_schedule(self):
        """Should return None when schedule is empty dict."""
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 1),
            bidding_schedule={}
        )
        assert result is None

    def test_returns_next_window_when_now_equals_first_result_date(self):
        """When now equals first date, returns next window's folder suffix."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 10, 14, 0), "Round 1 Window 1", "R1W1"),
                (datetime(2025, 7, 15, 14, 0), "Round 1 Window 2", "R1W2"),
            ]
        }
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 10, 14, 0),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T1_R1W2'

    def test_returns_next_window_when_between_two_windows(self):
        """Should return next window when now is between two windows."""
        schedule = {
            '2025-26_T1': [
                (datetime(2025, 7, 1, 14, 0), "Round 1 Window 1", "R1W1"),
                (datetime(2025, 7, 5, 14, 0), "Round 1 Window 2", "R1W2"),
                (datetime(2025, 7, 10, 14, 0), "Round 1 Window 3", "R1W3"),
            ]
        }
        # now is between R1W1 (7/1) and R1W2 (7/5)
        result = get_bidding_round_info_for_term(
            ay_term='2025-26_T1',
            now=datetime(2025, 7, 3),
            bidding_schedule=schedule
        )
        assert result == '2025-26_T1_R1W2'
