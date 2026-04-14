"""
Unit tests for schedule_resolver utilities.
"""
import pytest
from datetime import datetime

from src.utils.schedule_resolver import get_bidding_round_info_for_term


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
