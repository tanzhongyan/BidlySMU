"""
Unit tests for shared bidding window parsing helpers.
"""

from src.parser.bidding_window_parser import parse_bidding_window


class TestParseBiddingWindow:
    def test_parses_round_window_format(self):
        assert parse_bidding_window("Round 1A Window 2") == ("1A", 2)

    def test_parses_exchange_abbrev_format(self):
        assert parse_bidding_window("Incoming Exchange Rnd 1C Win 1") == ("1C", 1)

    def test_maps_freshmen_round(self):
        assert parse_bidding_window("Incoming Freshmen Rnd 1 Win 4") == ("1F", 4)

    def test_uses_defaults_when_unrecognized(self):
        assert parse_bidding_window("Unknown Window", default_round="1", default_window=1) == ("1", 1)

    def test_uses_generic_fallback_when_enabled(self):
        assert parse_bidding_window(
            "Bid Phase 2A Win 3",
            allow_generic_fallback=True,
            default_round="1",
            default_window=1,
        ) == ("2A", 3)
