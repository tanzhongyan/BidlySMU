"""
Tests for centralized formatting functions in src/config.py.
"""
import pytest
from src.config import (
    abbrev_window_to_full,
    build_window_abbrev,
    strip_term_prefix,
    encode_subterm_for_boss_id,
    acad_term_id_to_short,
    ROUND_ORDER,
)


class TestAcadTermIdToShort:
    """Tests for acad_term_id_to_short (BOSS ClassDetails URL param)."""

    def test_term_1(self):
        assert acad_term_id_to_short('AY202627T1') == '2610'

    def test_term_2(self):
        assert acad_term_id_to_short('AY202526T2') == '2520'

    def test_term_3a(self):
        assert acad_term_id_to_short('AY202526T3A') == '2531'

    def test_term_3b(self):
        assert acad_term_id_to_short('AY202526T3B') == '2532'


class TestAbbrevWindowToFull:
    """Tests for abbrev_window_to_full."""

    def test_simple_window(self):
        assert abbrev_window_to_full('R1W1') == 'Round 1 Window 1'

    def test_window_with_subround(self):
        assert abbrev_window_to_full('R1AW2') == 'Round 1A Window 2'

    def test_freshmen_window(self):
        assert abbrev_window_to_full('R1FW4') == 'Round 1F Window 4'

    def test_window_with_c_round(self):
        assert abbrev_window_to_full('R1CW1') == 'Round 1C Window 1'

    def test_full_format_returns_unchanged(self):
        assert abbrev_window_to_full('Round 1 Window 1') == 'Round 1 Window 1'

    def test_invalid_input_returns_unchanged(self):
        assert abbrev_window_to_full('not-a-window') == 'not-a-window'

    def test_empty_string(self):
        assert abbrev_window_to_full('') == ''


class TestBuildWindowAbbrev:
    """Tests for build_window_abbrev."""

    def test_simple(self):
        assert build_window_abbrev('1', 1) == 'R1W1'

    def test_with_subround(self):
        assert build_window_abbrev('1A', 2) == 'R1AW2'

    def test_freshmen(self):
        assert build_window_abbrev('1F', 4) == 'R1FW4'

    def test_round_2(self):
        assert build_window_abbrev('2', 1) == 'R2W1'

    def test_round_2a(self):
        assert build_window_abbrev('2A', 3) == 'R2AW3'


class TestStripTermPrefix:
    """Tests for strip_term_prefix."""

    def test_with_t_prefix(self):
        assert strip_term_prefix('T3A') == '3A'

    def test_with_t1(self):
        assert strip_term_prefix('T1') == '1'

    def test_without_prefix(self):
        assert strip_term_prefix('3A') == '3A'

    def test_empty_string(self):
        assert strip_term_prefix('') == ''

    def test_single_t(self):
        assert strip_term_prefix('T') == ''


class TestEncodeSubtermForBossId:
    """Tests for encode_subterm_for_boss_id."""

    def test_term_3a(self):
        assert encode_subterm_for_boss_id('3A') == '1'

    def test_term_3b(self):
        assert encode_subterm_for_boss_id('3B') == '2'

    def test_term_1(self):
        assert encode_subterm_for_boss_id('1') == '0'

    def test_term_2(self):
        assert encode_subterm_for_boss_id('2') == '0'

    def test_term_with_t_prefix_3a(self):
        assert encode_subterm_for_boss_id('T3A') == '1'


class TestRoundOrder:
    """Tests for ROUND_ORDER constant."""

    def test_round_1_order(self):
        assert ROUND_ORDER['1'] == 1

    def test_round_1a_order(self):
        assert ROUND_ORDER['1A'] == 2

    def test_round_1b_order(self):
        assert ROUND_ORDER['1B'] == 3

    def test_round_1c_order(self):
        assert ROUND_ORDER['1C'] == 4

    def test_round_1f_order(self):
        assert ROUND_ORDER['1F'] == 5

    def test_round_2_order(self):
        assert ROUND_ORDER['2'] == 6

    def test_round_2a_order(self):
        assert ROUND_ORDER['2A'] == 7

    def test_sort_order(self):
        """Verify ROUND_ORDER produces correct sorting for bid windows."""
        rounds = ['2', '1A', '1F', '1', '2A', '1B', '1C']
        sorted_rounds = sorted(rounds, key=lambda r: ROUND_ORDER.get(r, 99))
        assert sorted_rounds == ['1', '1A', '1B', '1C', '1F', '2', '2A']
