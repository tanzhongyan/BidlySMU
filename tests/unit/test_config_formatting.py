"""
Tests for centralized formatting functions in src/config.py.
"""
import pytest
from src.config import (
    dash_format_to_acad_term_id,
    acad_term_id_to_dash_format,
    abbrev_window_to_full,
    build_window_abbrev,
    dash_format_to_display_format,
    strip_term_prefix,
    encode_subterm_for_boss_id,
    ROUND_ORDER,
)


class TestDashFormatToAcadTermId:
    """Tests for dash_format_to_acad_term_id."""

    def test_basic_conversion(self):
        assert dash_format_to_acad_term_id('2026-27_T1') == 'AY202627T1'

    def test_with_subterm(self):
        assert dash_format_to_acad_term_id('2025-26_T3A') == 'AY202526T3A'

    def test_with_term_b(self):
        assert dash_format_to_acad_term_id('2025-26_T3B') == 'AY202526T3B'

    def test_with_term_2(self):
        assert dash_format_to_acad_term_id('2025-26_T2') == 'AY202526T2'


class TestAcadTermIdToDashFormat:
    """Tests for acad_term_id_to_dash_format."""

    def test_basic_conversion(self):
        assert acad_term_id_to_dash_format('AY202627T1') == '2026-27_T1'

    def test_with_subterm(self):
        assert acad_term_id_to_dash_format('AY202526T3A') == '2025-26_T3A'

    def test_short_input_returns_unchanged(self):
        assert acad_term_id_to_dash_format('AY2026') == 'AY2026'

    def test_empty_input(self):
        assert acad_term_id_to_dash_format('') == ''

    def test_none_input(self):
        assert acad_term_id_to_dash_format(None) == None


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


class TestDashFormatToDisplayFormat:
    """Tests for dash_format_to_display_format."""

    def test_term_1(self):
        assert dash_format_to_display_format('2026-27_T1') == '2026-27 Term 1'

    def test_term_3a(self):
        assert dash_format_to_display_format('2025-26_T3A') == '2025-26 Term 3A'

    def test_term_3b(self):
        assert dash_format_to_display_format('2025-26_T3B') == '2025-26 Term 3B'

    def test_term_2(self):
        assert dash_format_to_display_format('2025-26_T2') == '2025-26 Term 2'


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
