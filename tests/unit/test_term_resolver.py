"""
Unit tests for term_resolver utilities.
"""
import pytest

from src.utils.term_resolver import (
    get_term_code_map,
    get_all_terms,
    transform_term_format,
    convert_target_term_format,
    generate_academic_year_range,
    TERM_CODE_MAP,
    ALL_TERMS,
)


class TestTermCodeMap:
    """Tests for TERM_CODE_MAP constant."""

    def test_term_code_map_values(self):
        """TERM_CODE_MAP should have correct values."""
        assert TERM_CODE_MAP == {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}


class TestGetTermCodeMap:
    """Tests for get_term_code_map()."""

    def test_returns_copy(self):
        """get_term_code_map() should return a copy, not the original."""
        result = get_term_code_map()
        result['T1'] = '99'
        assert TERM_CODE_MAP['T1'] == '10'  # Original unchanged

    def test_returns_expected_mapping(self):
        """get_term_code_map() should return correct mapping."""
        result = get_term_code_map()
        assert result == {'T1': '10', 'T2': '20', 'T3A': '31', 'T3B': '32'}


class TestGetAllTerms:
    """Tests for get_all_terms()."""

    def test_returns_copy(self):
        """get_all_terms() should return a copy, not the original."""
        result = get_all_terms()
        result.append('T99')
        assert ALL_TERMS == ['T1', 'T2', 'T3A', 'T3B']  # Original unchanged

    def test_returns_all_terms(self):
        """get_all_terms() should return all term abbreviations."""
        result = get_all_terms()
        assert result == ['T1', 'T2', 'T3A', 'T3B']


class TestTransformTermFormat:
    """Tests for transform_term_format()."""

    def test_transforms_t1(self):
        """transform_term_format should transform T1 correctly."""
        result = transform_term_format('2025-26_T1')
        assert result == '2025-26 Term 1'

    def test_transforms_t2(self):
        """transform_term_format should transform T2 correctly."""
        result = transform_term_format('2024-25_T2')
        assert result == '2024-25 Term 2'

    def test_transforms_t3a(self):
        """transform_term_format should transform T3A correctly."""
        result = transform_term_format('2025-26_T3A')
        assert result == '2025-26 Term 3A'

    def test_transforms_t3b(self):
        """transform_term_format should transform T3B correctly."""
        result = transform_term_format('2025-26_T3B')
        assert result == '2025-26 Term 3B'

    def test_raises_on_invalid_term(self):
        """transform_term_format should raise ValueError on unknown term."""
        with pytest.raises(ValueError):
            transform_term_format('2025-26_T9')

    def test_raises_on_invalid_format(self):
        """transform_term_format should raise ValueError on invalid format."""
        with pytest.raises(ValueError, match="Invalid term format"):
            transform_term_format('2025-26')


class TestGenerateAcademicYearRange:
    """Tests for generate_academic_year_range()."""

    def test_single_term(self):
        """generate_academic_year_range should work for single term."""
        result = generate_academic_year_range('2025-26_T1', '2025-26_T1')
        assert result == ['2025-26_T1']

    def test_same_year_multiple_terms(self):
        """generate_academic_year_range should return all terms in same year."""
        result = generate_academic_year_range('2025-26_T1', '2025-26_T2')
        assert result == ['2025-26_T1', '2025-26_T2']

    def test_crosses_academic_year(self):
        """generate_academic_year_range should handle crossing academic years."""
        result = generate_academic_year_range('2025-26_T3B', '2026-27_T1')
        expected = ['2025-26_T3B', '2026-27_T1']
        assert result == expected

    def test_raises_on_invalid_start_term(self):
        """generate_academic_year_range should raise ValueError on invalid start."""
        with pytest.raises(ValueError):
            generate_academic_year_range('invalid', '2025-26_T1')

    def test_raises_on_invalid_end_term(self):
        """generate_academic_year_range should raise ValueError on invalid end."""
        with pytest.raises(ValueError):
            generate_academic_year_range('2025-26_T1', 'invalid')


class TestConvertTargetTermFormat:
    """Tests for convert_target_term_format()."""

    def test_converts_term_1(self):
        assert convert_target_term_format('2025-26_T1') == 'AY202526T1'

    def test_converts_term_3a(self):
        assert convert_target_term_format('2025-26_T3A') == 'AY202526T3A'

    def test_keeps_unrecognized_format(self):
        assert convert_target_term_format('AY202526T1') == 'AY202526T1'

    def test_keeps_empty_value(self):
        assert convert_target_term_format('') == ''
