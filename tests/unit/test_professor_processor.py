"""
Unit tests for ProfessorProcessor (refactored DTO pattern).
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import json

from src.pipeline.processors.professor_processor import (
    ProfessorProcessor, ProfessorDTO
)


class TestParseBossAliases:
    """Tests for parse_boss_aliases helper function."""

    def test_handles_none(self):
        assert ProfessorProcessor.parse_boss_aliases(None) == []

    def test_handles_empty_list(self):
        assert ProfessorProcessor.parse_boss_aliases([]) == []

    def test_handles_json_string(self):
        result = ProfessorProcessor.parse_boss_aliases('["JOHN SMITH", "JANE DOE"]')
        assert "JOHN SMITH" in result
        assert "JANE DOE" in result

    def test_handles_postgres_array_string(self):
        result = ProfessorProcessor.parse_boss_aliases('{"JOHN SMITH", "JANE DOE"}')
        assert "JOHN SMITH" in result
        assert "JANE DOE" in result

    def test_handles_single_string(self):
        result = ProfessorProcessor.parse_boss_aliases("JOHN SMITH")
        assert "JOHN SMITH" in result


class TestProfessorDTO:
    """Tests for ProfessorDTO."""

    def test_to_csv_row(self):
        dto = ProfessorDTO(
            id="test-uuid",
            name="John Smith",
            email="john@smu.edu.sg",
            slug="john-smith",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            belong_to_university=1,
            boss_aliases=["JOHN SMITH", "JOHN"],
            original_scraped_name="Dr. John Smith",
            updated_at=None
        )

        row = dto.to_csv_row()
        assert row['id'] == "test-uuid"
        assert row['name'] == "John Smith"
        assert row['email'] == "john@smu.edu.sg"
        assert row['slug'] == "john-smith"
        assert row['belong_to_university'] == 1
        assert '"JOHN SMITH"' in row['boss_aliases']  # JSON string
        assert row['original_scraped_name'] == "Dr. John Smith"
        assert 'updated_at' not in row  # None means not included

    def test_to_csv_row_with_updated_at(self):
        dto = ProfessorDTO(
            id="test-uuid",
            name="John Smith",
            email="john@smu.edu.sg",
            slug="john-smith",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["JOHN SMITH"],
            updated_at="2026-04-20T10:00:00Z"
        )

        row = dto.to_csv_row()
        assert row['updated_at'] == "2026-04-20T10:00:00Z"

    def test_from_row_creates_uuid(self):
        row = {
            'name': 'Jane Doe',
            'slug': 'jane-doe',
            'boss_aliases': '["JANE DOE"]',
            'original_scraped_name': 'Dr. Jane'
        }

        dto = ProfessorDTO.from_row(row)

        assert dto.id is not None  # UUID generated
        assert dto.name == 'Jane Doe'
        assert dto.email == 'enquiry@smu.edu.sg'  # Default
        assert dto.slug == 'jane-doe'
        assert "JANE DOE" in dto.boss_aliases
        assert dto.original_scraped_name == 'Dr. Jane'
        assert dto.updated_at is None  # CREATE mode


class TestExtractUniqueProfessors:
    """Tests for _extract_unique_professors method."""

    def test_extracts_simple_names(self):
        raw_data = pd.DataFrame({
            'professor_name': ['John Smith', 'Jane Doe', 'John Smith']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={}
        )

        unique_profs, variations = processor._extract_unique_professors()

        assert 'John Smith' in unique_profs
        assert 'Jane Doe' in unique_profs
        assert len(unique_profs) == 2  # Deduplicated

    def test_extracts_comma_separated_names(self):
        raw_data = pd.DataFrame({
            'professor_name': ['LIM CHONG BOON DENNIS, PhD', 'DENNIS LIM']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={}
        )

        unique_profs, variations = processor._extract_unique_professors()

        # Should split comma-separated names
        assert 'LIM CHONG BOON DENNIS' in unique_profs
        assert 'PhD' in unique_profs  # The extension after comma

    def test_skips_empty_and_tba(self):
        raw_data = pd.DataFrame({
            'professor_name': ['Valid Professor', None, '', 'TBA', 'TO BE ANNOUNCED', 'nan']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={}
        )

        unique_profs, variations = processor._extract_unique_professors()

        assert 'Valid Professor' in unique_profs
        assert len(unique_profs) == 1

    def test_tracks_variations(self):
        raw_data = pd.DataFrame({
            'professor_name': ['YUESHEN, BART ZHOU', 'BART ZHOU']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={}
        )

        unique_profs, variations = processor._extract_unique_professors()

        # Variations should track that YUESHEN, BART ZHOU maps to BART ZHOU
        assert 'YUESHEN, BART ZHOU' in unique_profs
        assert 'BART ZHOU' in unique_profs


class TestNormalizationFallback:
    """Tests for _normalize_professor_name_fallback method."""

    def test_handles_none(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        boss_name, afterclass_name = processor._normalize_professor_name_fallback(None)
        assert boss_name == "UNKNOWN"
        assert afterclass_name == "Unknown"

    def test_handles_asian_surname_format(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        boss_name, afterclass_name = processor._normalize_professor_name_fallback("LIM, CHONG BOON DENNIS")

        assert "LIM" in boss_name
        assert "CHONG" in afterclass_name

    def test_handles_western_given_name(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        # "DAVID LEE" - LEE is Asian surname, DAVID is Western given name
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("DAVID LEE")

        assert boss_name == "DAVID LEE"
        assert "LEE" in afterclass_name

    def test_removes_middle_initials(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        boss_name, afterclass_name = processor._normalize_professor_name_fallback("JOHN B. SMITH")

        # Should remove middle initial B.
        assert "B" not in boss_name.split()
        assert "SMITH" in afterclass_name


class TestResolutionChain:
    """Tests for the 8-strategy resolution chain."""

    def test_strategy1_direct_lookup(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        processor._professor_lookup["JOHN SMITH"] = {"database_id": "uuid-123", "boss_name": "JOHN SMITH", "afterclass_name": "John Smith"}

        result = processor._resolve_single_professor("John Smith", "JOHN SMITH", "John Smith")

        assert result == 'matched'

    def test_strategy2_boss_name_lookup(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        processor._professor_lookup["JOHN SMITH"] = {"database_id": "uuid-123", "boss_name": "JOHN SMITH", "afterclass_name": "John Smith"}

        # Looking for "SMITH, JOHN" but boss_name "JOHN SMITH" is in lookup
        result = processor._resolve_single_professor("Smith, John", "JOHN SMITH", "John Smith")

        assert result == 'matched'

    def test_strategy3_cache_direct_match(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={"JOHN SMITH": {"id": "uuid-456", "name": "John Smith"}}
        )

        result = processor._resolve_single_professor("John Smith", "JOHN SMITH", "John Smith")

        assert result == 'matched'

    def test_strategy8_subset_matching(self):
        """Strategy 7: Subset matching - 'WARREN CHIK' matches 'KAM WAI WARREN CHIK'."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        # Full name in lookup
        processor._professor_lookup["KAM WAI WARREN CHIK"] = {
            "database_id": "uuid-123",
            "boss_name": "KAM WAI WARREN CHIK",
            "afterclass_name": "Kam Wai Warren Chik"
        }

        # Subset name being resolved
        result = processor._resolve_single_professor("Warren Chik", "WARREN CHIK", "Warren Chik")

        assert result == 'matched'
        assert processor._professor_lookup["WARREN CHIK"]["database_id"] == "uuid-123"

    def test_strategy8_requires_min_2_words(self):
        """Strategy 7: Requires at least 2 words to avoid false positives."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        # Single word should NOT trigger subset matching
        processor._professor_lookup["JOHN SMITH"] = {
            "database_id": "uuid-123",
            "boss_name": "JOHN SMITH",
            "afterclass_name": "John Smith"
        }

        result = processor._resolve_single_professor("John", "JOHN", "John")

        # Single word should not match via subset (would return None = create new)
        assert result is None

    def test_strategy8_name_variations_kam_wai_chik(self):
        """Strategy 7: Test various name permutations of Kam Wai Warren Bartholomew Chik.

        This tests that different name variations (with/without initials, short forms,
        different word orderings) all resolve correctly via subset matching.
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        # Full name in lookup
        processor._professor_lookup["KAM WAI WARREN BARTHOLOMEW CHIK"] = {
            "database_id": "uuid-kam-wai",
            "boss_name": "KAM WAI WARREN BARTHOLOMEW CHIK",
            "afterclass_name": "Kam Wai Warren Bartholomew Chik"
        }

        # Test various name permutations
        test_cases = [
            # (input_name, expected_match)
            ("Kam Wai Warren Bartholomew CHIK", True),   # Exact
            ("Kam Wai CHIK", True),                     # Short form (first + surname)
            ("Warren B. CHIK", True),                   # Middle initial short form
            ("Kam Wai Warren B. CHIK", True),           # Partial with initial
            ("Kam Wai", True),                          # Very short
            ("Bartholomew CHIK", True),                  # Middle + surname
            ("Warren Bartholomew CHIK", True),          # Last part + surname
            ("KAM WAI CHIK", True),                    # All caps short
            ("WARREN CHIK", True),                      # Just last + surname
            ("KAM WAI WARREN CHIK", True),              # Partial
            # Additional variations
            ("Kam Wai W. CHIK", True),                  # Single initial
            ("Kam W. CHIK", True),                       # Partial with initial
            ("Warren Bartholomew B. CHIK", True),       # Two words + initial
            ("W. CHIK", False),                         # Single initial only (not 2+ words)
            ("Kam CHIK", True),                          # Short first + surname
            ("Wai CHIK", True),                         # Middle name + surname
            ("Kam Wai Bartholomew CHIK", True),         # Skip Warren
            ("WARREN BARTHOLOMEW CHIK", True),         # Two middle + surname
            ("Kam Wai Warren Bartholomew C.", True),   # Initial C removed → 4-word subset of 5
            ("KAM WAI WAREN CHIK", False),               # Typo - WAREN ≠ WARREN (not fuzzy)
        ]

        for name, should_match in test_cases:
            boss, after = processor._normalize_professor_name_fallback(name)
            result = processor._resolve_single_professor(name, boss, after)
            expected = 'matched' if should_match else None
            assert result == expected, \
                f"Failed for '{name}': expected {expected}, got {result}"

    def test_strategy8_session_deduplication(self):
        """Strategy 8: Session deduplication - matches against newly created professors.

        When multiple professors are processed in the same batch, Strategy 8 ensures
        that subsequent variations of the same professor are matched to the first
        created professor, not creating duplicates.
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        # Simulate a professor was already created in this session
        existing_dto = ProfessorDTO(
            id="uuid-session-1",
            name="Kam Wai Warren Bartholomew Chik",
            email="kamwai@smu.edu.sg",
            slug="kam-wai-warren-bartholomew-chik",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["KAM WAI WARREN BARTHOLOMEW CHIK", "KAM WAI CHIK", "WARREN CHIK"],
            original_scraped_name="Kam Wai Warren Bartholomew CHIK"
        )
        processor._new_professors_dtos.append(existing_dto)

        # Now try to resolve a variation of the same professor
        result = processor._resolve_single_professor(
            "Kam Wai CHIK",
            "KAM WAI CHIK",
            "Kam Wai CHIK"
        )

        # Should match via session deduplication (Strategy 8)
        assert result == 'matched'
        assert processor._professor_lookup["KAM WAI CHIK"]["database_id"] == "uuid-session-1"

    def test_creates_new_when_no_match(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        result = processor._resolve_single_professor("New Professor", "NEW PROFESSOR", "New Professor")

        assert result is None  # None means should CREATE


class TestProcessMethod:
    """Integration tests for process() method."""

    def test_process_returns_new_professors(self):
        raw_data = pd.DataFrame({
            'professor_name': ['Brand New Professor']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={}
        )

        new_profs, updated_profs = processor.process()

        assert len(new_profs) == 1
        assert isinstance(new_profs[0], ProfessorDTO)
        # afterclass_name normalizes last word to uppercase if identified as surname
        assert new_profs[0].name == 'Brand New PROFESSOR'
        assert len(updated_profs) == 0

    def test_process_matches_existing_professor(self):
        raw_data = pd.DataFrame({
            'professor_name': ['JOHN SMITH']
        })

        processor = ProfessorProcessor(
            raw_data=raw_data,
            professors_cache={"JOHN SMITH": {"id": "existing-uuid", "name": "John Smith"}}
        )

        new_profs, updated_profs = processor.process()

        # Should NOT create new - found in cache
        assert len(new_profs) == 0
        assert len(updated_profs) == 0


class TestCreateNewProfessor:
    """Tests for _create_new_professor method."""

    def test_creates_dto_with_correct_fields(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        dto = processor._create_new_professor(
            prof_name="Dr. Test Professor",
            boss_name="TEST PROFESSOR",
            afterclass_name="Test Professor",
            variations=set()
        )

        assert dto.id is not None  # UUID generated
        assert dto.name == "Test Professor"
        assert dto.email == "enquiry@smu.edu.sg"  # Default
        assert "TEST PROFESSOR" in dto.boss_aliases
        assert "DR. TEST PROFESSOR" in dto.boss_aliases  # original name added
        assert dto.original_scraped_name == "Dr. Test Professor"
        assert dto.updated_at is None  # CREATE mode

    def test_creates_with_variations(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        dto = processor._create_new_professor(
            prof_name="Test",
            boss_name="TEST",
            afterclass_name="Test Prof",
            variations={"TEST", "PROF TEST"}
        )

        # Should include variations in boss_aliases
        assert len(dto.boss_aliases) >= 2


class TestCleanAlias:
    """Tests for _clean_alias helper method.

    This method handles special character encoding and normalization.
    """

    def test_handles_none(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._clean_alias(None)
        assert result == ""

    def test_handles_empty_string(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._clean_alias("")
        assert result == ""

    def test_handles_regular_name(self):
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._clean_alias("JOHN SMITH")
        assert result == "JOHN SMITH"

    def test_normalizes_special_quotes(self):
        """Should normalize special quote characters to straight quotes."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._clean_alias("JOHN'S SMITH")
        assert "'" in result  # Should be normalized


class TestSplitProfessorNames:
    """Tests for _split_professor_names method.

    This method splits professor names separated by comma.
    Only splits if the part after comma is a single word.
    """

    def test_split_simple_comma_separated(self):
        """Should split 'SMITH, JOHN' into ['SMITH', 'JOHN']."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._split_professor_names("SMITH, JOHN")
        assert result == ["SMITH", "JOHN"]

    def test_no_split_single_name(self):
        """Should not split single name without comma."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._split_professor_names("JOHN SMITH")
        assert result == ["JOHN SMITH"]

    def test_no_split_multi_word_second_part(self):
        """Should NOT split if part after comma is multi-word.

        e.g., 'YUESHEN, BART ZHOU' should NOT be split because 'BART ZHOU' is 2 words.
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._split_professor_names("YUESHEN, BART ZHOU")
        # Multi-word second part should be kept intact
        assert result == ["YUESHEN, BART ZHOU"]

    def test_split_with_single_word_after_comma(self):
        """Should split if part after comma is a single word."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._split_professor_names("SMITH, JOHN")
        assert len(result) == 2
        assert "SMITH" in result
        assert "JOHN" in result


class TestBuildLookupFromCache:
    """Tests for _build_lookup_from_cache method.

    This builds the professor_lookup from professors_cache for resolution.
    """

    def test_builds_lookup_with_names(self):
        """Should build lookup with professor names from cache."""
        professors_cache = {
            "JOHN SMITH": {"id": "uuid-1", "name": "John Smith", "boss_aliases": []},
            "JANE DOE": {"id": "uuid-2", "name": "Jane Doe", "boss_aliases": []}
        }
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache=professors_cache
        )

        processor._build_lookup_from_cache()

        assert "JOHN SMITH" in processor._professor_lookup
        assert "JANE DOE" in processor._professor_lookup

    def test_builds_lookup_with_aliases(self):
        """Should also add boss_aliases to lookup."""
        professors_cache = {
            "JOHN SMITH": {
                "id": "uuid-1",
                "name": "John Smith",
                "boss_aliases": '["J. SMITH", "JOHN S."]'
            }
        }
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache=professors_cache
        )

        processor._build_lookup_from_cache()

        # Should have both the original name and aliases
        assert "JOHN SMITH" in processor._professor_lookup
        assert "J. SMITH" in processor._professor_lookup
        assert "JOHN S." in processor._professor_lookup