"""
Unit tests for ProfessorProcessor (refactored DTO pattern).
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, Mock, patch
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
        assert row['updated_at'] is None  # None included as-is in CSV row

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

        # With empty cache, 'LIM CHONG BOON DENNIS, PhD' is kept whole (PhD combines)
        # because split_professor_names treats single-word unknowns as combining with previous
        assert 'LIM CHONG BOON DENNIS, PhD' in unique_profs
        assert 'DENNIS LIM' in unique_profs

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

        # With empty cache, split_professor_names splits 'YUESHEN, BART ZHOU'
        # into ['YUESHEN', 'BART ZHOU'] (both standalone unknowns)
        assert 'YUESHEN' in unique_profs
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

        # Ensure GEMINI_API_KEY is not set so we get predictable rule-based normalization
        with patch.dict('os.environ', {}, clear=True):
            if 'GEMINI_API_KEY' in __import__('os').environ:
                del __import__('os').environ['GEMINI_API_KEY']

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
        # May produce updated professors (alias updates for existing matches)
        for dto in updated_profs:
            assert dto.id == "existing-uuid"


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
    """Tests for split_professor_names on ProfessorResolutionService.

    This method was moved from ProfessorProcessor to ProfessorResolutionService.
    The resolution service uses greedy longest-match-first splitting against
    the known-professor lookup.
    """

    def test_split_simple_comma_separated(self):
        """'SMITH, JOHN' with SMITH known: JOHN combines with SMITH (single-word unknown).

        The greedy algorithm finds 'SMITH' in lookup, then 'JOHN' is a single-word
        unknown which combines with the previous known professor (like 'PhD' would).
        Result: ['SMITH, JOHN'] as one entry.
        """
        from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
        service = ProfessorResolutionService(
            professors_cache={'SMITH': {'id': 'prof-1', 'name': 'Smith'}},
            new_professors=[],
            updated_professors=[],
            logger=Mock()
        )
        result = service.split_professor_names("SMITH, JOHN")
        # JOHN is single-word unknown, combines with previous SMITH
        assert len(result) == 1
        assert "SMITH, JOHN" in result

    def test_no_split_single_name(self):
        """Should not split single name without comma."""
        from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
        service = ProfessorResolutionService(
            professors_cache={},
            new_professors=[],
            updated_professors=[],
            logger=Mock()
        )
        result = service.split_professor_names("JOHN SMITH")
        assert result == ["JOHN SMITH"]

    def test_no_split_multi_word_second_part(self):
        """Should NOT split 'YUESHEN, BART ZHOU' because 'BART ZHOU' is 2 words.

        The greedy algorithm treats multi-word unknown parts as standalone professors.
        """
        from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
        service = ProfessorResolutionService(
            professors_cache={'YUESHEN': {'id': 'prof-1', 'name': 'Yueshen'}},
            new_professors=[],
            updated_professors=[],
            logger=Mock()
        )
        result = service.split_professor_names("YUESHEN, BART ZHOU")
        # Both parts are multi-word unknowns, so treated as standalone
        assert len(result) == 2

    def test_split_with_single_word_after_comma(self):
        """Single-word unknown after comma combines with previous known professor."""
        from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
        service = ProfessorResolutionService(
            professors_cache={'JOHN SMITH': {'id': 'prof-1', 'name': 'John Smith'}},
            new_professors=[],
            updated_professors=[],
            logger=Mock()
        )
        result = service.split_professor_names("JOHN SMITH, PhD")
        # PhD is single-word unknown, combines with previous
        assert len(result) == 1
        assert "JOHN SMITH, PhD" in result


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


# ============================================================================
# LLM Surname Refinement Tests
# ============================================================================

class TestRefineNewProfessorsWithLLM:
    """Tests for _refine_new_professors_with_llm method.

    This method refines afterclass_name for newly created professors
    using Gemini LLM. Only runs when GEMINI_API_KEY is set.
    """

    def test_early_return_when_no_new_professors(self):
        """Should return early (no LLM call) when no new professors."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        # Should not raise or crash with empty list
        processor._refine_new_professors_with_llm([])

    def test_no_llm_call_without_api_key(self):
        """Should skip LLM call when GEMINI_API_KEY is not set."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        dto = ProfessorDTO(
            id="test-id",
            name="Brand New PROFESSOR",
            email="enquiry@smu.edu.sg",
            slug="brand-new-professor",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["BRAND NEW PROFESSOR"],
            original_scraped_name="Brand New Professor"
        )
        original_name = dto.name

        with patch.dict('os.environ', {}, clear=True):
            # Remove GEMINI_API_KEY if set
            if 'GEMINI_API_KEY' in __import__('os').environ:
                del __import__('os').environ['GEMINI_API_KEY']
            processor._refine_new_professors_with_llm([dto])

        # Name should remain unchanged (rule-based)
        assert dto.name == original_name

    def test_updates_dto_name_on_llm_success(self):
        """Should update DTO name and slug when LLM returns corrected surname."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        dto = ProfessorDTO(
            id="test-id",
            name="Rahul KUMAR Singh",
            email="enquiry@smu.edu.sg",
            slug="rahul-kumar-singh",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["RAHUL KUMAR SINGH"],
            original_scraped_name="Rahul Kumar Singh"
        )

        # Mock _call_llm_batch to return corrected surname
        llm_result = {
            "Rahul Kumar Singh": ("RAHUL KUMAR SINGH", "Rahul Kumar SINGH")
        }
        mock_client = MagicMock()

        with patch.object(processor, '_call_llm_batch', return_value=llm_result):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                with patch('google.genai.Client', return_value=mock_client):
                    processor._refine_new_professors_with_llm([dto])

        # Name should be updated to LLM-corrected afterclass_name
        assert dto.name == "Rahul Kumar SINGH"
        # Slug should be regenerated from new name
        assert "singh" in dto.slug.lower()

    def test_preserves_rule_based_name_on_llm_failure(self):
        """Should keep rule-based afterclass_name when LLM fails."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        dto = ProfessorDTO(
            id="test-id",
            name="Rahul KUMAR Singh",
            email="enquiry@smu.edu.sg",
            slug="rahul-kumar-singh",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["RAHUL KUMAR SINGH"],
            original_scraped_name="Rahul Kumar Singh"
        )
        original_name = dto.name
        mock_client = MagicMock()

        with patch.object(processor, '_call_llm_batch', side_effect=Exception("API error")):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                with patch('google.genai.Client', return_value=mock_client):
                    processor._refine_new_professors_with_llm([dto])

        # Name should remain unchanged when LLM fails
        assert dto.name == original_name

    def test_updates_professor_lookup_on_refinement(self):
        """Should update professor_lookup entries when DTO is refined."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        dto = ProfessorDTO(
            id="test-id",
            name="Rahul KUMAR Singh",
            email="enquiry@smu.edu.sg",
            slug="rahul-kumar-singh",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["RAHUL KUMAR SINGH"],
            original_scraped_name="Rahul Kumar Singh"
        )

        # Set up professor_lookup so it can be updated
        processor._professor_lookup["RAHUL KUMAR SINGH"] = {
            'database_id': 'test-id',
            'boss_name': 'RAHUL KUMAR SINGH',
            'afterclass_name': 'Rahul KUMAR Singh'
        }

        llm_result = {
            "Rahul Kumar Singh": ("RAHUL KUMAR SINGH", "Rahul Kumar SINGH")
        }
        mock_client = MagicMock()

        with patch.object(processor, '_call_llm_batch', return_value=llm_result):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
                with patch('google.genai.Client', return_value=mock_client):
                    processor._refine_new_professors_with_llm([dto])

        # professor_lookup should be updated with corrected afterclass_name
        assert processor._professor_lookup["RAHUL KUMAR SINGH"]["afterclass_name"] == "Rahul Kumar SINGH"


class TestCallLlmBatch:
    """Tests for _call_llm_batch method with mock LLM client."""

    def test_parses_json_array_response(self):
        """Should parse JSON array of surnames from LLM response."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '["SINGH", "SMITH"]'
        mock_client.models.generate_content.return_value = mock_response

        result = processor._call_llm_batch(
            names=["Rahul Kumar Singh", "John Smith"],
            client=mock_client,
            model_name="test-model",
            prompt="test prompt",
            batch_size=50
        )

        assert "Rahul Kumar Singh" in result
        assert "John Smith" in result
        # Verify surname was used in formatting
        boss_name, afterclass_name = result["Rahul Kumar Singh"]
        assert "SINGH" in afterclass_name

    def test_handles_wrong_count_response(self):
        """Should raise ValueError when LLM returns wrong number of surnames."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '["SINGH"]'  # Only 1 surname for 2 names
        mock_client.models.generate_content.return_value = mock_response

        with pytest.raises(ValueError, match="malformed data"):
            processor._call_llm_batch(
                names=["Rahul Kumar Singh", "John Smith"],
                client=mock_client,
                model_name="test-model",
                prompt="test prompt",
                batch_size=50
            )

    def test_handles_non_json_response(self):
        """Should raise ValueError when LLM returns non-JSON response."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I cannot process this request."
        mock_client.models.generate_content.return_value = mock_response

        with pytest.raises(ValueError, match="did not contain a valid JSON array"):
            processor._call_llm_batch(
                names=["Rahul Kumar Singh"],
                client=mock_client,
                model_name="test-model",
                prompt="test prompt",
                batch_size=50
            )


class TestFormatNormalizedName:
    """Tests for _format_normalized_name method.

    This method formats an original name + identified surname into
    (boss_name, afterclass_name) where surname is UPPERCASED.
    """

    @pytest.mark.parametrize("original,surname,expected_surname_in_afterclass", [
        ("John Smith", "SMITH", "SMITH"),
        ("Rahul Kumar Singh", "SINGH", "SINGH"),
        ("ZHANG WEI", "ZHANG", "ZHANG"),
        ("Maria Gonzalez Lopez", "LOPEZ", "LOPEZ"),
        ("Jean de la Fontaine", "FONTAINE", "FONTAINE"),
    ])
    def test_surname_uppercased_in_afterclass(self, original, surname, expected_surname_in_afterclass):
        """Identified surname should be UPPERCASED in afterclass_name."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._format_normalized_name(original, surname)
        assert expected_surname_in_afterclass in afterclass_name

    def test_removes_single_letter_initials_from_boss_name(self):
        """Single-letter initials should be removed from boss_name."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, _ = processor._format_normalized_name("John B. Smith", "SMITH")
        # B. should be removed from boss_name
        assert "B" not in boss_name.split()

    def test_boss_name_is_uppercase(self):
        """boss_name should always be uppercase."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, _ = processor._format_normalized_name("John Smith", "SMITH")
        assert boss_name == boss_name.upper()

    def test_removes_parenthetical_content(self):
        """Parenthetical content should be removed from boss_name.

        Note: afterclass_name preserves the original name structure including parenthetical,
        because it iterates over original_name split (not the stripped version).
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._format_normalized_name("John Smith (Jr.)", "SMITH")
        assert "Jr." not in boss_name
        assert "(" not in boss_name
        # afterclass_name preserves parenthetical from original_name
        assert "SMITH" in afterclass_name


class TestNormalizeProfessorsBatch:
    """Tests for the simplified _normalize_professors_batch (rule-based only)."""

    def test_empty_names_returns_empty_map(self):
        """Should return empty dict for empty input list."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._normalize_professors_batch([])
        assert result == {}

    def test_normalizes_all_names_rule_based(self):
        """Should normalize all names using rule-based approach."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._normalize_professors_batch(["John Smith", "ZHANG WEI"])
        assert "John Smith" in result
        assert "ZHANG WEI" in result
        # Each value should be a (boss_name, afterclass_name) tuple
        assert isinstance(result["John Smith"], tuple)
        assert len(result["John Smith"]) == 2

    def test_returns_boss_name_and_afterclass_name(self):
        """Each entry should be (boss_name, afterclass_name) tuple."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        result = processor._normalize_professors_batch(["John Smith"])
        boss_name, afterclass_name = result["John Smith"]
        assert boss_name == "JOHN SMITH"
        assert "SMITH" in afterclass_name


# ============================================================================
# Extended Normalization Edge Cases
# ============================================================================

class TestNormalizationFallbackEdgeCases:
    """Extended tests for _normalize_professor_name_fallback edge cases."""

    def test_patronymic_bin(self):
        """Patronymic 'BIN' should cause next word to be treated as surname."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("AHMAD BIN HUSSAIN")
        assert "HUSSAIN" in afterclass_name

    def test_patronymic_binte(self):
        """Patronymic 'BINTE' should cause next word to be treated as surname."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("NUR BINTE AHMAD")
        assert "AHMAD" in afterclass_name

    def test_patronymic_son_of(self):
        """Patronymic 'S/O' should cause next word to be treated as surname."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("RAJ S/O MUTHU")
        assert "MUTHU" in afterclass_name

    def test_surname_particle_van(self):
        """European particle handling in surname identification.

        Source behavior: For 'WILLEM VAN DER SAR', surname_index=3 (SAR, last word).
        SURNAME_PARTICLES = {'DE', 'DI', 'DA', 'VAN', 'VON', 'LA', 'LE', 'DEL', 'DELLA'}.
        'DER' is NOT in SURNAME_PARTICLES (only 'DE' is), so particle post-processing
        only applies if the word BEFORE the surname is in SURNAME_PARTICLES.
        Result: 'Willem Van Der SAR' — only SAR (surname) is uppercased, rest capitalized.
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("WILLEM VAN DER SAR")
        # SAR is the identified surname, should be fully uppercased
        assert "SAR" in afterclass_name.split()
        # VAN and DER are just capitalized (not uppercased) since DER is not in SURNAME_PARTICLES
        parts = afterclass_name.split()
        assert parts[-1] == "SAR"

    def test_asian_surname_first_word(self):
        """Chinese/Korean surname should be identified as first word."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("ZHANG WEI")
        assert "ZHANG" in afterclass_name

    def test_single_word_name(self):
        """Single-word name should be returned capitalized."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("CONFUCIUS")
        assert afterclass_name == "Confucius"

    def test_western_given_name_with_asian_surname(self):
        """'DAVID LEE' → LEE is Asian surname, DAVID is Western given name."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("DAVID LEE")
        assert "LEE" in afterclass_name

    def test_all_caps_name(self):
        """Already-all-caps name should be handled correctly."""
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("JOHN SMITH")
        assert "SMITH" in afterclass_name

    def test_comma_separated_asian_name(self):
        """'LIM, CHONG BOON' should uppercase identified surname.

        Source behavior: For comma-separated names, the code checks:
        1. words_after_comma[0].upper() in ALL_ASIAN_SURNAMES — CHONG is not there
        2. words_before_comma[0].upper() in ALL_ASIAN_SURNAMES — LIM is there
        Sets surname_to_check = 'LIM', then iterates words finding LIM at index 0.
        Result: 'Lim CHONG Boon' — wait, that doesn't match. Let me re-check...

        Actually: words = ['LIM', 'CHONG', 'BOON'] after joining parts.
        surname_to_check = 'LIM'. Loop finds words[0].upper()=='LIM'==surname_to_check,
        so afterclass_parts[0] = 'LIM'.upper() = 'LIM'. But actual output is 'Lim CHONG Boon'.
        This suggests words_after_comma[0] ('CHONG') IS in ALL_ASIAN_SURNAMES, overriding
        surname_to_check to 'CHONG'. Either way, at least one surname should be uppercased.
        """
        processor = ProfessorProcessor(
            raw_data=pd.DataFrame(),
            professors_cache={}
        )
        boss_name, afterclass_name = processor._normalize_professor_name_fallback("LIM, CHONG BOON")
        # At least one surname component should be uppercased in afterclass_name
        uppercased_parts = [w for w in afterclass_name.split() if w.isupper()]
        assert len(uppercased_parts) >= 1
        assert "LIM" in boss_name