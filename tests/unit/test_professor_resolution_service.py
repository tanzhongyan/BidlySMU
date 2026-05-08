"""
Unit tests for ProfessorResolutionService - the single source of truth
for professor name resolution used by both ProfessorProcessor and ClassProcessor.
"""
import pytest
from unittest.mock import Mock

from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService
from src.pipeline.dtos.professor_dto import ProfessorDTO


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def empty_service():
    """Service with no cache data."""
    return ProfessorResolutionService(
        professors_cache={},
        new_professors=[],
        updated_professors=[],
        logger=Mock()
    )


@pytest.fixture
def service_with_cache():
    """Service with a populated DB cache.

    boss_aliases in the DB cache are stored as JSON strings (from PostgreSQL),
    NOT as Python lists containing JSON strings.
    """
    cache = {
        'JOHN SMITH': {'id': 'prof-1', 'name': 'SMITH, John', 'boss_aliases': '["JOHN SMITH", "J. SMITH"]'},
        'JANE DOE': {'id': 'prof-2', 'name': 'DOE, Jane', 'boss_aliases': '["JANE DOE"]'},
        'KAM WAI WARREN CHIK': {'id': 'prof-3', 'name': 'CHIK, Kam Wai Warren', 'boss_aliases': '["KAM WAI WARREN CHIK", "WARREN CHIK", "KAM WAI CHIK"]'},
        'ZHANG WEI': {'id': 'prof-4', 'name': 'ZHANG, Wei', 'boss_aliases': '[]'},
    }
    return ProfessorResolutionService(
        professors_cache=cache,
        new_professors=[],
        updated_professors=[],
        logger=Mock()
    )


@pytest.fixture
def service_with_valid_ids():
    """Service with valid_professor_ids set for validation."""
    cache = {
        'JOHN SMITH': {'id': 'prof-1', 'name': 'SMITH, John', 'boss_aliases': []},
    }
    return ProfessorResolutionService(
        professors_cache=cache,
        new_professors=[],
        updated_professors=[],
        logger=Mock(),
        valid_professor_ids={'prof-1'}
    )


# ============================================================================
# split_professor_names
# ============================================================================

class TestSplitProfessorNames:
    """Tests for greedy longest-match-first splitting algorithm."""

    def test_single_name_no_split(self, service_with_cache):
        """Single name without commas should be returned as-is."""
        result = service_with_cache.split_professor_names("JOHN SMITH")
        assert result == ["JOHN SMITH"]

    def test_empty_string(self, service_with_cache):
        """Empty string should return empty list."""
        result = service_with_cache.split_professor_names("")
        assert result == []

    def test_none_input(self, service_with_cache):
        """None input should return empty list."""
        result = service_with_cache.split_professor_names(None)
        assert result == []

    def test_entire_string_known_professor(self, service_with_cache):
        """If entire string is a known professor, return as single entry."""
        result = service_with_cache.split_professor_names("JOHN SMITH")
        assert result == ["JOHN SMITH"]

    def test_two_known_professors_comma_separated(self, service_with_cache):
        """Two known professors separated by comma should both be returned."""
        result = service_with_cache.split_professor_names("JOHN SMITH, JANE DOE")
        assert len(result) == 2
        assert "JOHN SMITH" in result
        assert "JANE DOE" in result

    def test_three_known_professors_comma_separated(self, service_with_cache):
        """Three known professors separated by commas should all be returned."""
        result = service_with_cache.split_professor_names("ZHANG WEI, JOHN SMITH, JANE DOE")
        assert len(result) == 3

    def test_greedy_longest_match(self, service_with_cache):
        """Greedy algorithm should match longest combination first.

        'KAM WAI WARREN CHIK' is a single known professor (4 words).
        A naive split-by-comma would incorrectly split this if commas existed.
        """
        result = service_with_cache.split_professor_names("KAM WAI WARREN CHIK, JANE DOE")
        assert len(result) == 2
        assert "KAM WAI WARREN CHIK" in result
        assert "JANE DOE" in result

    def test_single_word_unknown_combined_with_previous(self, service_with_cache):
        """Single-word unknown part should combine with previous known professor.

        If "PhD" is unknown, it should be appended to the previous known professor,
        not treated as a standalone entry.
        """
        result = service_with_cache.split_professor_names("JOHN SMITH, PhD")
        assert len(result) == 1
        assert "JOHN SMITH, PhD" in result

    def test_multi_word_unknown_treated_as_standalone(self, service_with_cache):
        """Multi-word unknown part should be treated as standalone professor."""
        result = service_with_cache.split_professor_names("JOHN SMITH, NEW PROFESSOR")
        assert len(result) == 2

    def test_no_commas_returns_as_is(self, service_with_cache):
        """Name without commas should be returned as single-element list."""
        result = service_with_cache.split_professor_names("UNKNOWN PROFESSOR")
        assert result == ["UNKNOWN PROFESSOR"]

    def test_all_unknown_comma_separated(self, service_with_cache):
        """All unknown parts should each be standalone if multi-word."""
        result = service_with_cache.split_professor_names("ALPHA BETA, GAMMA DELTA")
        assert len(result) == 2

    def test_whitespace_handling(self, service_with_cache):
        """Extra whitespace around parts should be trimmed."""
        result = service_with_cache.split_professor_names("  JOHN SMITH  ,  JANE DOE  ")
        assert len(result) == 2


# ============================================================================
# resolve_professor_name
# ============================================================================

class TestResolveProfessorName:
    """Tests for the 7-strategy resolution chain."""

    # --- Strategy 1: Direct lookup ---

    def test_strategy1_direct_lookup_match(self, service_with_cache):
        """Strategy 1: Exact name match in direct_lookup."""
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == 'prof-1'

    def test_strategy1_case_insensitive(self, service_with_cache):
        """Strategy 1: Lookup should be case-insensitive (uppercased internally)."""
        result = service_with_cache.resolve_professor_name("john smith")
        assert result == 'prof-1'

    # --- Strategy 2: Variation lookup ---

    def test_strategy2_comma_variation(self, service_with_cache):
        """Strategy 2: 'LAST, FIRST' variation should match 'FIRST LAST'."""
        result = service_with_cache.resolve_professor_name("SMITH, JOHN")
        assert result == 'prof-1'

    # --- Strategy 3: Full lookup (database_id) ---

    def test_strategy3_full_lookup_match(self, service_with_cache):
        """Strategy 3: Name found in full_lookup with database_id."""
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result is not None

    # --- Strategy 4: Boss alias lookup ---

    def test_strategy4_boss_alias_match(self, service_with_cache):
        """Strategy 4: Match via boss_aliases field."""
        result = service_with_cache.resolve_professor_name("J. SMITH")
        assert result == 'prof-1'

    def test_strategy4_alias_warren_chik(self, service_with_cache):
        """Strategy 4: Match 'WARREN CHIK' via boss_aliases of KAM WAI WARREN CHIK."""
        result = service_with_cache.resolve_professor_name("WARREN CHIK")
        assert result == 'prof-3'

    # --- Strategy 5: Subset matching ---

    def test_strategy5_subset_match(self, service_with_cache):
        """Strategy 5: 'KAM WAI CHIK' is a subset of 'KAM WAI WARREN CHIK'."""
        result = service_with_cache.resolve_professor_name("KAM WAI CHIK")
        assert result == 'prof-3'

    def test_strategy5_subset_requires_min_2_words(self, service_with_cache):
        """Strategy 5: Single-word queries should NOT trigger subset matching."""
        result = service_with_cache.resolve_professor_name("CHIK")
        assert result is None

    def test_strategy5_subset_no_self_match(self, service_with_cache):
        """Strategy 5: Should not match a name against itself."""
        # 'JOHN SMITH' should match via Strategy 1, not Strategy 5
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == 'prof-1'

    # --- Strategy 6: New professors ---

    def test_strategy6_new_professor_match(self):
        """Strategy 6: Match against session-created new professors."""
        new_dto = ProfessorDTO(
            id="new-prof-1",
            name="NEW PROFESSOR",
            email="enquiry@smu.edu.sg",
            slug="new-professor",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["NEW PROFESSOR", "N. PROFESSOR"],
            original_scraped_name="New Professor"
        )
        service = ProfessorResolutionService(
            professors_cache={},
            new_professors=[new_dto],
            updated_professors=[],
            logger=Mock()
        )

        result = service.resolve_professor_name("NEW PROFESSOR")
        assert result == "new-prof-1"

    def test_strategy6_new_professor_alias_match(self):
        """Strategy 6: Match via alias of session-created professor."""
        new_dto = ProfessorDTO(
            id="new-prof-2",
            name="ANOTHER PROF",
            email="enquiry@smu.edu.sg",
            slug="another-prof",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["ANOTHER PROF", "A. PROF"],
            original_scraped_name="Another Prof"
        )
        service = ProfessorResolutionService(
            professors_cache={},
            new_professors=[new_dto],
            updated_professors=[],
            logger=Mock()
        )

        result = service.resolve_professor_name("A. PROF")
        assert result == "new-prof-2"

    # --- Strategy 7: No match ---

    def test_strategy7_no_match_returns_none(self, service_with_cache):
        """Strategy 7: Unresolvable name should return None."""
        result = service_with_cache.resolve_professor_name("COMPLETELY UNKNOWN PERSON")
        assert result is None

    # --- Edge cases ---

    def test_empty_string_returns_none(self, service_with_cache):
        """Empty string should return None."""
        result = service_with_cache.resolve_professor_name("")
        assert result is None

    def test_none_returns_none(self, service_with_cache):
        """None should return None."""
        result = service_with_cache.resolve_professor_name(None)
        assert result is None

    def test_tba_returns_none(self, service_with_cache):
        """'TBA' should return None (filtered as placeholder)."""
        result = service_with_cache.resolve_professor_name("TBA")
        assert result is None

    def test_nan_string_returns_none(self, service_with_cache):
        """String 'nan' should return None (filtered as placeholder)."""
        result = service_with_cache.resolve_professor_name("nan")
        assert result is None

    def test_to_be_announced_returns_none(self, service_with_cache):
        """'TO BE ANNOUNCED' should return None."""
        result = service_with_cache.resolve_professor_name("TO BE ANNOUNCED")
        assert result is None


# ============================================================================
# _generate_variations
# ============================================================================

class TestGenerateVariations:
    """Tests for name variation generation used by resolution chain."""

    def test_generates_comma_removal_variation(self, service_with_cache):
        """Should generate variation with comma removed."""
        variations = service_with_cache._generate_variations("SMITH, JOHN")
        assert "SMITH JOHN" in variations

    def test_generates_space_normalization(self, service_with_cache):
        """Should generate variation with normalized spaces."""
        variations = service_with_cache._generate_variations("SMITH  JOHN")
        assert "SMITH JOHN" in variations

    def test_generates_reordering_variation(self, service_with_cache):
        """Should generate 'FIRST LAST' → 'FIRST, LAST' comma variation."""
        variations = service_with_cache._generate_variations("JOHN SMITH")
        assert "JOHN, SMITH" in variations

    def test_generates_three_word_comma_variation(self, service_with_cache):
        """Should generate comma variations for three-word names."""
        variations = service_with_cache._generate_variations("KAM WAI CHIK")
        assert "KAM, WAI CHIK" in variations
        assert "KAM WAI, CHIK" in variations

    def test_no_comma_variation_for_already_comma_name(self, service_with_cache):
        """Should not generate reordering variation when commas already present."""
        variations = service_with_cache._generate_variations("SMITH, JOHN")
        # Should NOT add "SMITH, JOHN, " (double-comma variation)
        comma_variations = [v for v in variations if v != "SMITH, JOHN" and ',' in v and 'SMITH' in v]
        # At least the comma-removed version should exist
        assert any("SMITH JOHN" in v for v in variations)


# ============================================================================
# _is_valid_professor_id
# ============================================================================

class TestIsValidProfessorId:
    """Tests for professor ID validation."""

    def test_valid_id_in_set(self, service_with_valid_ids):
        """ID in valid_professor_ids should be valid."""
        assert service_with_valid_ids._is_valid_professor_id('prof-1') is True

    def test_invalid_id_not_in_set(self, service_with_valid_ids):
        """ID not in valid_professor_ids should be invalid."""
        assert service_with_valid_ids._is_valid_professor_id('prof-999') is False

    def test_session_created_id_always_valid(self, service_with_valid_ids):
        """Session-created professor ID should always be valid."""
        new_dto = ProfessorDTO(
            id="new-session-id",
            name="SESSION PROF",
            email="enquiry@smu.edu.sg",
            slug="session-prof",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["SESSION PROF"],
            original_scraped_name="Session Prof"
        )
        service_with_valid_ids.update_with_session_professors([new_dto], [])
        assert service_with_valid_ids._is_valid_professor_id('new-session-id') is True

    def test_no_validation_set_accepts_all(self, empty_service):
        """Without valid_professor_ids, all IDs should be accepted."""
        assert empty_service._is_valid_professor_id('any-id') is True


# ============================================================================
# update_with_session_professors
# ============================================================================

class TestUpdateWithSessionProfessors:
    """Tests for incremental lookup update with session professors."""

    def test_adds_new_professor_to_lookups(self, empty_service):
        """New professor should be added to all internal lookups."""
        new_dto = ProfessorDTO(
            id="new-1",
            name="NEW PROF",
            email="enquiry@smu.edu.sg",
            slug="new-prof",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["NEW PROF", "N. PROF"],
            original_scraped_name="New Prof"
        )
        empty_service.update_with_session_professors([new_dto], [])

        result = empty_service.resolve_professor_name("NEW PROF")
        assert result == "new-1"

    def test_updated_professor_does_not_overwrite_existing(self, service_with_cache):
        """Updated professor should NOT overwrite existing correct mappings.

        If 'JOHN SMITH' already maps to prof-1 in DB, an updated professor
        with the same name should NOT change the mapping.
        """
        updated_dto = ProfessorDTO(
            id="different-id",
            name="SMITH, John",
            email="enquiry@smu.edu.sg",
            slug="smith-john",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["JOHN SMITH"],
            original_scraped_name="John Smith"
        )
        # prof-1 is the existing DB ID for JOHN SMITH
        original_id = service_with_cache.resolve_professor_name("JOHN SMITH")

        service_with_cache.update_with_session_professors([], [updated_dto])

        # Should still return original prof-1, not different-id
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == original_id

    def test_adds_new_aliases_for_updated_professor(self, service_with_cache):
        """Updated professor's new aliases should be added if not already present."""
        updated_dto = ProfessorDTO(
            id="prof-1",
            name="SMITH, John",
            email="enquiry@smu.edu.sg",
            slug="smith-john",
            photo_url="https://smu.edu.sg",
            profile_url="https://smu.edu.sg",
            boss_aliases=["JOHN SMITH", "DR. JOHN SMITH"],
            original_scraped_name="Dr. John Smith"
        )
        service_with_cache.update_with_session_professors([], [updated_dto])

        # "DR. JOHN SMITH" should now be resolvable
        result = service_with_cache.resolve_professor_name("DR. JOHN SMITH")
        assert result == "prof-1"


# ============================================================================
# update_with_professor_lookup
# ============================================================================

class TestUpdateWithProfessorLookup:
    """Tests for external professor_lookup merge."""

    def test_merges_new_entries(self, empty_service):
        """Should merge entries from external lookup into direct_lookup."""
        lookup = {
            'JOHN SMITH': {'database_id': 'prof-1', 'boss_name': 'JOHN SMITH', 'afterclass_name': 'Smith, John'}
        }
        empty_service.update_with_professor_lookup(lookup)

        result = empty_service.resolve_professor_name("JOHN SMITH")
        assert result == 'prof-1'

    def test_does_not_overwrite_existing(self, service_with_cache):
        """Should NOT overwrite existing entries in direct_lookup."""
        original_id = service_with_cache.resolve_professor_name("JOHN SMITH")

        lookup = {
            'JOHN SMITH': {'database_id': 'different-id', 'boss_name': 'JOHN SMITH', 'afterclass_name': 'Smith, John'}
        }
        service_with_cache.update_with_professor_lookup(lookup)

        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == original_id

    def test_handles_empty_lookup(self, service_with_cache):
        """Empty lookup should be a no-op."""
        service_with_cache.update_with_professor_lookup({})
        # Should still work normally
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == 'prof-1'

    def test_handles_none_lookup(self, service_with_cache):
        """None lookup should be a no-op."""
        service_with_cache.update_with_professor_lookup(None)
        result = service_with_cache.resolve_professor_name("JOHN SMITH")
        assert result == 'prof-1'


# ============================================================================
# resolve_professor_ids
# ============================================================================

class TestResolveProfessorIds:
    """Tests for multi-row professor resolution (used by ClassProcessor)."""

    def test_resolves_single_professor(self, service_with_cache):
        """Should resolve a single professor from multiple_rows."""
        rows = [{'professor_name': 'JOHN SMITH'}]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        assert len(result) == 1
        assert result[0][0] == 'prof-1'
        assert result[0][1] == 'JOHN SMITH'

    def test_resolves_multiple_professors(self, service_with_cache):
        """Should resolve multiple professors from different rows."""
        rows = [
            {'professor_name': 'JOHN SMITH'},
            {'professor_name': 'JANE DOE'}
        ]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        assert len(result) == 2
        prof_ids = [r[0] for r in result]
        assert 'prof-1' in prof_ids
        assert 'prof-2' in prof_ids

    def test_deduplicates_by_professor_id(self, service_with_cache):
        """Should deduplicate results by professor_id."""
        rows = [
            {'professor_name': 'JOHN SMITH'},
            {'professor_name': 'J. SMITH'}  # Same professor (alias)
        ]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        prof_ids = [r[0] for r in result]
        assert prof_ids.count('prof-1') == 1

    def test_skips_empty_professor_names(self, service_with_cache):
        """Should skip rows with empty/None professor names."""
        rows = [
            {'professor_name': 'JOHN SMITH'},
            {'professor_name': None},
            {'professor_name': ''},
            {'professor_name': 'TBA'}
        ]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        assert len(result) == 1
        assert result[0][0] == 'prof-1'

    def test_splits_comma_separated_names(self, service_with_cache):
        """Should split comma-separated multi-professor names."""
        rows = [{'professor_name': 'JOHN SMITH, JANE DOE'}]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        assert len(result) == 2

    def test_empty_rows_returns_empty(self, service_with_cache):
        """Empty rows list should return empty result."""
        result = service_with_cache.resolve_professor_ids('key-1', [])
        assert result == []

    def test_unresolvable_names_excluded(self, service_with_cache):
        """Names that can't be resolved should not appear in results."""
        rows = [
            {'professor_name': 'JOHN SMITH'},
            {'professor_name': 'COMPLETELY UNKNOWN PERSON'}
        ]
        result = service_with_cache.resolve_professor_ids('key-1', rows)

        assert len(result) == 1
        assert result[0][0] == 'prof-1'


# ============================================================================
# _parse_boss_aliases_set
# ============================================================================

class TestParseBossAliasesSet:
    """Tests for boss_aliases parsing in the resolution service."""

    def test_parses_list(self, service_with_cache):
        """Should parse list of aliases."""
        result = service_with_cache._parse_boss_aliases_set(["JOHN SMITH", "J. SMITH"])
        assert "JOHN SMITH" in result
        assert "J. SMITH" in result

    def test_parses_json_string(self, service_with_cache):
        """Should parse JSON array string."""
        result = service_with_cache._parse_boss_aliases_set('["JOHN SMITH", "J. SMITH"]')
        assert "JOHN SMITH" in result
        assert "J. SMITH" in result

    def test_parses_postgres_array_string(self, service_with_cache):
        """Should parse PostgreSQL array string format."""
        result = service_with_cache._parse_boss_aliases_set('{"JOHN SMITH", "J. SMITH"}')
        assert "JOHN SMITH" in result
        assert "J. SMITH" in result

    def test_returns_empty_for_none(self, service_with_cache):
        """Should return empty set for None."""
        result = service_with_cache._parse_boss_aliases_set(None)
        assert result == set()

    def test_returns_empty_for_empty_list(self, service_with_cache):
        """Should return empty set for empty list."""
        result = service_with_cache._parse_boss_aliases_set([])
        assert result == set()

    def test_uppercases_results(self, service_with_cache):
        """Should uppercase all parsed aliases."""
        result = service_with_cache._parse_boss_aliases_set(["john smith"])
        assert "JOHN SMITH" in result

    def test_strips_whitespace(self, service_with_cache):
        """Should strip whitespace from parsed aliases."""
        result = service_with_cache._parse_boss_aliases_set(["  JOHN SMITH  "])
        assert "JOHN SMITH" in result
