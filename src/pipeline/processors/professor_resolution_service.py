"""
ProfessorResolutionService - Single source of truth for professor name resolution.

This service centralizes all professor splitting and resolution logic that was
previously duplicated across ProfessorProcessor and ClassProcessor.

Responsibilities:
1. Split multi-professor names using greedy longest-match-first algorithm
2. Resolve professor names to IDs using 8-strategy resolution chain
3. Build lookups from both DB cache and session-created professors

Usage:
    service = ProfessorResolutionService(
        professors_cache=db_cache,
        new_professors=new_professor_dtos,
        updated_professors=updated_professor_dtos,
        logger=logger
    )

    # For class processing
    professor_mappings = service.resolve_professor_ids(
        record_key,
        multiple_rows  # rows from multiple_lookup.get(record_key, [])
    )
"""
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.pipeline.dtos.professor_dto import ProfessorDTO


class ProfessorResolutionService:
    """Centralized professor name resolution with 7-strategy lookup chain."""

    def __init__(
        self,
        professors_cache: Dict[str, Dict],
        new_professors: List[ProfessorDTO],
        updated_professors: List[ProfessorDTO],
        logger: Optional[logging.Logger] = None,
        valid_professor_ids: Optional[Set[str]] = None
    ):
        """
        Initialize service with combined lookups from DB cache and session DTOs.

        Args:
            professors_cache: Dict from DB cache, keyed by name_upper
            new_professors: ProfessorDTOs created this session
            updated_professors: ProfessorDTOs updated this session
            logger: Optional logger for debug messages
            valid_professor_ids: Optional set of valid professor IDs from DB for validation
        """
        self._logger = logger
        self._direct_lookup: Dict[str, str] = {}  # boss_name_upper -> professor_id
        self._full_lookup: Dict[str, Dict] = {}  # boss_name_upper -> {database_id, boss_name, afterclass_name}
        self._boss_alias_lookup: Dict[str, str] = {}  # alias_upper -> boss_name_upper
        self._new_professor_ids: Dict[str, str] = {}  # boss_name_upper -> professor_id (session only)
        self._valid_professor_ids = valid_professor_ids  # Set of valid IDs from DB for validation

        self._build_lookups(professors_cache, new_professors, updated_professors)

    def _build_lookups(
        self,
        professors_cache: Dict[str, Dict],
        new_professors: List[ProfessorDTO],
        updated_professors: List[ProfessorDTO]
    ) -> None:
        """Build all internal lookups from combined sources."""
        # 1. Populate from DB cache
        for name_upper, prof_data in professors_cache.items():
            prof_id = str(prof_data.get('id'))
            afterclass_name = prof_data.get('name', '')

            self._direct_lookup[name_upper] = prof_id
            self._full_lookup[name_upper] = {
                'database_id': prof_id,
                'boss_name': name_upper,
                'afterclass_name': afterclass_name
            }

            # Parse and store boss_aliases
            boss_aliases = prof_data.get('boss_aliases', [])
            alias_set = self._parse_boss_aliases_set(boss_aliases)
            for alias in alias_set:
                self._boss_alias_lookup[alias.upper()] = name_upper
                self._direct_lookup[alias.upper()] = prof_id

        # 2. Populate from new_professors (session-created)
        for dto in new_professors:
            prof_id = dto.id
            boss_name_upper = dto.name.upper()

            self._direct_lookup[boss_name_upper] = prof_id
            self._full_lookup[boss_name_upper] = {
                'database_id': prof_id,
                'boss_name': boss_name_upper,
                'afterclass_name': dto.name
            }
            self._new_professor_ids[boss_name_upper] = prof_id

            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                self._direct_lookup[alias_upper] = prof_id
                self._boss_alias_lookup[alias_upper] = boss_name_upper
                self._new_professor_ids[alias_upper] = prof_id

        # 3. Populate from updated_professors (session-updated)
        for dto in updated_professors:
            prof_id = dto.id
            boss_name_upper = dto.name.upper()

            self._direct_lookup[boss_name_upper] = prof_id
            self._full_lookup[boss_name_upper] = {
                'database_id': prof_id,
                'boss_name': boss_name_upper,
                'afterclass_name': dto.name
            }

            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                self._direct_lookup[alias_upper] = prof_id
                self._boss_alias_lookup[alias_upper] = boss_name_upper

    def update_with_session_professors(
        self,
        new_professors: List[ProfessorDTO],
        updated_professors: List[ProfessorDTO]
    ) -> None:
        """Add session-created professors to existing lookups.

        This allows updating the service after professors are created/updated
        during processing, enabling the service to be built early with DB cache
        only, then updated with session professors before class processing.
        """
        # Add new_professors (session-created)
        for dto in new_professors:
            prof_id = dto.id
            boss_name_upper = dto.name.upper()

            self._direct_lookup[boss_name_upper] = prof_id
            self._full_lookup[boss_name_upper] = {
                'database_id': prof_id,
                'boss_name': boss_name_upper,
                'afterclass_name': dto.name
            }
            self._new_professor_ids[boss_name_upper] = prof_id

            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                self._direct_lookup[alias_upper] = prof_id
                self._boss_alias_lookup[alias_upper] = boss_name_upper
                self._new_professor_ids[alias_upper] = prof_id

        # Add updated_professors (session-updated)
        # Only add NEW aliases/entries — do NOT overwrite existing _direct_lookup
        # entries, since they already have the correct DB professor_id.
        for dto in updated_professors:
            prof_id = dto.id
            boss_name_upper = dto.name.upper()

            # Only update full_lookup if not already present
            if boss_name_upper not in self._full_lookup:
                self._full_lookup[boss_name_upper] = {
                    'database_id': prof_id,
                    'boss_name': boss_name_upper,
                    'afterclass_name': dto.name
                }

            for alias in dto.boss_aliases:
                alias_upper = alias.upper()
                # Only add alias if not already in direct_lookup (preserve existing correct mapping)
                if alias_upper not in self._direct_lookup:
                    self._direct_lookup[alias_upper] = prof_id
                    self._boss_alias_lookup[alias_upper] = boss_name_upper

    def update_with_professor_lookup(self, professor_lookup: Dict[str, Dict]) -> None:
        """Merge an external professor_lookup dictionary into our direct_lookup.

        This is called by ProfessorProcessor after processing to share all the
        discovered name variations with the resolution service before class processing.

        Args:
            professor_lookup: Dictionary mapping normalized_name -> lookup_data
                              (same format as V4's professor_lookup)
        """
        if not professor_lookup:
            return

        merged_count = 0
        for normalized_name, lookup_data in professor_lookup.items():
            # Only add if not already present
            if normalized_name not in self._direct_lookup:
                prof_id = lookup_data.get('database_id') or lookup_data.get('id')
                if prof_id:
                    self._direct_lookup[normalized_name] = str(prof_id)
                    merged_count += 1

        if self._logger and merged_count > 0:
            self._logger.info(f"Merged {merged_count} entries from professor_lookup into resolution service")

    def _parse_boss_aliases_set(self, boss_aliases) -> Set[str]:
        """Parse boss_aliases from DB cache into a set of clean strings."""
        if not boss_aliases:
            return set()
        if isinstance(boss_aliases, list):
            return {str(a).strip().upper() for a in boss_aliases if a and str(a).strip()}
        if isinstance(boss_aliases, str):
            if boss_aliases.startswith('{') and boss_aliases.endswith('}'):
                content = boss_aliases[1:-1]
                return {item.strip().strip('"').upper() for item in content.split(',') if item.strip()}
            import json
            try:
                parsed = json.loads(boss_aliases)
                if isinstance(parsed, list):
                    return {str(a).strip().upper() for a in parsed if a and str(a).strip()}
            except:
                pass
        return set()

    def resolve_professor_ids(
        self,
        record_key: str,
        multiple_rows: List[dict]
    ) -> List[Tuple[str, str]]:
        """
        Resolve all professor IDs for a record_key from multiple sheet rows.

        Args:
            record_key: The record_key to look up
            multiple_rows: List of rows from multiple_lookup.get(record_key, [])

        Returns:
            List[(professor_id, original_scraped_name)], deduplicated by professor_id
        """
        professor_mappings = []
        seen_professor_ids = set()

        for row in multiple_rows:
            prof_name_raw = row.get('professor_name')
            if not prof_name_raw or pd.isna(prof_name_raw):
                continue

            prof_name_str = str(prof_name_raw).strip()
            split_names = self.split_professor_names(prof_name_str)

            for split_name in split_names:
                prof_id = self.resolve_professor_name(split_name)
                if prof_id and prof_id not in seen_professor_ids:
                    professor_mappings.append((prof_id, split_name))
                    seen_professor_ids.add(prof_id)

        return professor_mappings

    def _is_valid_professor_id(self, professor_id: str) -> bool:
        """Check if a professor ID is valid (exists in DB or was created this session)."""
        # Session-created professors are always valid
        if professor_id in self._new_professor_ids.values():
            return True
        if not self._valid_professor_ids:
            # If no validation set provided, accept all IDs
            return True
        return professor_id in self._valid_professor_ids

    def resolve_professor_name(self, name: str) -> Optional[str]:
        """
        Resolve a single professor name to professor_id using 7-strategy chain.

        Returns:
            professor_id or None if not found

        Strategy 1: Direct lookup in _direct_lookup
        Strategy 2: Variation lookup (remove commas, normalize spaces)
        Strategy 3: _full_lookup database_id match
        Strategy 4: Boss aliases lookup via _boss_alias_lookup
        Strategy 5: Subset matching (partial word match, >=2 words)
        Strategy 6: New professors match (session deduplication via _new_professor_ids)
        Strategy 7: Return None if no match
        """
        if not name:
            return None

        name_str = str(name).strip()
        if not name_str or name_str.lower() in ['nan', 'tba', 'to be announced']:
            return None

        normalized_name = name_str.upper()
        variations = self._generate_variations(normalized_name)

        # Strategy 1: Direct lookup in _direct_lookup
        if normalized_name in self._direct_lookup:
            prof_id = self._direct_lookup[normalized_name]
            if self._is_valid_professor_id(prof_id):
                return prof_id

        # Strategy 2: Boss name lookup via variations
        for var in variations:
            if var in self._direct_lookup:
                prof_id = self._direct_lookup[var]
                if self._is_valid_professor_id(prof_id):
                    return prof_id

        # Strategy 3: Check _full_lookup for database_id matches
        for var in variations:
            if var in self._full_lookup:
                prof_id = self._full_lookup[var]['database_id']
                if self._is_valid_professor_id(prof_id):
                    return prof_id

        # Strategy 4: Boss aliases from _boss_alias_lookup
        for var in variations:
            if var in self._boss_alias_lookup:
                boss_name_key = self._boss_alias_lookup[var]
                if boss_name_key in self._full_lookup:
                    prof_id = self._full_lookup[boss_name_key]['database_id']
                    if self._is_valid_professor_id(prof_id):
                        return prof_id

        # Strategy 5: Subset matching (partial word match, requires >=2 words)
        search_words = set(normalized_name.replace(',', ' ').split())
        if len(search_words) >= 2:
            for lookup_boss_name, lookup_data in self._full_lookup.items():
                if lookup_boss_name == normalized_name:
                    continue
                lookup_words = set(lookup_boss_name.replace(',', ' ').split())
                if search_words.issubset(lookup_words):
                    prof_id = lookup_data['database_id']
                    if self._is_valid_professor_id(prof_id):
                        return prof_id

        # Strategy 6: New professors match (session deduplication)
        if normalized_name in self._new_professor_ids:
            prof_id = self._new_professor_ids[normalized_name]
            if self._is_valid_professor_id(prof_id):
                return prof_id
        for var in variations:
            if var in self._new_professor_ids:
                prof_id = self._new_professor_ids[var]
                if self._is_valid_professor_id(prof_id):
                    return prof_id

        # Strategy 7: No match found
        return None

    def _generate_variations(self, normalized_name: str) -> List[str]:
        """Generate name variations for lookup attempts."""
        variations = [
            normalized_name,
            normalized_name.replace(',', ''),
            ' '.join(normalized_name.replace(',', ' ').split()),
            normalized_name.replace(',', '').replace(' ', ''),
        ]
        if ',' not in normalized_name:
            parts = normalized_name.split()
            if len(parts) >= 2:
                comma_variation = f"{parts[0]}, {' '.join(parts[1:])}"
                variations.append(comma_variation)
                if len(parts) >= 3:
                    comma_variation2 = f"{parts[0]} {parts[1]}, {' '.join(parts[2:])}"
                    variations.append(comma_variation2)
        return variations

    def split_professor_names(self, name: str) -> List[str]:
        """
        Split professor names using greedy longest-match-first algorithm.

        Uses internal _direct_lookup to identify known professors. Unknown single-word
        parts are combined with the previous known professor, not treated as standalone.

        Handles multi-professor names like:
        "ZHANG WEI, NYDIA REMOLINA LEON, AURELIO GURREA MARTINEZ"

        Algorithm:
        1. If entire string is a known professor, return as-is
        2. If no commas, return as-is (single professor)
        3. Split by commas, greedily match longest combinations against _direct_lookup
        4. Unknown single-word parts combine with previous known professor
        """
        if not name:
            return []

        # DEBUG: Log lookup size for troubleshooting
        if self._logger and len(name) > 3:
            self._logger.debug(f"split_professor_names: name='{name}', lookup_size={len(self._direct_lookup)}")

        name_str = str(name).strip()
        if not name_str:
            return []

        # 1. Quick return: if entire string is a known professor, return as-is
        if name_str.upper() in self._direct_lookup:
            return [name_str]

        # 2. No commas means single professor
        if ',' not in name_str:
            return [name_str] if name_str else []

        # 3. Split by commas and greedily match
        parts = [p.strip() for p in name_str.split(',') if p.strip()]
        found_professors = []
        i = 0

        while i < len(parts):
            match_found = False

            # Try longest combination first
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])
                if candidate.upper() in self._direct_lookup:
                    found_professors.append(candidate)
                    i = j
                    match_found = True
                    break

            # 4. No match found - handle unknown parts
            if not match_found:
                unknown_part = parts[i]
                if found_professors and len(unknown_part.split()) == 1:
                    # Single-word unknown: combine with previous professor
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                else:
                    # Multi-word unknown: treat as standalone
                    found_professors.append(unknown_part)
                i += 1

        return found_professors