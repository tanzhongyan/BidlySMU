"""
ClassProcessor - handles class CREATE vs UPDATE logic.
Refactored to pure function pattern with explicit parameters.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.course_dto import CourseDTO


class ClassProcessor:
    """Processes class records from standalone data with CREATE/UPDATE."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        multiple_lookup: Dict[str, List[dict]],
        course_lookup: Dict[str, 'CourseDTO'],
        professor_lookup: Dict[str, str],
        existing_classes_cache: List[dict],
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._multiple_lookup = multiple_lookup
        self._course_lookup = course_lookup
        self._professor_lookup = professor_lookup
        self._existing_classes_cache = existing_classes_cache
        self._logger = logger

        self._existing_class_lookup: Dict[Tuple, dict] = {}
        self._processed_class_keys: Set[Tuple] = set()
        self._new_classes: List['ClassDTO'] = []
        self._updated_classes: List['ClassDTO'] = []
        # Track record_key -> [class_ids] mapping for timing processing
        self._record_key_to_class_ids: Dict[str, List[str]] = {}

    def process(self) -> Tuple[List['ClassDTO'], List['ClassDTO']]:
        """Execute class processing logic. Returns (new_classes, updated_classes)."""
        self._build_existing_lookup()
        self._process_all_rows()
        return self._new_classes, self._updated_classes

    def get_record_key_to_class_ids_mapping(self) -> Dict[str, List[str]]:
        """Return the record_key -> [class_ids] mapping built during processing."""
        return self._record_key_to_class_ids

    def _build_existing_lookup(self) -> None:
        """Build existing_class_lookup from cache for O(1) lookups."""
        for c in self._existing_classes_cache:
            acad_term_id = c.get('acad_term_id')
            boss_id = c.get('boss_id')
            professor_id = c.get('professor_id')
            if acad_term_id and boss_id is not None:
                key = (acad_term_id, boss_id, professor_id)
                self._existing_class_lookup[key] = c

    def _process_all_rows(self) -> None:
        """Process all rows in raw_data."""
        self._logger.info("Processing classes with CREATE vs UPDATE logic...")

        for _, row in self._raw_data.iterrows():
            try:
                self._process_row(row)
            except Exception as e:
                self._logger.info(f"Exception processing class row: {e}")

        self._logger.info(f"Class processing complete. New: {len(self._new_classes)}, Updates: {len(self._updated_classes)}.")

    def _process_row(self, row: dict) -> None:
        """Process a single row of class data."""
        acad_term_id = row.get('acad_term_id')
        class_boss_id = row.get('class_boss_id')
        course_code = row.get('course_code')
        section = str(row.get('section'))

        if pd.isna(acad_term_id) or pd.isna(class_boss_id):
            return

        course_dto = self._course_lookup.get(course_code)
        if not course_dto:
            return

        record_key = row.get('record_key')
        professor_mappings = self._find_professors_for_class(record_key) if record_key else []

        if not professor_mappings:
            professor_mappings = [(None, '')]
        else:
            # Debug: log when professor is found
            self._logger.debug(f"_find_professors_for_class({record_key}) -> {professor_mappings}")

        is_multi_professor = len(professor_mappings) > 1
        warn_inaccuracy = is_multi_professor

        for prof_id, prof_name in professor_mappings:
            class_key = (acad_term_id, class_boss_id, prof_id)

            if class_key in self._processed_class_keys:
                continue
            self._processed_class_keys.add(class_key)

            existing_class = self._existing_class_lookup.get(class_key)

            if existing_class:
                self._process_update(existing_class, row, section, warn_inaccuracy, record_key, prof_id)
            else:
                self._process_create(row, acad_term_id, class_boss_id, course_dto.id, section, prof_id, prof_name, warn_inaccuracy, record_key)

    def _process_update(self, existing_class: dict, row: dict, incoming_section: str, warn_inaccuracy: bool, record_key: str = None, prof_id: Optional[str] = None) -> None:
        """Handle UPDATE case for existing class."""
        needs_update = False
        update_data = {'id': existing_class['id']}

        # Check if professor_id needs to be updated
        # Only update if prof_id is provided and differs from existing
        if prof_id is not None and prof_id != existing_class.get('professor_id'):
            update_data['professor_id'] = prof_id
            needs_update = True

        fields_to_check = {
            'grading_basis': row.get('grading_basis') if not pd.isna(row.get('grading_basis')) else None,
            'course_outline_url': row.get('course_outline_url'),
            'boss_id': int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None,
        }

        for field, new_value in fields_to_check.items():
            old_value = existing_class.get(field)
            new_value, old_value, changed = self._compare_values(new_value, old_value)
            if changed:
                update_data[field] = new_value
                needs_update = True

        existing_section = existing_class.get('section')
        if existing_section != incoming_section:
            update_data['section'] = incoming_section
            needs_update = True

        if needs_update:
            now = datetime.now()
            updated_dto = ClassDTO(
                id=existing_class['id'],
                section=update_data.get('section', existing_class.get('section')),
                course_id=existing_class.get('course_id'),
                professor_id=update_data.get('professor_id', existing_class.get('professor_id')),
                acad_term_id=existing_class.get('acad_term_id'),
                grading_basis=update_data.get('grading_basis', existing_class.get('grading_basis')),
                course_outline_url=update_data.get('course_outline_url', existing_class.get('course_outline_url')),
                boss_id=update_data.get('boss_id', existing_class.get('boss_id')),
                warn_inaccuracy=warn_inaccuracy,
                created_at=existing_class.get('created_at'),
                updated_at=now
            )
            self._updated_classes.append(updated_dto)

        # Map record_key to existing class ID for timing processing
        if record_key and existing_class.get('id'):
            if record_key not in self._record_key_to_class_ids:
                self._record_key_to_class_ids[record_key] = []
            if existing_class['id'] not in self._record_key_to_class_ids[record_key]:
                self._record_key_to_class_ids[record_key].append(existing_class['id'])

    def _process_create(self, row: dict, acad_term_id: str, class_boss_id: int, course_id: str, section: str, prof_id: Optional[str], prof_name: str, warn_inaccuracy: bool, record_key: str = None) -> None:
        """Handle CREATE case for new class."""
        grading_basis_val = row.get('grading_basis')
        if pd.isna(grading_basis_val):
            grading_basis_val = None

        now = datetime.now()

        new_dto = ClassDTO(
            id=str(uuid.uuid4()),
            section=section,
            course_id=course_id,
            professor_id=prof_id,
            acad_term_id=acad_term_id,
            grading_basis=grading_basis_val,
            course_outline_url=row.get('course_outline_url') if pd.notna(row.get('course_outline_url')) else None,
            boss_id=int(class_boss_id) if class_boss_id is not None else None,
            warn_inaccuracy=warn_inaccuracy,
            created_at=now,
            updated_at=now
        )
        self._new_classes.append(new_dto)

        # Map record_key to new class ID for timing processing
        if record_key and new_dto.id:
            if record_key not in self._record_key_to_class_ids:
                self._record_key_to_class_ids[record_key] = []
            if new_dto.id not in self._record_key_to_class_ids[record_key]:
                self._record_key_to_class_ids[record_key].append(new_dto.id)

    def _split_professor_names(self, name: str) -> List[str]:
        """Split professor names using greedy longest-match-first algorithm.

        Uses professor_lookup to identify known professors. Unknown single-word
        parts are combined with the previous known professor, not treated as standalone.
        This handles multi-professor names like "ZHANG WEI, NYDIA REMOLINA LEON, AURELIO GURREA MARTINEZ".
        """
        if not name:
            return []

        name_str = str(name).strip()

        # Quick return: if entire string is a known professor, return as-is
        if name_str.upper() in self._professor_lookup:
            return [name_str]

        # No commas means single professor
        if ',' not in name_str:
            return [name_str] if name_str else []

        parts = [p.strip() for p in name_str.split(',') if p.strip()]

        found_professors = []
        i = 0

        while i < len(parts):
            match_found = False

            # Try longest combination first (greedy matching)
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])
                if candidate.upper() in self._professor_lookup:
                    found_professors.append(candidate)
                    i = j
                    match_found = True
                    break

            # No match found - handle unknown parts
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

    def _find_professors_for_class(self, record_key: str) -> List[Tuple]:
        """Find professor IDs for a class - uses splitting for multi-professor names."""
        if not record_key or pd.isna(record_key):
            return []

        rows = self._multiple_lookup.get(record_key, [])
        professor_mappings = []
        seen_professor_ids = set()

        for row in rows:
            prof_name_raw = row.get('professor_name')
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue

            prof_name = str(prof_name_raw).strip()
            if not prof_name or prof_name.lower() == 'nan':
                continue

            # Split multi-professor names (e.g., "ZHANG WEI, NYDIA REMOLINA LEON, AURELIO GURREA MARTINEZ")
            split_names = self._split_professor_names(prof_name)

            for split_name in split_names:
                # Try direct lookup first, then variation-based lookup
                normalized = split_name.upper()
                prof_id = None

                if normalized in self._professor_lookup:
                    prof_id = self._professor_lookup[normalized]
                    self._logger.info(f"DEBUG: DIRECT MATCH '{normalized}' -> {prof_id}")
                else:
                    # Try variations (remove commas, normalize spaces, etc.)
                    variations = [
                        normalized,
                        normalized.replace(',', ''),
                        ' '.join(normalized.replace(',', ' ').split()),
                        normalized.replace(',', '').replace(' ', ''),
                    ]
                    # If no comma in normalized, try inserting comma after first word
                    # This handles "GOH JING RONG" -> "GOH, JING RONG" lookup
                    if ',' not in normalized:
                        parts = normalized.split()
                        if len(parts) >= 2:
                            # Try comma after first word: "GOH JING RONG" -> "GOH, JING RONG"
                            comma_variation = f"{parts[0]}, {' '.join(parts[1:])}"
                            variations.append(comma_variation)
                            # Also try after second word if first is short
                            if len(parts) >= 3:
                                comma_variation2 = f"{parts[0]} {parts[1]}, {' '.join(parts[2:])}"
                                variations.append(comma_variation2)

                    for variation in variations:
                        if variation in self._professor_lookup:
                            prof_id = self._professor_lookup[variation]
                            self._logger.info(f"DEBUG: VARIATION MATCH '{normalized}' -> '{variation}' -> {prof_id}")
                            break

                if prof_id and prof_id not in seen_professor_ids:
                    professor_mappings.append((prof_id, split_name))
                    seen_professor_ids.add(prof_id)

        return professor_mappings

    def _compare_values(self, new_value, old_value):
        """Compare values handling numpy types."""
        if hasattr(old_value, '__iter__') and not isinstance(old_value, str):
            try:
                if hasattr(old_value, 'item'):
                    old_value = old_value.item()
                elif hasattr(old_value, '__len__') and len(old_value) > 0:
                    old_value = old_value[0]
                else:
                    old_value = None
            except:
                old_value = None

        if hasattr(new_value, '__iter__') and not isinstance(new_value, str):
            try:
                if hasattr(new_value, 'item'):
                    new_value = new_value.item()
                elif hasattr(new_value, '__len__') and len(new_value) > 0:
                    new_value = new_value[0]
                else:
                    new_value = None
            except:
                new_value = None

        try:
            if pd.isna(new_value) and pd.isna(old_value):
                return new_value, old_value, False
            elif pd.isna(new_value) or pd.isna(old_value):
                return new_value, old_value, pd.notna(new_value)
            new_val_str = str(new_value).strip()
            old_val_str = str(old_value).strip()
            return new_value, old_value, new_val_str != old_val_str
        except:
            return new_value, old_value, pd.notna(new_value)

    