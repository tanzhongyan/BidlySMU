"""
ClassProcessor - handles class CREATE vs UPDATE logic with proper professor transition handling.

Refactored to support professor transitions (0→1, 1→0, 1→N, swaps) without creating duplicates.
Uses group-based reconciliation: existing vs incoming state compared at (acad_term_id, boss_id) level.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.pipeline.dtos.class_dto import ClassDTO
from src.pipeline.dtos.course_dto import CourseDTO
from src.pipeline.processors.professor_resolution_service import ProfessorResolutionService


class ClassProcessor:
    """Processes class records with proper professor transition handling."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        multiple_lookup: Dict[str, List[dict]],
        course_lookup: Dict[str, 'CourseDTO'],
        professor_resolution_service: 'ProfessorResolutionService',
        existing_classes_cache: List[dict],
        logger: Optional[object] = None
    ):
        self._raw_data = raw_data
        self._multiple_lookup = multiple_lookup
        self._course_lookup = course_lookup
        self._professor_resolution_service = professor_resolution_service
        self._existing_classes_cache = existing_classes_cache
        self._logger = logger

        # New group-based data structures for reconciliation
        # {(acad_term_id, boss_id): [class_records]}
        self._existing_classes_by_group: Dict[Tuple[str, int], List[dict]] = {}
        # {(acad_term_id, boss_id): [(prof_id, prof_name), ...]}
        self._incoming_state_by_group: Dict[Tuple[str, int], List[Tuple[Optional[str], str]]] = {}
        # {(acad_term_id, boss_id): set(prof_ids)} - dedup tracker for incoming state
        self._incoming_prof_ids_by_group: Dict[Tuple[str, int], Set[Optional[str]]] = {}
        # {(acad_term_id, boss_id)} - groups with multi-professor classes
        self._multi_professor_groups: Set[Tuple[str, int]] = set()
        # {(acad_term_id, boss_id): row} - pre-built index for O(1) row lookup
        self._raw_data_by_group: Dict[Tuple[str, int], dict] = {}
        # Classes that should be soft-deactivated (excess old records)
        self._classes_to_deactivate: List[dict] = []

        # Output collections
        self._new_classes: List['ClassDTO'] = []
        self._updated_classes: List['ClassDTO'] = []
        self._record_key_to_class_ids: Dict[str, List[str]] = {}

    def process(self) -> Tuple[List['ClassDTO'], List['ClassDTO']]:
        """Execute class processing with professor transition handling."""
        self._logger.info("Processing classes with professor transition handling...")

        # Phase 1: Build existing state grouped by (acad_term_id, boss_id)
        self._build_existing_group_lookup()

        # Phase 2: Build incoming state and identify multi-professor groups
        self._build_incoming_state()

        # Phase 3: Reconcile existing vs incoming state for each group
        self._reconcile_all_groups()

        self._logger.info(f"Class processing complete. New: {len(self._new_classes)}, Updates: {len(self._updated_classes)}, To Deactivate: {len(self._classes_to_deactivate)}")

        return self._new_classes, self._updated_classes

    def get_record_key_to_class_ids_mapping(self) -> Dict[str, List[str]]:
        """Return the record_key -> [class_ids] mapping built during processing."""
        return self._record_key_to_class_ids

    def get_classes_to_deactivate(self) -> List[dict]:
        """Return list of class records that should be deactivated (excess records)."""
        return self._classes_to_deactivate

    # ============================================================================
    # Phase 1: Build Existing State
    # ============================================================================

    def _build_existing_group_lookup(self) -> None:
        """Group existing classes by (acad_term_id, boss_id) for reconciliation."""
        for c in self._existing_classes_cache:
            acad_term_id = c.get('acad_term_id')
            boss_id = c.get('boss_id')
            if acad_term_id and boss_id is not None:
                key = (acad_term_id, int(boss_id))
                if key not in self._existing_classes_by_group:
                    self._existing_classes_by_group[key] = []
                self._existing_classes_by_group[key].append(c)

        self._logger.info(f"Built existing state: {len(self._existing_classes_by_group)} groups, {len(self._existing_classes_cache)} total classes")

    # ============================================================================
    # Phase 2: Build Incoming State
    # ============================================================================

    def _build_incoming_state(self) -> None:
        """Build incoming state map, raw_data index, and identify multi-professor groups."""
        for _, row in self._raw_data.iterrows():
            acad_term_id = row.get('acad_term_id')
            class_boss_id = row.get('class_boss_id')
            record_key = row.get('record_key')

            if pd.isna(acad_term_id) or pd.isna(class_boss_id) or not record_key:
                continue

            group_key = (acad_term_id, int(class_boss_id))

            # Build raw_data index for O(1) lookup during reconciliation
            if group_key not in self._raw_data_by_group:
                self._raw_data_by_group[group_key] = row

            professor_mappings = self._professor_resolution_service.resolve_professor_ids(
                record_key,
                self._multiple_lookup.get(record_key, [])
            ) if record_key else []

            # Check for multi-professor
            if len(professor_mappings) > 1:
                self._multi_professor_groups.add(group_key)

            # Store incoming state (deduplicate by professor_id within group)
            if group_key not in self._incoming_state_by_group:
                self._incoming_state_by_group[group_key] = []
                self._incoming_prof_ids_by_group[group_key] = set()

            # Add professor mappings (or TBA if empty), skipping duplicates
            if professor_mappings:
                for prof_id, prof_name in professor_mappings:
                    if prof_id not in self._incoming_prof_ids_by_group[group_key]:
                        self._incoming_state_by_group[group_key].append((prof_id, prof_name))
                        self._incoming_prof_ids_by_group[group_key].add(prof_id)
            elif None not in self._incoming_prof_ids_by_group[group_key]:
                self._incoming_state_by_group[group_key].append((None, ''))
                self._incoming_prof_ids_by_group[group_key].add(None)

        self._logger.info(f"Built incoming state: {len(self._incoming_state_by_group)} groups, {len(self._multi_professor_groups)} multi-professor groups")

    # ============================================================================
    # Phase 3: Reconcile All Groups
    # ============================================================================

    def _reconcile_all_groups(self) -> None:
        """Reconcile existing vs incoming state for each group.

        Only iterates over groups with incoming data. Existing-only groups
        (no raw_data) are left untouched per design — they belong to previous
        terms that aren't being processed this run.
        """
        for group_key in self._incoming_state_by_group:
            existing_classes = self._existing_classes_by_group.get(group_key, [])
            incoming_profs = self._incoming_state_by_group.get(group_key, [])

            self._reconcile_single_group(group_key, existing_classes, incoming_profs)

    def _reconcile_single_group(
        self,
        group_key: Tuple[str, int],
        existing_classes: List[dict],
        incoming_profs: List[Tuple[Optional[str], str]]
    ) -> None:
        """
        Reconcile a single group's existing state with incoming state.

        Logic:
        1. Match existing classes to incoming professors (1-to-1 matching)
        2. UPDATE matched classes with new professor assignments
        3. CREATE new classes for unmatched incoming professors
        4. Mark excess existing classes for deactivation
        """
        acad_term_id, boss_id = group_key
        warn_inaccuracy = group_key in self._multi_professor_groups

        # Get raw data for this group (need at least one row for context)
        row = self._get_sample_row_for_group(acad_term_id, boss_id)
        if row is None:
            self._logger.warning(f"No raw data found for group {group_key}, skipping")
            return

        course_code = row.get('course_code')
        course_dto = self._course_lookup.get(course_code)
        if not course_dto:
            self._logger.warning(f"Course not found for group {group_key}, skipping")
            return

        # Build mapping for O(1) lookup
        existing_by_prof: Dict[Optional[str], dict] = {c.get('professor_id'): c for c in existing_classes}

        # Track which existing classes we've matched to avoid reuse
        matched_existing_ids: Set[str] = set()

        # Handle empty incoming state: if no professors, treat as TBA (single None entry)
        # This ensures 1 existing class is repurposed with professor_id=None rather than
        # all being marked for deactivation (handles N→0 professor transitions).
        working_incoming_profs = incoming_profs if incoming_profs else [(None, '')]

        # Process incoming professors in order
        for idx, (prof_id, prof_name) in enumerate(working_incoming_profs):
            existing_class = existing_by_prof.get(prof_id)

            if existing_class and existing_class['id'] not in matched_existing_ids:
                # Case: Existing class with this professor - update it
                matched_existing_ids.add(existing_class['id'])
                self._process_update(
                    existing_class=existing_class,
                    row=row,
                    incoming_section=str(row.get('section')),
                    warn_inaccuracy=warn_inaccuracy,
                    record_key=row.get('record_key'),
                    prof_id=prof_id
                )
            else:
                # Need to find an available existing class to repurpose
                # Priority: 1) Unmatched TBA (professor_id=None), 2) Unmatched professor being removed
                available_existing = None

                # First: Look for unmatched TBA class
                for c in existing_classes:
                    if c['id'] not in matched_existing_ids and c.get('professor_id') is None:
                        available_existing = c
                        break

                # Second: Look for any unmatched existing class to repurpose
                if not available_existing:
                    for c in existing_classes:
                        if c['id'] not in matched_existing_ids:
                            available_existing = c
                            break

                if available_existing:
                    # Repurpose this existing class
                    matched_existing_ids.add(available_existing['id'])
                    self._process_update(
                        existing_class=available_existing,
                        row=row,
                        incoming_section=str(row.get('section')),
                        warn_inaccuracy=warn_inaccuracy,
                        record_key=row.get('record_key'),
                        prof_id=prof_id
                    )
                else:
                    # No available existing class - CREATE new
                    self._process_create(
                        row=row,
                        acad_term_id=acad_term_id,
                        class_boss_id=boss_id,
                        course_id=course_dto.id,
                        section=str(row.get('section')),
                        prof_id=prof_id,
                        prof_name=prof_name,
                        warn_inaccuracy=warn_inaccuracy,
                        record_key=row.get('record_key')
                    )

        # Mark any unmatched existing classes for deactivation
        for c in existing_classes:
            if c['id'] not in matched_existing_ids:
                self._classes_to_deactivate.append(c)
                self._logger.info(
                    f"Marking class for deactivation: {c['id']} (professor_id={c.get('professor_id')}, "
                    f"acad_term_id={c.get('acad_term_id')}, boss_id={c.get('boss_id')})"
                )

    def _get_sample_row_for_group(self, acad_term_id: str, boss_id: int) -> Optional[dict]:
        """Get a sample row from pre-built index for O(1) lookup."""
        return self._raw_data_by_group.get((acad_term_id, boss_id))

    def _process_update(self, existing_class: dict, row: dict, incoming_section: str, warn_inaccuracy: bool, record_key: str = None, prof_id: Optional[str] = None) -> None:
        """Handle UPDATE case for existing class."""
        needs_update = False
        update_data = {'id': existing_class['id']}

        # Check for professor_id change (this is now supported!)
        if prof_id is not None and prof_id != existing_class.get('professor_id'):
            update_data['professor_id'] = prof_id
            needs_update = True
        elif prof_id is None and existing_class.get('professor_id') is not None:
            # Transition to TBA (None)
            update_data['professor_id'] = None
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

        # Always update warn_inaccuracy to ensure consistency within group
        if existing_class.get('warn_inaccuracy') != warn_inaccuracy:
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

        if record_key and new_dto.id:
            if record_key not in self._record_key_to_class_ids:
                self._record_key_to_class_ids[record_key] = []
            if new_dto.id not in self._record_key_to_class_ids[record_key]:
                self._record_key_to_class_ids[record_key].append(new_dto.id)

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
