"""
ClassProcessor - handles class CREATE vs UPDATE logic with TBA conversion.
Extracted from table_builder.py process_classes method.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Set
import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext


class ClassProcessor(AbstractProcessor):
    """Processes class records from standalone data with CREATE/UPDATE/TBA conversion."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)
        self._existing_class_lookup = {}
        self._processed_class_keys: Set = set()
        self._processed_update_class_ids: Set = set()

    def _load_cache(self) -> None:
        # Build existing_class_lookup from existing_classes_cache
        if hasattr(self.context, 'existing_classes_cache') and self.context.existing_classes_cache:
            for c in self.context.existing_classes_cache:
                acad_term_id = c.get('acad_term_id')
                class_boss_id = c.get('boss_id')
                professor_id = c.get('professor_id')
                if acad_term_id and class_boss_id is not None:
                    key = (acad_term_id, class_boss_id, professor_id)
                    self._existing_class_lookup[key] = c

    def _do_process(self) -> None:
        """Execute class processing logic."""
        self._logger.info("Processing classes with robust CREATE vs. UPDATE logic...")

        for idx, row in self.context.standalone_data.iterrows():
            try:
                self._process_row(row, idx)
            except Exception as e:
                self._logger.error(f"Exception processing class row {idx}: {e}")

        self._logger.info(f"Class processing complete. New: {self.context.stats['classes_created']}, Updates: {len(self.context.update_classes)}.")

    def _process_row(self, row, idx: int) -> None:
        """Process a single row of class data."""
        acad_term_id = row.get('acad_term_id')
        class_boss_id = row.get('class_boss_id')
        course_code = row.get('course_code')
        section = str(row.get('section'))

        if pd.isna(acad_term_id) or pd.isna(class_boss_id):
            return

        course_id = self._get_course_id(course_code)
        if not course_id:
            return

        record_key = row.get('record_key')
        professor_mappings = self._find_professors_for_class(record_key) if record_key else []

        # Handle TBA class getting a professor assigned
        self._handle_tba_conversion(row, acad_term_id, class_boss_id, course_id, section, record_key, professor_mappings)

        # If no professors found, create one class with professor_id = None
        if not professor_mappings:
            professor_mappings = [(None, '')]

        # Check if multi-professor
        is_multi_professor = len(professor_mappings) > 1
        warn_inaccuracy = is_multi_professor

        # Process each professor
        for prof_id, prof_name in professor_mappings:
            class_key = (acad_term_id, class_boss_id, prof_id)

            if class_key in self._processed_class_keys:
                continue
            self._processed_class_keys.add(class_key)

            existing_class = self._existing_class_lookup.get(class_key)

            if existing_class:
                self._process_update(existing_class, row, record_key, warn_inaccuracy)
            else:
                self._process_create(row, acad_term_id, class_boss_id, course_id, section, prof_id, prof_name, record_key, warn_inaccuracy)

    def _handle_tba_conversion(self, row, acad_term_id, class_boss_id, course_id, section, record_key, professor_mappings) -> None:
        """Handle TBA class getting professor assigned - converts via UPDATE."""
        class_rows_in_multiple = [r for r in self.context.multiple_lookup.get(record_key, []) if r.get('type') == 'CLASS']

        if len(professor_mappings) == 1 and len(class_rows_in_multiple) == 1:
            new_prof_id = professor_mappings[0][0]

            tba_class_to_update = None
            if hasattr(self.context, 'existing_classes_cache') and self.context.existing_classes_cache:
                for existing_class in self.context.existing_classes_cache:
                    if (existing_class.get('course_id') == course_id and
                        str(existing_class.get('section')) == section and
                        existing_class.get('acad_term_id') == acad_term_id and
                        pd.isna(existing_class.get('professor_id'))):
                        tba_class_to_update = existing_class
                        break

            if tba_class_to_update and tba_class_to_update['id'] not in self._processed_update_class_ids:
                self._logger.info(f"Converting TBA class {row.get('course_code')}-{section} to assigned.")
                self._processed_update_class_ids.add(tba_class_to_update['id'])

                update_record = {
                    'id': tba_class_to_update['id'],
                    'professor_id': new_prof_id
                }
                self.context.update_classes.append(update_record)

                if record_key:
                    if record_key not in self.context.class_id_mapping:
                        self.context.class_id_mapping[record_key] = []
                    if tba_class_to_update['id'] not in self.context.class_id_mapping[record_key]:
                        self.context.class_id_mapping[record_key].append(tba_class_to_update['id'])

                new_assigned_key = (acad_term_id, class_boss_id, new_prof_id)
                old_tba_key = (acad_term_id, class_boss_id, None)

                updated_class_record = tba_class_to_update.copy()
                updated_class_record['professor_id'] = new_prof_id
                self._existing_class_lookup[new_assigned_key] = updated_class_record

                if old_tba_key in self._existing_class_lookup:
                    del self._existing_class_lookup[old_tba_key]

                class_key_for_processing = (acad_term_id, class_boss_id, new_prof_id)
                self._processed_class_keys.add(class_key_for_processing)

    def _process_update(self, existing_class: Dict, row, record_key: str, warn_inaccuracy: bool) -> None:
        """Handle UPDATE case for existing class."""
        update_record = {'id': existing_class['id']}
        needs_update = False

        fields_to_check = {
            'grading_basis': row.get('grading_basis') if not pd.isna(row.get('grading_basis')) else None,
            'course_outline_url': row.get('course_outline_url'),
            'boss_id': int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None,
            'warn_inaccuracy': warn_inaccuracy
        }

        for field, new_value in fields_to_check.items():
            old_value = existing_class.get(field)
            new_value, old_value, changed = self._compare_values(new_value, old_value)
            if changed:
                update_record[field] = new_value
                needs_update = True

        if needs_update:
            self.context.update_classes.append(update_record)

        if record_key:
            if record_key not in self.context.class_id_mapping:
                self.context.class_id_mapping[record_key] = []
            if existing_class['id'] not in self.context.class_id_mapping[record_key]:
                self.context.class_id_mapping[record_key].append(existing_class['id'])

    def _process_create(self, row, acad_term_id, class_boss_id, course_id, section, prof_id, prof_name, record_key, warn_inaccuracy) -> None:
        """Handle CREATE case for new class."""
        already_created = False
        for new_class in self.context.new_classes:
            if (new_class['acad_term_id'] == acad_term_id and
                str(new_class.get('boss_id')) == str(class_boss_id) and
                new_class.get('professor_id') == prof_id):
                already_created = True
                if record_key:
                    if record_key not in self.context.class_id_mapping:
                        self.context.class_id_mapping[record_key] = []
                    if new_class['id'] not in self.context.class_id_mapping[record_key]:
                        self.context.class_id_mapping[record_key].append(new_class['id'])
                break

        if not already_created:
            class_id = str(uuid.uuid4())
            # Safely handle grading_basis - convert NaN to None for enum column
            grading_basis_val = row.get('grading_basis')
            if pd.isna(grading_basis_val):
                grading_basis_val = None

            new_class = {
                'id': class_id,
                'section': section,
                'course_id': course_id,
                'professor_id': prof_id,
                'acad_term_id': acad_term_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'grading_basis': grading_basis_val,
                'course_outline_url': row.get('course_outline_url'),
                'boss_id': int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None,
                'warn_inaccuracy': warn_inaccuracy
            }
            self.context.new_classes.append(new_class)
            self.context.stats['classes_created'] += 1
            self._existing_class_lookup[(acad_term_id, class_boss_id, prof_id)] = new_class

            if record_key:
                if record_key not in self.context.class_id_mapping:
                    self.context.class_id_mapping[record_key] = []
                self.context.class_id_mapping[record_key].append(class_id)

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

    def _get_course_id(self, course_code: str):
        """Get course ID from course code."""
        return self.context.courses_cache.get(course_code, {}).get('id')

    def _find_professors_for_class(self, record_key: str) -> List[Tuple]:
        """Find professor IDs for a class."""
        if not record_key or pd.isna(record_key):
            return []

        rows = self.context.multiple_lookup.get(record_key, [])
        professor_mappings = []
        seen_professor_ids = set()

        for row in rows:
            prof_name_raw = row.get('professor_name')
            if prof_name_raw is None or pd.isna(prof_name_raw):
                continue
            original_prof_name = str(prof_name_raw).strip()
            if not original_prof_name or original_prof_name.lower() == 'nan':
                continue

            split_professors = self._split_professor_names(original_prof_name)
            for prof_name in split_professors:
                if prof_name and prof_name.strip():
                    prof_id = self._lookup_professor_with_fallback(prof_name.strip())
                    if prof_id and prof_id not in seen_professor_ids:
                        professor_mappings.append((prof_id, prof_name.strip()))
                        seen_professor_ids.add(prof_id)

        return professor_mappings

    def _split_professor_names(self, prof_name: str) -> List[str]:
        """Split professor names using greedy longest-match-first."""
        if not prof_name or not str(prof_name).strip():
            return []

        prof_name_str = str(prof_name).strip()
        if prof_name_str.upper() in self.context.professor_lookup:
            return [prof_name_str]

        if ',' not in prof_name_str:
            return [prof_name_str]

        parts = [p.strip() for p in prof_name_str.split(',') if p.strip()]
        found_professors = []
        i = 0

        while i < len(parts):
            match_found = False
            for j in range(len(parts), i, -1):
                candidate = ', '.join(parts[i:j])
                if candidate.upper() in self.context.professor_lookup:
                    found_professors.append(candidate)
                    i = j
                    match_found = True
                    break

            if not match_found:
                unknown_part = parts[i]
                if found_professors and len(unknown_part.split()) == 1:
                    found_professors[-1] = f"{found_professors[-1]}, {unknown_part}"
                else:
                    found_professors.append(unknown_part)
                i += 1

        return found_professors

    def _lookup_professor_with_fallback(self, prof_name: str):
        """Look up professor by name with fallback."""
        if prof_name.upper() in self.context.professor_lookup:
            return self.context.professor_lookup[prof_name.upper()].get('database_id')
        return None

    def _collect_results(self) -> None:
        pass

    def _persist(self) -> None:
        pass