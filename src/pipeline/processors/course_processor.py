"""
CourseProcessor - handles course CREATE vs UPDATE logic.
Extracted from table_builder.py process_courses method.
"""
import re
import uuid
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd

from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.processor_context import ProcessorContext


class CourseProcessor(AbstractProcessor):
    """Processes course records from standalone data."""

    def __init__(self, context: ProcessorContext):
        super().__init__(context)
        self._prefix_faculty_index: Dict[str, int] = {}
        self._fallback_faculty_id: Optional[int] = None

    def process(self):
        """Template method for course processing."""
        self._load_cache()
        self._do_process()
        self._collect_results()
        self._persist()

    def _load_cache(self) -> None:
        # Cache already loaded into context.courses_cache by TableBuilder
        pass

    def _do_process(self) -> None:
        """Execute course processing logic."""
        self._logger.info("Processing courses with robust CREATE vs. UPDATE logic...")

        # Build faculty resolution index once before processing
        self._build_faculty_resolution_index()

        processed_course_codes_in_run = set()

        for idx, row in self.context.standalone_data.iterrows():
            course_code = row.get('course_code')
            if pd.isna(course_code) or course_code in processed_course_codes_in_run:
                continue

            processed_course_codes_in_run.add(course_code)

            # Check if the course already exists in our database cache
            if course_code in self.context.courses_cache:
                self._process_update(row, course_code)
            else:
                self._process_create(row, course_code)

        self._logger.info(f"Course processing complete. New: {self.context.stats['courses_created']}, Updated: {self.context.stats['courses_updated']}.")

    def _build_faculty_resolution_index(self) -> None:
        """
        Build prefix-to-faculty index from existing courses for fast lookup.
        Also determines the fallback faculty (most common overall).
        """
        prefix_faculty_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        overall_faculty_counts: Dict[int, int] = defaultdict(int)

        for existing_code, existing_course in self.context.courses_cache.items():
            faculty_id = existing_course.get('belong_to_faculty')
            if faculty_id is None:
                continue

            # Handle numpy types and validate
            try:
                fid = int(faculty_id)
            except (ValueError, TypeError):
                continue

            overall_faculty_counts[fid] += 1

            # Extract prefix
            prefix_match = re.match(r'^([A-Z-]+)', existing_code.upper())
            if prefix_match:
                prefix = prefix_match.group(1)
                prefix_faculty_counts[prefix][fid] += 1

        # Build index with majority selection for each prefix
        for prefix, faculty_counts in prefix_faculty_counts.items():
            most_common_faculty = max(faculty_counts.keys(), key=lambda fid: faculty_counts[fid])
            self._prefix_faculty_index[prefix] = most_common_faculty

        # Determine fallback faculty (most common overall)
        if overall_faculty_counts:
            self._fallback_faculty_id = max(
                overall_faculty_counts.keys(),
                key=lambda fid: overall_faculty_counts[fid]
            )

        self._logger.info(
            f"Built faculty resolution index: {len(self._prefix_faculty_index)} prefixes, "
            f"fallback faculty: {self._fallback_faculty_id}"
        )

    def _determine_faculty_for_course(self, course_code: str) -> int:
        """
        Determine the faculty for a new course using prefix matching.

        Returns:
            Faculty ID (never None - uses fallback if prefix unknown)
        """
        prefix_match = re.match(r'^([A-Z-]+)', course_code.upper())
        if not prefix_match:
            # No prefix extracted - use fallback
            self._logger.warning(
                f"Could not extract prefix from course code '{course_code}'. "
                f"Using fallback faculty {self._fallback_faculty_id}."
            )
            return self._fallback_faculty_id

        prefix = prefix_match.group(1)

        if prefix in self._prefix_faculty_index:
            faculty_id = self._prefix_faculty_index[prefix]
            return faculty_id

        # Unknown prefix - log warning and use fallback
        known_prefixes = list(self._prefix_faculty_index.keys())
        self._logger.warning(
            f"Unknown course prefix '{prefix}' for course '{course_code}'. "
            f"This prefix has not been seen in existing courses. "
            f"Using fallback faculty {self._fallback_faculty_id} (most common overall). "
            f"Please verify this is correct or add the new faculty to the database. "
            f"Known prefixes: {sorted(known_prefixes)[:20]}..."
        )
        return self._fallback_faculty_id

    def _process_update(self, row, course_code: str) -> None:
        """Handle UPDATE case for existing course."""
        existing_course = self.context.courses_cache[course_code]
        update_record = {'id': existing_course['id'], 'code': course_code}

        field_mapping = {
            'name': 'course_name',
            'description': 'course_description',
            'credit_units': 'credit_units',
            'course_area': 'course_area',
            'enrolment_requirements': 'enrolment_requirements'
        }

        if self._needs_update(existing_course, row, field_mapping):
            for db_field, raw_field in field_mapping.items():
                new_value = row.get(raw_field)
                if pd.notna(new_value) and str(new_value) != str(existing_course.get(db_field)):
                    update_record[db_field] = new_value

            self.context.update_courses.append(update_record)
            self.context.stats['courses_updated'] += 1

    def _process_create(self, row, course_code: str) -> None:
        """Handle CREATE case for new course."""
        course_id = str(uuid.uuid4())

        # Determine faculty upfront using prefix-based pattern matching
        faculty_id = self._determine_faculty_for_course(course_code)

        new_course = {
            'id': course_id,
            'code': course_code,
            'name': row.get('course_name', 'Unknown Course'),
            'description': row.get('course_description', 'No description available'),
            'credit_units': float(row.get('credit_units', 1.0)) if pd.notna(row.get('credit_units')) else 1.0,
            'belong_to_university': 1,
            'belong_to_faculty': faculty_id,
            'course_area': row.get('course_area'),
            'enrolment_requirements': row.get('enrolment_requirements')
        }
        self.context.new_courses.append(new_course)
        self.context.courses_cache[course_code] = new_course
        self.context.stats['courses_created'] += 1

    def _needs_update(self, existing_record: Dict, new_record_or_row, field_mapping: Dict[str, str]) -> bool:
        """
        Check if existing record needs updates based on field mapping.
        Handles dictionary-to-dictionary and row-to-record comparisons.
        """
        for db_field, raw_field in field_mapping.items():
            old_value = existing_record.get(db_field)
            new_value = new_record_or_row.get(raw_field) if hasattr(new_record_or_row, 'get') else None

            # Type-specific comparison
            if db_field == 'credit_units':
                new_value = float(new_value) if pd.notna(new_value) else None
                old_value = float(old_value) if pd.notna(old_value) else None
            else:
                if pd.isna(new_value):
                    new_value = None
                else:
                    new_value = str(new_value).strip()

                if pd.isna(old_value):
                    old_value = None
                else:
                    old_value = str(old_value).strip() if old_value is not None else None

            # Check for actual change
            if new_value != old_value:
                # Don't overwrite existing data with empty data
                if new_value is None or new_value == '':
                    if old_value is not None and old_value != '':
                        continue
                return True
        return False

    def _collect_results(self) -> None:
        # Output already appended to context.new_courses / update_courses during _do_process
        pass

    def _persist(self) -> None:
        # Courses are persisted via TableBuilder's _execute_db_operations
        pass