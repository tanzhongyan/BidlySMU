"""
CourseProcessor - handles course CREATE vs UPDATE logic.
Class-based processor that returns (new_courses, updated_courses) DTOs.
"""
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import pandas as pd

from src.logging.logger import get_logger
from src.pipeline.abstract_processor import AbstractProcessor
from src.pipeline.dtos.course_dto import CourseDTO


class CourseProcessor(AbstractProcessor):
    """Processes course records from standalone data."""

    def __init__(
        self,
        raw_data: pd.DataFrame,
        courses_cache: Dict[str, Any],
        faculties_cache: Dict[int, Any]
    ):
        self._logger = get_logger(__name__)
        self._raw_data = raw_data
        self._courses_cache = courses_cache
        self._faculties_cache = faculties_cache
        self._prefix_faculty_index: Dict[str, int] = {}
        self._fallback_faculty_id: int = 1

    def process(self) -> Tuple[List[CourseDTO], List[CourseDTO]]:
        """Process courses and return (new_courses, updated_courses) DTOs."""
        self._build_faculty_resolution_index()
        return self._do_process()

    def _build_faculty_resolution_index(self) -> None:
        """Build prefix-to-faculty index from existing courses for fast lookup."""
        prefix_faculty_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for _, course in self._courses_cache.items():
            faculty_id = course.get('belong_to_faculty')
            if faculty_id is None:
                continue

            try:
                fid = int(faculty_id)
            except (ValueError, TypeError):
                continue

            prefix_match = re.match(r'^([A-Z-]+)', course.get('code', '').upper())
            if prefix_match:
                prefix = prefix_match.group(1)
                prefix_faculty_counts[prefix][fid] += 1

        # Build index with majority selection for each prefix
        for prefix, faculty_counts in prefix_faculty_counts.items():
            most_common_faculty = max(faculty_counts.keys(), key=lambda fid: faculty_counts[fid])
            self._prefix_faculty_index[prefix] = most_common_faculty

        # Determine fallback faculty (first faculty in cache or 1)
        if self._faculties_cache:
            self._fallback_faculty_id = list(self._faculties_cache.keys())[0]

        self._logger.info(
            f"Built faculty resolution index: {len(self._prefix_faculty_index)} prefixes, "
            f"fallback faculty: {self._fallback_faculty_id}"
        )

    def _determine_faculty_for_course(self, course_code: str) -> int:
        """Determine the faculty for a new course using prefix matching."""
        prefix_match = re.match(r'^([A-Z-]+)', course_code.upper())
        if not prefix_match:
            self._logger.warning(
                f"Could not extract prefix from course code '{course_code}'. "
                f"Using fallback faculty {self._fallback_faculty_id}."
            )
            return self._fallback_faculty_id

        prefix = prefix_match.group(1)

        if prefix in self._prefix_faculty_index:
            return self._prefix_faculty_index[prefix]

        # Unknown prefix - log warning and use fallback
        self._logger.warning(
            f"Unknown course prefix '{prefix}' for course '{course_code}'. "
            f"This prefix has not been seen in existing courses. "
            f"Using fallback faculty {self._fallback_faculty_id} (most common overall). "
            f"Please verify this is correct or add the new faculty to the database."
        )
        return self._fallback_faculty_id

    def _do_process(self) -> Tuple[List[CourseDTO], List[CourseDTO]]:
        """Execute course processing logic.

        Returns:
            Tuple of (new_courses, updated_courses) as lists of CourseDTO.
        """
        self._logger.info("Processing courses with robust CREATE vs. UPDATE logic...")

        results_new: List[CourseDTO] = []
        results_updated: List[CourseDTO] = []
        processed_codes = set()

        for _, row in self._raw_data.iterrows():
            course_code = row.get('course_code')

            # Skip NaN or duplicate course codes within this run
            if pd.isna(course_code) or course_code in processed_codes:
                continue
            processed_codes.add(course_code)

            # Check if the course already exists in our database cache
            if course_code in self._courses_cache:
                self._process_update(row, course_code, results_updated)
            else:
                self._process_create(row, course_code, results_new)

        self._logger.info(
            f"Course processing complete. "
            f"New: {len(results_new)}, Updated: {len(results_updated)}."
        )

        return results_new, results_updated

    def _needs_update(self, existing_record: Dict, new_record_or_row) -> bool:
        """Check if course needs update based on field comparison."""
        field_comparisons = [
            ('course_name', 'name'),
            ('course_description', 'description'),
            ('credit_units', 'credit_units'),
            ('course_area', 'course_area'),
            ('enrolment_requirements', 'enrolment_requirements')
        ]

        for csv_field, db_field in field_comparisons:
            new_val = new_record_or_row.get(csv_field) if hasattr(new_record_or_row, 'get') else None
            old_val = existing_record.get(db_field)

            # Handle NaN/NA values
            new_is_na = pd.isna(new_val)
            old_is_na = pd.isna(old_val)

            if new_is_na and old_is_na:
                continue
            if new_is_na or old_is_na:
                return True

            # Compare string values (strip whitespace)
            new_str = str(new_val).strip()
            old_str = str(old_val).strip()

            if new_str != old_str:
                return True

        return False

    def _process_update(
        self,
        row: pd.Series,
        course_code: str,
        results_updated: List[CourseDTO]
    ) -> None:
        """Handle UPDATE case for existing course."""
        existing_course = self._courses_cache[course_code]

        if not self._needs_update(existing_course, row):
            return

        updated_dto = CourseDTO(
            id=existing_course['id'],
            code=course_code,
            name=str(row.get('course_name')) if pd.notna(row.get('course_name')) else existing_course.get('name', 'Unknown Course'),
            description=str(row.get('course_description')) if pd.notna(row.get('course_description')) else existing_course.get('description', 'No description available'),
            credit_units=float(row.get('credit_units')) if pd.notna(row.get('credit_units')) else existing_course.get('credit_units', 1.0),
            belong_to_university=1,
            belong_to_faculty=existing_course.get('belong_to_faculty', self._fallback_faculty_id),
            course_area=str(row.get('course_area')) if pd.notna(row.get('course_area')) else None,
            enrolment_requirements=str(row.get('enrolment_requirements')) if pd.notna(row.get('enrolment_requirements')) else None,
            updated_at=datetime.now(timezone.utc)  # Explicitly set on UPDATE
        )
        results_updated.append(updated_dto)

    def _process_create(
        self,
        row: pd.Series,
        course_code: str,
        results_new: List[CourseDTO]
    ) -> None:
        """Handle CREATE case for new course."""
        faculty_id = self._determine_faculty_for_course(course_code)
        new_dto = CourseDTO.from_row(row, faculty_id)
        results_new.append(new_dto)