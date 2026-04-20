# CourseDTO - Data Transfer Object for course records
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
import uuid


@dataclass
class CourseDTO:
    """DTO representing a course record."""

    # Column name mapping (CSV = DB in this case, per Prisma @map)
    COLUMNS = {
        'id': 'id',
        'code': 'code',
        'name': 'name',
        'description': 'description',
        'credit_units': 'credit_units',
        'belong_to_university': 'belong_to_university',
        'belong_to_faculty': 'belong_to_faculty',
        'course_area': 'course_area',
        'enrolment_requirements': 'enrolment_requirements',
        'updated_at': 'updated_at'
    }

    id: str  # UUID - generated before insert
    code: str  # Unique
    name: str
    description: str
    credit_units: float
    belong_to_university: int  # Always 1
    belong_to_faculty: int  # Determined via prefix matching
    course_area: Optional[str]
    enrolment_requirements: Optional[str]
    updated_at: Optional[datetime] = None  # Set explicitly on UPDATE

    def to_csv_row(self) -> dict:
        """Convert to CSV row."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}

    def to_db_row(self) -> dict:
        """Convert to database row."""
        return {self.COLUMNS[k]: getattr(self, k) for k in self.COLUMNS}

    @staticmethod
    def from_row(row, faculty_id: int) -> 'CourseDTO':
        """Factory method to create DTO from DataFrame row."""
        import pandas as pd

        def safe_val(val, default):
            if pd.isna(val):
                return default
            return val

        return CourseDTO(
            id=str(uuid.uuid4()),
            code=str(row.get('course_code', '')),
            name=str(safe_val(row.get('course_name'), 'Unknown Course')),
            description=str(safe_val(row.get('course_description'), 'No description available')),
            credit_units=float(safe_val(row.get('credit_units'), 1.0)),
            belong_to_university=1,
            belong_to_faculty=faculty_id,
            course_area=str(row.get('course_area')) if pd.notna(row.get('course_area')) else None,
            enrolment_requirements=str(row.get('enrolment_requirements')) if pd.notna(row.get('enrolment_requirements')) else None,
            updated_at=None  # CREATE doesn't need updated_at - DB default handles it
        )
