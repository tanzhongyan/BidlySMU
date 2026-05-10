"""
ClassDTO - Data Transfer Object for class records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class ClassDTO:
    """DTO representing a class record."""

    COLUMNS = {
        'id': 'id',
        'section': 'section',
        'course_id': 'course_id',
        'professor_id': 'professor_id',
        'acad_term_id': 'acad_term_id',
        'grading_basis': 'grading_basis',
        'course_outline_url': 'course_outline_url',
        'boss_id': 'boss_id',
        'warn_inaccuracy': 'warn_inaccuracy',
        'created_at': 'created_at',
        'updated_at': 'updated_at'
    }

    id: str
    section: str
    course_id: str
    professor_id: Optional[str]
    acad_term_id: str
    grading_basis: Optional[str] = None
    course_outline_url: Optional[str] = None
    boss_id: Optional[int] = None
    warn_inaccuracy: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        return {
            'id': self.id,
            'section': self.section,
            'course_id': self.course_id,
            'professor_id': self.professor_id,
            'acad_term_id': self.acad_term_id,
            'grading_basis': self.grading_basis,
            'course_outline_url': self.course_outline_url,
            'boss_id': self.boss_id,
            'warn_inaccuracy': self.warn_inaccuracy,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        from datetime import datetime, timezone
        return {
            'id': self.id,
            'section': self.section,
            'course_id': self.course_id,
            'professor_id': self.professor_id,
            'created_at': self.created_at or datetime.now(timezone.utc),
            'updated_at': self.updated_at or datetime.now(timezone.utc),
            'acad_term_id': self.acad_term_id,
            'boss_id': self.boss_id,
            'course_outline_url': self.course_outline_url,
            'grading_basis': self.grading_basis,
            'warn_inaccuracy': self.warn_inaccuracy
        }

    @staticmethod
    def from_row(row: dict, course_id: str, professor_id: Optional[str]) -> 'ClassDTO':
        """Factory for CREATE - generates UUID, sets timestamps."""
        import pandas as pd

        def safe_val(val, default=None):
            if pd.isna(val):
                return default
            return val

        section = str(row.get('section', ''))
        acad_term_id = str(row.get('acad_term_id', ''))
        grading_basis = safe_val(row.get('grading_basis'))
        course_outline_url = safe_val(row.get('course_outline_url'))
        boss_id = int(row.get('class_boss_id')) if pd.notna(row.get('class_boss_id')) else None
        warn_inaccuracy = len(row.get('professor_names', [])) > 1 if isinstance(row.get('professor_names'), list) else False

        now = datetime.now()

        return ClassDTO(
            id=str(uuid.uuid4()),
            section=section,
            course_id=course_id,
            professor_id=professor_id,
            acad_term_id=acad_term_id,
            grading_basis=grading_basis,
            course_outline_url=course_outline_url,
            boss_id=boss_id,
            warn_inaccuracy=warn_inaccuracy,
            created_at=now,
            updated_at=now
        )