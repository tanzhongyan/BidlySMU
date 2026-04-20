"""
ProfessorDTO - Data Transfer Object for professor records.
Encapsulates serialization logic and factory methods for CREATE.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import json
import uuid
from datetime import datetime, timezone


@dataclass
class ProfessorDTO:
    """DTO representing a professor record."""

    COLUMNS = {
        'id': 'id',
        'name': 'name',
        'email': 'email',
        'slug': 'slug',
        'photo_url': 'photo_url',
        'profile_url': 'profile_url',
        'belong_to_university': 'belong_to_university',
        'boss_aliases': 'boss_aliases',
        'original_scraped_name': 'original_scraped_name'
    }

    # Default values
    DEFAULT_EMAIL = 'enquiry@smu.edu.sg'
    DEFAULT_PHOTO_URL = 'https://smu.edu.sg'
    DEFAULT_PROFILE_URL = 'https://smu.edu.sg'
    DEFAULT_UNIVERSITY_ID = 1

    id: str
    name: str
    email: str
    slug: str
    photo_url: str
    profile_url: str
    belong_to_university: int = 1
    boss_aliases: List[str] = field(default_factory=list)
    original_scraped_name: str = ''
    variations_found: List[str] = field(default_factory=list, repr=False)
    updated_at: Optional[str] = None  # None for CREATE, datetime for UPDATE

    def to_csv_row(self) -> dict:
        """Convert to CSV row for script_output."""
        row = {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'slug': self.slug,
            'photo_url': self.photo_url,
            'profile_url': self.profile_url,
            'belong_to_university': self.belong_to_university,
            'boss_aliases': json.dumps(self.boss_aliases),
            'original_scraped_name': self.original_scraped_name
        }
        if self.updated_at:
            row['updated_at'] = self.updated_at
        return row

    def to_db_row(self) -> dict:
        """Convert to database row for INSERT."""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'slug': self.slug,
            'photo_url': self.photo_url,
            'profile_url': self.profile_url,
            'belong_to_university': self.belong_to_university,
            'boss_aliases': json.dumps(self.boss_aliases)
        }

    @staticmethod
    def from_row(row: dict, **kwargs) -> 'ProfessorDTO':
        """Factory for CREATE - generates UUID, sets updated_at=None."""
        boss_aliases_raw = row.get('boss_aliases', '[]')
        if isinstance(boss_aliases_raw, str):
            try:
                boss_aliases = json.loads(boss_aliases_raw)
            except json.JSONDecodeError:
                boss_aliases = []
        else:
            boss_aliases = boss_aliases_raw or []

        return ProfessorDTO(
            id=str(uuid.uuid4()),
            name=row.get('name', ''),
            email=row.get('email', ProfessorDTO.DEFAULT_EMAIL),
            slug=row.get('slug', ''),
            photo_url=row.get('photo_url', ProfessorDTO.DEFAULT_PHOTO_URL),
            profile_url=row.get('profile_url', ProfessorDTO.DEFAULT_PROFILE_URL),
            belong_to_university=ProfessorDTO.DEFAULT_UNIVERSITY_ID,
            boss_aliases=boss_aliases,
            original_scraped_name=row.get('original_scraped_name', ''),
            updated_at=None
        )