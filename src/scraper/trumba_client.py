"""
Truba API client for fetching SMU calendar events.

This module provides a clean API for fetching and parsing BOSS bidding events
from the Truba JSON API. It follows SOLID principles:

- Single Responsibility: Only handles Truba API interaction
- Open/Closed: Configurable via TrubaConfig without modification
- Liskov Substitution: Implements TrubaClientInterface
- Interface Segregation: Minimal public interface
- Dependency Inversion: Depends on abstractions (config)

Usage:
    from src.scraper.trumba_client import TrubaClient, TrubaConfig

    config = TrubaConfig(months_ahead=12)
    client = TrubaClient(config)
    events = client.fetch_boss_events()
"""
import re
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Protocol


@dataclass(frozen=True)
class TrubaConfig:
    """Immutable configuration for Truba API client."""
    api_url: str = "https://www.trumba.com/calendars/SMU_RO_Acad.json"
    months_ahead: int = 12
    timeout: int = 60
    user_agent: str = "BidlySMU/1.0"


@dataclass(frozen=True)
class BossEvent:
    """Immutable BOSS bidding event."""
    term: str
    datetime: str
    abbrev: str
    title: str
    is_results: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization."""
        return {
            "term": self.term,
            "datetime": self.datetime,
            "abbrev": self.abbrev,
            "title": self.title,
            "is_results": self.is_results
        }


class TrubaClientInterface(Protocol):
    """Protocol defining the Truba client interface."""

    def fetch_events(self) -> List[Dict]:
        """Fetch all events from Truba API."""
        ...

    def fetch_boss_events(self) -> List[BossEvent]:
        """Fetch only BOSS bidding events."""
        ...


class TrubaClient:
    """
    Client for fetching events from Truba JSON API.

    This client handles:
    - HTTP requests to Truba API
    - Parsing of event data
    - Extraction of BOSS bidding events
    - Term parsing from custom fields
    """

    def __init__(self, config: TrubaConfig):
        """
        Initialize Truba client.

        Args:
            config: TrubaConfig instance with API settings
        """
        self._config = config

    def fetch_events(self) -> List[Dict]:
        """
        Fetch all events from Truba API.

        Returns:
            List of event dictionaries from API

        Raises:
            requests.RequestException: If HTTP request fails
        """
        today = datetime.now()
        start_date = today.strftime("%Y%m%d")

        url = f"{self._config.api_url}?startdate={start_date}&months={self._config.months_ahead}"

        response = requests.get(
            url,
            timeout=self._config.timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": self._config.user_agent
            }
        )
        response.raise_for_status()
        return response.json()

    def fetch_boss_events(self) -> List[BossEvent]:
        """
        Fetch only BOSS bidding events (Results events only).

        Results events represent the actual bidding result release times.
        Window events are excluded as they only define the bidding period.

        Returns:
            List of BossEvent objects

        Raises:
            requests.RequestException: If HTTP request fails
        """
        events_data = self.fetch_events()
        boss_events = []

        for event_data in events_data:
            event = self._parse_boss_event(event_data)
            if event:
                boss_events.append(event)

        return boss_events

    def _parse_boss_event(self, event_data: Dict) -> Optional[BossEvent]:
        """
        Parse a single BOSS event from API response.

        Args:
            event_data: Event dictionary from API

        Returns:
            BossEvent if valid BOSS Results event, None otherwise
        """
        try:
            title = event_data.get("title", "")

            # Filter for BOSS events only
            if "BOSS" not in title.upper():
                return None

            # Only process Results events (actual result release times)
            is_results = "RESULTS" in title.upper()
            if not is_results:
                return None

            # Get datetime (startDateTime for Results events)
            datetime_str = event_data.get("startDateTime")
            if not datetime_str:
                return None

            # Parse datetime
            parsed_dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

            # Extract term
            term = self._extract_term(event_data)

            # Parse round/window from title
            round_info, window_info = self._parse_title(title)
            abbrev = f"{round_info}{window_info}"

            # Clean title (remove "Results" suffix)
            clean_title = title.replace(" Results", "").strip()

            return BossEvent(
                term=term,
                datetime=parsed_dt.isoformat(),
                abbrev=abbrev,
                title=clean_title,
                is_results=is_results
            )

        except Exception:
            return None

    def _extract_term(self, event_data: Dict) -> str:
        """
        Extract term identifier from event custom fields.

        Args:
            event_data: Event dictionary from API

        Returns:
            Term identifier (e.g., "2026-27_T1")
        """
        try:
            custom_fields = event_data.get("customFields", [])

            for field in custom_fields:
                if field.get("label") == "Term":
                    term_text = field.get("value", "")

                    # Parse "2026-27 Term 1 - Regular Academic Session"
                    match = re.search(
                        r"(\d{4})-(\d{2,4})\s+Term\s+(\d+[AB]?)",
                        term_text,
                        re.IGNORECASE
                    )
                    if match:
                        start_year = match.group(1)
                        end_year = match.group(2)
                        if len(end_year) == 4:
                            end_year = end_year[-2:]
                        term_num = match.group(3)
                        return f"{start_year}-{end_year}_T{term_num}"

        except Exception:
            pass

        # Fallback: infer term from current date
        return self._infer_term_from_date()

    def _infer_term_from_date(self) -> str:
        """
        Infer term identifier from current date.

        SMU terms:
        - Term 1: August - December (AY starts)
        - Term 2: January - May
        - Term 3: May - July (Term 3A/3B)

        Returns:
            Term identifier (e.g., "2025-26_T3A")
        """
        now = datetime.now()
        year = now.year

        if now.month >= 8:
            return f"{year}-{(year+1) % 100:02d}_T1"
        elif now.month >= 1 and now.month <= 4:
            return f"{year-1}-{year % 100:02d}_T2"
        elif now.month == 5:
            return f"{year-1}-{year % 100:02d}_T3A"
        else:
            return f"{year-1}-{year % 100:02d}_T3B"

    def _parse_title(self, title: str) -> tuple:
        """
        Parse BOSS event title to extract round and window abbreviations.

        Args:
            title: Event title (e.g., "BOSS Round 1A Window 1 Results")

        Returns:
            tuple: (round_abbrev, window_abbrev) e.g., ("R1A", "W1")
        """
        match = re.search(
            r"Round\s+(\d+)([A-B]?)\s+Window\s+(\d+)",
            title,
            re.IGNORECASE
        )

        if match:
            round_num = match.group(1)
            round_suffix = match.group(2).upper() if match.group(2) else ""
            window_num = match.group(3)
            return (f"R{round_num}{round_suffix}", f"W{window_num}")

        return ("R1", "W1")
