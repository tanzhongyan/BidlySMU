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

from src.config import ROUND_ORDER, SGT, _parse_sgt, parse_bidding_window, build_window_abbrev


@dataclass(frozen=True)
class TrubaConfig:
    """Immutable configuration for Truba API client."""
    api_url: str = "https://www.trumba.com/calendars/SMU_RO_Acad.json"
    months_ahead: int = 12
    timeout: int = 60
    user_agent: str = "BidlySMU/1.0"


@dataclass(frozen=True)
class BossEvent:
    """Immutable BOSS bidding event (window or results)."""
    term: str
    abbrev: str
    title: str
    is_results: bool
    start_dt: Optional[str] = None   # ISO 8601: opensAt (window) or resultsAt (results)
    end_dt: Optional[str] = None     # ISO 8601: closesAt (window only, None for results)

    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization."""
        return {
            "term": self.term,
            "abbrev": self.abbrev,
            "title": self.title,
            "is_results": self.is_results,
            "start_dt": self.start_dt,
            "end_dt": self.end_dt,
        }

    def to_schedule_entry(self) -> list:
        """Convert to the canonical schedule entry format: `[results_at, title, abbrev, opens_at, closes_at]`.

        Uses start_dt as results_at for results events.
        opens_at/closes_at are None for raw BossEvent (only BossWindow has them).
        """
        return [self.start_dt, self.title, self.abbrev, None, None]


@dataclass(frozen=True)
class BossWindow:
    """Paired BOSS bidding window with all three timeline dates."""
    term: str
    abbrev: str
    title: str
    opens_at: str     # ISO 8601 in SGT
    closes_at: str    # ISO 8601 in SGT
    results_at: str   # ISO 8601 in SGT

    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization."""
        return {
            "term": self.term,
            "abbrev": self.abbrev,
            "title": self.title,
            "opens_at": self.opens_at,
            "closes_at": self.closes_at,
            "results_at": self.results_at,
        }

    def to_schedule_entry(self) -> list:
        """Convert to the canonical schedule entry format: `[results_at, title, abbrev, opens_at, closes_at]`.

        results_at is the results-release timestamp; opens_at/closes_at are the
        actual window open/close dates.
        """
        return [self.results_at, self.title, self.abbrev, self.opens_at, self.closes_at]


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
        today = datetime.now(SGT)
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
        Fetch all BOSS events (both Window and Results) from Trumba API.

        Window events provide opensAt/closesAt (startDateTime/endDateTime).
        Results events provide resultsAt (startDateTime).

        Returns:
            List of BossEvent objects (both window and results events)

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

    def fetch_boss_windows(self) -> List[BossWindow]:
        """
        Fetch and pair BOSS events into complete bid windows with all three dates.

        Matches Window events (opensAt/closesAt) with their corresponding
        Results events (resultsAt) by term + abbrev.

        Returns:
            List of BossWindow objects with opensAt, closesAt, resultsAt

        Raises:
            requests.RequestException: If HTTP request fails
        """
        all_events = self.fetch_boss_events()

        # Separate into windows and results
        windows_by_key: Dict[str, BossEvent] = {}
        results_by_key: Dict[str, BossEvent] = {}

        for event in all_events:
            if not event.abbrev:
                # Skip events whose round/window couldn't be parsed (e.g.
                # BOSS-named events that aren't bid windows).
                continue
            key = f"{event.term}|{event.abbrev}"
            if event.is_results:
                results_by_key[key] = event
            else:
                windows_by_key[key] = event

        # Pair windows with their results
        paired: List[BossWindow] = []
        for key, window_event in windows_by_key.items():
            results_event = results_by_key.get(key)
            if results_event and window_event.start_dt and window_event.end_dt:
                paired.append(BossWindow(
                    term=window_event.term,
                    abbrev=window_event.abbrev,
                    title=window_event.title,
                    opens_at=window_event.start_dt,
                    closes_at=window_event.end_dt,
                    results_at=results_event.start_dt,
                ))

        # Sort by term, then by abbrev (R1W1, R1AW1, etc.)
        def sort_key(bw: BossWindow) -> tuple:
            # Extract round prefix for ordering (e.g., "R1AW1" → "R1A")
            import re
            m = re.match(r'(R\d+[A-F]?)W\d+', bw.abbrev)
            round_prefix = m.group(1) if m else bw.abbrev
            # Strip "R" prefix for centralized ROUND_ORDER lookup
            round_key = round_prefix.lstrip('R') if round_prefix.startswith('R') else round_prefix
            return (bw.term, ROUND_ORDER.get(round_key, 99), bw.abbrev)

        paired.sort(key=sort_key)
        return paired

    def _parse_boss_event(self, event_data: Dict) -> Optional[BossEvent]:
        """
        Parse a single BOSS event from API response.

        Handles both Window events (provide opensAt/closesAt) and
        Results events (provide resultsAt).

        Args:
            event_data: Event dictionary from API

        Returns:
            BossEvent if valid BOSS event, None otherwise
        """
        try:
            title = event_data.get("title", "")

            # Filter for BOSS events only
            if "BOSS" not in title.upper():
                return None

            # Get start and end datetimes from Trumba
            start_str = event_data.get("startDateTime")
            end_str = event_data.get("endDateTime")

            # Parse start datetime (always present); normalize to SGT
            if not start_str:
                return None
            start_dt = _parse_sgt(start_str)

            # Parse end datetime (present for Window events, may be absent for Results)
            end_dt = _parse_sgt(end_str) if end_str else None

            is_results = "RESULTS" in title.upper()

            # Extract term
            term = self._extract_term(event_data)

            # Parse round/window from title (None when not a bid window)
            abbrev = self._parse_title(title)
            if abbrev is None:
                return None

            # Clean title (remove "Results" suffix and "BOSS " prefix for consistency)
            clean_title = title.replace(" Results", "").replace("BOSS ", "", 1).strip()

            # Persist SGT wall-clock WITHOUT tzinfo — bidding_schedules.json
            # stores naive SGT wall-clock datetimes.
            return BossEvent(
                term=term,
                abbrev=abbrev,
                title=clean_title,
                is_results=is_results,
                start_dt=start_dt.astimezone(SGT).replace(tzinfo=None).isoformat(),
                end_dt=end_dt.astimezone(SGT).replace(tzinfo=None).isoformat() if end_dt else None,
            )

        except Exception:
            return None

    def _extract_term(self, event_data: Dict) -> str:
        """
        Extract term identifier from event custom fields.

        Args:
            event_data: Event dictionary from API

        Returns:
            Term identifier (e.g., "AY202627T1")
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
                        return f"AY{start_year}{end_year}T{term_num}"

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
            Term identifier (e.g., "AY202526T3A")
        """
        now = datetime.now(SGT)
        year = now.year

        if now.month >= 8:
            return f"AY{year}{(year+1) % 100:02d}T1"
        elif now.month >= 1 and now.month <= 4:
            return f"AY{year-1}{year % 100:02d}T2"
        elif now.month == 5:
            return f"AY{year-1}{year % 100:02d}T3A"
        else:
            return f"AY{year-1}{year % 100:02d}T3B"

    def _parse_title(self, title: str) -> Optional[str]:
        """
        Parse BOSS event title into a window abbrev (e.g. "R1AW2"), or None.

        Uses the canonical config helpers (parse_bidding_window +
        build_window_abbrev) so round/window vocabulary matches the rest of
        the codebase. Handles:
        - Regular rounds: "BOSS Round 1A Window 1 Results" -> "R1AW1"
        - Incoming Exchange: "BOSS Incoming Exchange Round 1C Window 1" -> "R1CW1"
        - Incoming Freshmen: "BOSS Incoming Freshmen Round 1 Window 1" -> "R1FW1"

        Returns None for titles that are not a bid window (e.g. "BOSS Course
        Registration") instead of a bogus "R1W1" that would alias real windows.

        Args:
            title: Event title (e.g., "BOSS Round 1A Window 1 Results")

        Returns:
            Abbreviated window key (e.g. "R1AW2") or None if unparseable.
        """
        round_str, window_num = parse_bidding_window(title, allow_abbrev=True)
        if round_str is None or window_num is None:
            return None
        return build_window_abbrev(round_str, window_num)
