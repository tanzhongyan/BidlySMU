"""
Configuration for BidlySMU pipeline.

Load environment variables from .env file.
"""
import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', '5432'),
}


# ============================================================================
# ENVIRONMENT-BASED CONFIGURATION
# ============================================================================

# Academic term ID in BOSS database format (from env)
# Example: AY202526T3A = Academic Year 2025/26, Term 3A
ACAD_TERM_ID = os.getenv('ACAD_TERM_ID', 'AY202526T3A')

# Bidding schedules loaded from JSON file (path from env)
# Format: {"2025-26_T1": [["2025-07-09T14:00:00", "Round 1 Window 1", "R1W1"], ...]}
def _load_bidding_schedules() -> Dict[str, List[Tuple[datetime, str, str]]]:
    """Load bidding schedules from JSON file specified in env."""
    schedules_path = os.getenv('BIDDING_SCHEDULES_PATH', 'script_input/bidding_schedules.json')

    try:
        with open(schedules_path, 'r') as f:
            raw_schedules = json.load(f)

        # Convert JSON format to Python format with datetime objects
        schedules = {}
        for term, entries in raw_schedules.items():
            schedules[term] = [
                (datetime.fromisoformat(entry[0]), entry[1], entry[2])
                for entry in entries
            ]
        return schedules
    except FileNotFoundError:
        print(f"Warning: BIDDING_SCHEDULES_PATH '{schedules_path}' not found. Using empty schedules.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading bidding schedules: {e}. Using empty schedules.")
        return {}

BIDDING_SCHEDULES = _load_bidding_schedules()


# ============================================================================
# DERIVED VALUES
# ============================================================================

# START_AY_TERM is the dash format used by bidding schedules (e.g., '2025-26_T3A')
# Derived from ACAD_TERM_ID
_START_YEAR = ACAD_TERM_ID[2:6]
_END_YEAR = ACAD_TERM_ID[6:8]
_TERM = ACAD_TERM_ID[8:]
START_AY_TERM = f"{_START_YEAR}-{_END_YEAR}_{_TERM}"

# Term display format mapping for BOSS UI (e.g., 'T3A' -> 'Term 3A')
TERM_DISPLAY_MAP = {
    'T1': 'Term 1',
    'T2': 'Term 2',
    'T3A': 'Term 3A',
    'T3B': 'Term 3B'
}

def acad_term_id_to_display_format(acad_term_id: str) -> str:
    """
    Convert ACAD_TERM_ID (e.g., 'AY202526T3A') to display format (e.g., '2025-26 Term 3A').

    Usage:
        acad_term_id_to_display_format('AY202526T3A') -> '2025-26 Term 3A'
    """
    if not acad_term_id or len(acad_term_id) < 9:
        return acad_term_id
    start_year = acad_term_id[2:6]
    end_year = acad_term_id[6:8]
    term_code = acad_term_id[8:]
    display_term = TERM_DISPLAY_MAP.get(term_code, term_code)
    return f"{start_year}-{end_year} {display_term}"

# START_AY_TERM_DISPLAY is the BOSS UI format (e.g., '2025-26 Term 3A')
# Used by scrapers to interact with BOSS dropdown selectors
START_AY_TERM_DISPLAY = acad_term_id_to_display_format(ACAD_TERM_ID)

# Precomputed values for current term - derived once, used everywhere
# Short format for BOSS URLs (e.g., "2531" for T1, "253A" for T3A)
# Format: last 2 digits of start year + term code digits
# Example: AY202526T3A -> "253A", AY202526T1 -> "251"
ACAD_TERM_SHORT = f"{ACAD_TERM_ID[4:6]}{_TERM[1:]}" if _TERM.startswith('T') else f"{ACAD_TERM_ID[4:6]}{_TERM}"

def display_format_to_acad_term_id(display_format: str) -> str:
    """
    Convert display format (e.g., '2025-26 Term 3A') to ACAD_TERM_ID (e.g., 'AY202526T3A').

    Usage:
        display_format_to_acad_term_id('2025-26 Term 3A') -> 'AY202526T3A'
    """
    if not display_format:
        return display_format

    # Pattern: "2025-26 Term 3A" -> extract year, term code
    match = re.match(r'(\d{4})-(\d{2})\s+Term\s+([A-Z0-9]+)', display_format)
    if match:
        start = match.group(1)
        end = match.group(2)
        term_code = match.group(3).upper()  # e.g., '3A', '1', '3B'
        # Add T prefix if not present - '1' -> 'T1', '3A' -> 'T3A'
        if not term_code.startswith('T'):
            term_code = 'T' + term_code
        # Derive from current ACAD_TERM_ID pattern: AY + start + end + term_code
        acad_term_id = f"AY{start}{end}{term_code}"
        return acad_term_id

    # Try dash format "2025-26_T3A"
    match = re.match(r'(\d{4})-(\d{2})_([A-Z0-9]+)', display_format)
    if match:
        start = match.group(1)
        end = match.group(2)
        term_code = match.group(3).upper()
        if not term_code.startswith('T'):
            term_code = 'T' + term_code
        acad_term_id = f"AY{start}{end}{term_code}"
        return acad_term_id

    return display_format

# Lazy evaluation cache for expensive/time-dependent config values
_config_cache: Dict[str, object] = {}


def _compute_and_cache_window_names() -> None:
    """Compute CURRENT_WINDOW_NAME and PREVIOUS_WINDOW_NAME lazily on first access."""
    
    target_current = os.getenv('TARGET_CURRENT_WINDOW')
    target_previous = os.getenv('TARGET_PREVIOUS_WINDOW')
    
    if target_current is not None or target_previous is not None:
        def _parse_target(val):
            if val is None: return None
            if val.lower() == 'none' or val.strip() == '': return None
            return val
            
        _config_cache['CURRENT_WINDOW_NAME'] = _parse_target(target_current)
        _config_cache['PREVIOUS_WINDOW_NAME'] = _parse_target(target_previous)
        return

    schedule = BIDDING_SCHEDULES.get(START_AY_TERM, [])
    now = datetime.now()

    current_window_name = None
    previous_window_name = None

    for i, (results_date, window_name, *rest) in enumerate(schedule):
        if now < results_date:
            current_window_name = window_name
            if i > 0:
                previous_window_name = schedule[i - 1][1]
            break

    if current_window_name is None and schedule:
        current_window_name = None
        previous_window_name = schedule[-1][1]

    _config_cache['CURRENT_WINDOW_NAME'] = current_window_name
    _config_cache['PREVIOUS_WINDOW_NAME'] = previous_window_name


def __getattr__(name):
    """Lazy evaluation for time-dependent and I/O-dependent config values."""
    if name == 'CURRENT_WINDOW_NAME':
        if name not in _config_cache:
            _compute_and_cache_window_names()
        return _config_cache[name]
    if name == 'PREVIOUS_WINDOW_NAME':
        if name not in _config_cache:
            _compute_and_cache_window_names()
        return _config_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _reset_config_cache() -> None:
    """Clear cached config values. For testing only."""
    _config_cache.clear()


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for PipelineCoordinator."""

    def __init__(
        self,
        bidding_schedules: dict,
        start_ay_term: str,
        db_config: dict,
        input_file: str = 'script_input/raw_data.xlsx',
        output_base: str = 'script_output',
        verify_dir: str = 'script_output/verify',
        cache_dir: str = 'db_cache',
        overall_results_dir: str = 'script_input/overallBossResults',
    ):
        self.bidding_schedules = bidding_schedules
        self.start_ay_term = start_ay_term
        self.db_config = db_config
        self.input_file = input_file
        self.output_base = output_base
        self.verify_dir = verify_dir
        self.cache_dir = cache_dir
        self.overall_results_dir = overall_results_dir

    @classmethod
    def from_env(cls, bidding_schedules: dict, start_ay_term: str, db_config: dict):
        return cls(
            bidding_schedules=bidding_schedules,
            start_ay_term=start_ay_term,
            db_config=db_config,
        )


# ============================================================================
# BIDDING WINDOW PARSING
# ============================================================================

class _BiddingWindowFormat:
    """A single bidding window parsing format with regex pattern and mapper."""

    def __init__(self, pattern: str, mapper: Callable[[Any], Tuple[str, int]]):
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._mapper = mapper

    def try_parse(self, text: str) -> Optional[Tuple[str, int]]:
        m = self._pattern.search(text)
        return self._mapper(m) if m else None


_FORMATS: List[_BiddingWindowFormat] = [
    _BiddingWindowFormat(
        r'Incoming\s+Freshmen\s+Rnd\s+(\w+)\s+Win\s+(\d+)',
        lambda m: ("1F" if m.group(1) == "1" else f"{m.group(1)}F", int(m.group(2))),
    ),
    _BiddingWindowFormat(
        r'Incoming\s+Exchange\s+Rnd\s+(\w+)\s+Win\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
    _BiddingWindowFormat(
        r'Round\s+(\d[A-C]?|\d+F?)\s+Window\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
]
_FORMATS_WITH_ABBREV = _FORMATS + [
    _BiddingWindowFormat(
        r'Rnd\s+(\d[A-C]?|\d+F?)\s+Win\s+(\d+)',
        lambda m: (m.group(1), int(m.group(2))),
    ),
]


def parse_bidding_window(
    bidding_window_str: str,
    *,
    allow_abbrev: bool = True,
    allow_generic_fallback: bool = False,
    default_round: Optional[str] = None,
    default_window: Optional[int] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse bidding window text into (round, window).

    Supported formats include:
    - Round 1 Window 1
    - Round 1A Window 2
    - Incoming Exchange Rnd 1C Win 1
    - Incoming Freshmen Rnd 1 Win 4
    - Rnd 1A Win 2
    """
    if bidding_window_str is None or (isinstance(bidding_window_str, float) and bidding_window_str != bidding_window_str):
        return default_round, default_window

    window_str = str(bidding_window_str).strip()
    if not window_str:
        return default_round, default_window

    for fmt in (_FORMATS_WITH_ABBREV if allow_abbrev else _FORMATS):
        result = fmt.try_parse(window_str)
        if result:
            return result

    if allow_generic_fallback:
        fallback_round_match = re.search(r'(\d[A-C]?|\d+F?)', window_str)
        if fallback_round_match:
            fallback_window_match = re.search(r'Window\s+(\d+)|Win\s+(\d+)', window_str, re.IGNORECASE)
            if fallback_window_match:
                window_num = int(fallback_window_match.group(1) or fallback_window_match.group(2))
                return fallback_round_match.group(1), window_num
            return fallback_round_match.group(1), 1

    return default_round, default_window