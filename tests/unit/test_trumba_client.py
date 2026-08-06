"""
Unit tests for TrubaClient and Lambda monthly scheduler components.

Tests cover:
- TrubaConfig and BossEvent dataclasses
- TrubaClient API interaction and parsing
- EventBridge scheduler integration
- Bidding schedule management
- Schedule tracking
"""
import importlib.util
import json
import pytest
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime
from dataclasses import FrozenInstanceError
from pathlib import Path
import requests

from src.scraper.trumba_client import TrubaConfig, BossEvent, TrubaClient


# Import Lambda module (lambda is a reserved keyword, so use importlib)
LAMBDA_PATH = Path(__file__).parent.parent.parent / "lambda" / "monthly_scheduler" / "lambda_function.py"

# Check if boto3 is available for Lambda tests
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def _load_lambda_module():
    """Load the Lambda function module using importlib."""
    if not BOTO3_AVAILABLE:
        pytest.skip("boto3 not available - skipping Lambda tests")
    spec = importlib.util.spec_from_file_location("lambda_function", LAMBDA_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================
# TrubaConfig Tests
# ==============================================================================

class TestTrubaConfig:
    """Tests for TrubaConfig dataclass."""

    def test_config_defaults(self):
        """Config should have correct default values."""
        config = TrubaConfig()
        assert config.api_url == "https://www.trumba.com/calendars/SMU_RO_Acad.json"
        assert config.months_ahead == 12
        assert config.timeout == 60
        assert config.user_agent == "BidlySMU/1.0"

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = TrubaConfig(
            api_url="https://custom.url/api.json",
            months_ahead=6,
            timeout=30,
            user_agent="CustomAgent/1.0"
        )
        assert config.api_url == "https://custom.url/api.json"
        assert config.months_ahead == 6
        assert config.timeout == 30
        assert config.user_agent == "CustomAgent/1.0"

    def test_config_is_frozen(self):
        """Config should be immutable (frozen dataclass)."""
        config = TrubaConfig()
        with pytest.raises(FrozenInstanceError):
            config.timeout = 120


# ==============================================================================
# BossEvent Tests
# ==============================================================================

class TestBossEvent:
    """Tests for BossEvent dataclass."""

    def test_boss_event_creation(self):
        """BossEvent should store all fields correctly."""
        event = BossEvent(
            term="AY202627T1",
            start_dt="2026-07-08T14:00:00",
            abbrev="R1W1",
            title="BOSS Round 1 Window 1",
            is_results=True
        )
        assert event.term == "AY202627T1"
        assert event.start_dt == "2026-07-08T14:00:00"
        assert event.abbrev == "R1W1"
        assert event.title == "BOSS Round 1 Window 1"
        assert event.is_results is True

    def test_boss_event_to_dict(self):
        """to_dict should return correct dictionary representation."""
        event = BossEvent(
            term="AY202627T1",
            start_dt="2026-07-08T14:00:00",
            abbrev="R1W1",
            title="BOSS Round 1 Window 1",
            is_results=True
        )
        result = event.to_dict()
        assert result == {
            "term": "AY202627T1",
            "start_dt": "2026-07-08T14:00:00",
            "abbrev": "R1W1",
            "title": "BOSS Round 1 Window 1",
            "is_results": True,
            "end_dt": None,
        }

    def test_boss_event_is_frozen(self):
        """BossEvent should be immutable (frozen dataclass)."""
        event = BossEvent(
            term="AY202627T1",
            start_dt="2026-07-08T14:00:00",
            abbrev="R1W1",
            title="BOSS Round 1 Window 1",
            is_results=True
        )
        with pytest.raises(FrozenInstanceError):
            event.term = "2026-27_T2"


# ==============================================================================
# TrubaClient Initialization Tests
# ==============================================================================

class TestTrubaClientInit:
    """Tests for TrubaClient initialization."""

    def test_initializes_with_config(self, truba_config):
        """Client should initialize with provided config."""
        client = TrubaClient(truba_config)
        assert client._config == truba_config

    def test_uses_default_config_when_not_provided(self):
        """Client should use default config values."""
        client = TrubaClient(TrubaConfig())
        assert client._config.api_url == "https://www.trumba.com/calendars/SMU_RO_Acad.json"
        assert client._config.months_ahead == 12


# ==============================================================================
# TrubaClient fetch_events Tests
# ==============================================================================

class TestTrubaClientFetchEvents:
    """Tests for TrubaClient.fetch_events method."""

    def test_fetch_events_returns_json(self, truba_config, mock_requests):
        """fetch_events should return parsed JSON response."""
        mock_requests["response"].json.return_value = [{"id": 1, "title": "Event 1"}]

        client = TrubaClient(truba_config)
        result = client.fetch_events()

        assert result == [{"id": 1, "title": "Event 1"}]

    def test_fetch_events_includes_startdate(self, truba_config, mock_requests):
        """fetch_events should include startdate parameter (today minus 30-day lookback)."""
        with patch('src.scraper.trumba_client.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 18)
            client = TrubaClient(truba_config)
            client.fetch_events()

        call_args = mock_requests["get"].call_args
        # hardcoded 30-day lookback: 2026-05-18 minus 30 days = 2026-04-18
        assert "startdate=20260418" in call_args[0][0]

    def test_fetch_events_includes_months(self, truba_config, mock_requests):
        """fetch_events should include months parameter from config."""
        config = TrubaConfig(months_ahead=6)
        client = TrubaClient(config)
        client.fetch_events()

        call_args = mock_requests["get"].call_args
        assert "months=6" in call_args[0][0]

    def test_fetch_events_raises_on_http_error(self, truba_config, mock_requests):
        """fetch_events should raise on HTTP error."""
        mock_requests["response"].raise_for_status.side_effect = requests.HTTPError("404")

        client = TrubaClient(truba_config)
        with pytest.raises(requests.HTTPError):
            client.fetch_events()


# ==============================================================================
# TrubaClient fetch_boss_events Tests
# ==============================================================================

class TestTrubaClientFetchBossEvents:
    """Tests for TrubaClient.fetch_boss_events method."""

    def test_fetch_boss_events_filters_non_boss(self, truba_config, mock_requests):
        """Should filter out non-BOSS events."""
        mock_requests["response"].json.return_value = [
            {"title": "File for Graduation", "startDateTime": "2026-07-08T14:00:00Z"},
            {"title": "BOSS Round 1 Window 1 Results", "startDateTime": "2026-07-08T14:00:00Z"},
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert len(result) == 1
        assert result[0].title == "Round 1 Window 1"

    def test_fetch_boss_events_includes_both_window_and_results(self, truba_config, mock_requests):
        """Should return both Window and Results BOSS events."""
        mock_requests["response"].json.return_value = [
            {"title": "BOSS Round 1 Window 1", "startDateTime": "2026-07-06T10:00:00Z"},
            {"title": "BOSS Round 1 Window 1 Results", "startDateTime": "2026-07-08T14:00:00Z"},
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert len(result) == 2
        assert result[0].is_results is False  # Window event
        assert result[1].is_results is True   # Results event

    def test_fetch_boss_events_extracts_term(self, truba_config, mock_requests):
        """Should extract term from customFields."""
        mock_requests["response"].json.return_value = [
            {
                "title": "BOSS Round 1 Window 1 Results",
                "startDateTime": "2026-07-08T14:00:00Z",
                "customFields": [
                    {"label": "Term", "value": "2026-27 Term 1 - Regular Academic Session"}
                ]
            }
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert result[0].term == "AY202627T1"

    def test_fetch_boss_events_parses_abbrev(self, truba_config, mock_requests):
        """Should parse round/window abbreviation from title."""
        mock_requests["response"].json.return_value = [
            {
                "title": "BOSS Round 1A Window 2 Results",
                "startDateTime": "2026-07-13T14:00:00Z",
                "customFields": []
            }
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert result[0].abbrev == "R1AW2"

    def test_fetch_boss_events_returns_boss_event_objects(self, truba_config, mock_requests):
        """Should return list of BossEvent objects."""
        mock_requests["response"].json.return_value = [
            {
                "title": "BOSS Round 1 Window 1 Results",
                "startDateTime": "2026-07-08T14:00:00Z",
                "customFields": [
                    {"label": "Term", "value": "2026-27 Term 1 - Regular Academic Session"}
                ]
            }
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert len(result) == 1
        assert isinstance(result[0], BossEvent)

    def test_fetch_boss_events_skips_boss_non_window_events(self, truba_config, mock_requests):
        """BOSS-named events that aren't bid windows should be skipped, not aliased to R1W1."""
        mock_requests["response"].json.return_value = [
            {"title": "BOSS Course Registration", "startDateTime": "2026-07-08T14:00:00Z"},
            {"title": "BOSS Round 1 Window 1 Results", "startDateTime": "2026-07-08T14:00:00Z"},
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert len(result) == 1
        assert result[0].abbrev == "R1W1"

    def test_fetch_boss_events_normalizes_utc_to_sgt_wall_clock(self, truba_config, mock_requests):
        """UTC (Z) timestamps should be emitted as SGT wall-clock without tzinfo."""
        mock_requests["response"].json.return_value = [
            {"title": "BOSS Round 1 Window 1", "startDateTime": "2026-07-06T10:00:00Z"},
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert result[0].start_dt == "2026-07-06T18:00:00"  # UTC 10:00 -> SGT 18:00, no tzinfo
        assert result[0].end_dt is None

    def test_fetch_boss_events_keeps_naive_sgt_timestamps(self, truba_config, mock_requests):
        """Naive timestamps (already SGT wall-clock) should pass through unchanged."""
        mock_requests["response"].json.return_value = [
            {"title": "BOSS Round 1 Window 1", "startDateTime": "2026-07-06T10:00:00"},
        ]

        client = TrubaClient(truba_config)
        result = client.fetch_boss_events()

        assert result[0].start_dt == "2026-07-06T10:00:00"


class TestTrubaClientFetchBossWindows:
    """Tests for TrubaClient.fetch_boss_windows method."""

    def test_pairs_windows_with_results_and_skips_unparseable(self, truba_config, mock_requests):
        """Should pair window+results events and skip non-window BOSS events."""
        mock_requests["response"].json.return_value = [
            {"title": "BOSS Course Registration", "startDateTime": "2026-07-01T10:00:00Z"},
            {
                "title": "BOSS Round 1 Window 1",
                "startDateTime": "2026-07-06T10:00:00Z",
                "endDateTime": "2026-07-08T10:00:00Z",
            },
            {"title": "BOSS Round 1 Window 1 Results", "startDateTime": "2026-07-08T14:00:00Z"},
        ]

        client = TrubaClient(truba_config)
        windows = client.fetch_boss_windows()

        assert len(windows) == 1
        assert windows[0].abbrev == "R1W1"
        assert windows[0].opens_at == "2026-07-06T18:00:00"
        assert windows[0].closes_at == "2026-07-08T18:00:00"
        assert windows[0].results_at == "2026-07-08T22:00:00"


# ==============================================================================
# TrubaClient _extract_term Tests
# ==============================================================================

class TestTrubaClientExtractTerm:
    """Tests for TrubaClient._extract_term method."""

    @pytest.mark.parametrize("term_text,expected", [
        ("2026-27 Term 1 - Regular Academic Session", "AY202627T1"),
        ("2025-26 Term 2 - Regular Academic Session", "AY202526T2"),
        ("2025-26 Term 3A - Regular Academic Session", "AY202526T3A"),
        ("2025-26 Term 3B - Regular Academic Session", "AY202526T3B"),
        ("AY2026-27 Term 1 - Regular Academic Session", "AY202627T1"),
        ("2025-26 Term 1 - 1A Session", "AY202526T1"),
    ])
    def test_extract_valid_terms(self, truba_config, term_text, expected):
        """Should extract term identifiers correctly."""
        client = TrubaClient(truba_config)
        event_data = {
            "customFields": [{"label": "Term", "value": term_text}]
        }
        result = client._extract_term(event_data)
        assert result == expected

    def test_extract_term_fallback_to_date_inference(self, truba_config):
        """Should fall back to date inference when term not found."""
        client = TrubaClient(truba_config)
        event_data = {"customFields": []}
        result = client._extract_term(event_data)
        assert result.startswith("AY")

    def test_extract_term_returns_ay_format(self):
        """Trumba '2026-27 Term 1 ...' is parsed directly to AY — no dash intermediate."""
        client = TrubaClient(TrubaConfig())
        term = client._extract_term({
            'customFields': [{'label': 'Term', 'value': '2026-27 Term 1 - Regular Academic Session'}]
        })
        assert term == 'AY202627T1'


# ==============================================================================
# TrubaClient _infer_term_from_date Tests
# ==============================================================================

class TestTrubaClientInferTermFromDate:
    """Tests for TrubaClient._infer_term_from_date method."""

    @pytest.mark.parametrize("month,expected_term_suffix", [
        (8, "T1"),   # August -> Term 1
        (9, "T1"),   # September -> Term 1
        (10, "T1"),  # October -> Term 1
        (11, "T1"),  # November -> Term 1
        (12, "T1"),  # December -> Term 1
        (1, "T2"),   # January -> Term 2
        (2, "T2"),   # February -> Term 2
        (3, "T2"),   # March -> Term 2
        (4, "T2"),   # April -> Term 2
        (5, "T3A"),  # May -> Term 3A
        (6, "T3B"),  # June -> Term 3B
        (7, "T3B"),  # July -> Term 3B
    ])
    def test_infer_term_by_month(self, truba_config, month, expected_term_suffix):
        """Should infer correct term based on month."""
        with patch('src.scraper.trumba_client.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, month, 15)
            client = TrubaClient(truba_config)
            result = client._infer_term_from_date()
            assert result.endswith(expected_term_suffix)


# ==============================================================================
# TrubaClient _parse_title Tests
# ==============================================================================

class TestTrubaClientParseTitle:
    """Tests for TrubaClient._parse_title method."""

    @pytest.mark.parametrize("title,expected_abbrev", [
        ("BOSS Round 1 Window 1", "R1W1"),
        ("BOSS Round 1A Window 1", "R1AW1"),
        ("BOSS Round 1B Window 2", "R1BW2"),
        ("BOSS Round 2 Window 3", "R2W3"),
        ("BOSS Round 1 Window 1 Results", "R1W1"),
        ("BOSS Round 1A Window 1 Results", "R1AW1"),
        ("BOSS Round 1B Window 2 Results", "R1BW2"),
        ("BOSS Round 2 Window 3 Results", "R2W3"),
        ("BOSS Round 1A Window 2", "R1AW2"),
        ("BOSS Round 2 Window 4", "R2W4"),
        ("BOSS Round 2A Window 1", "R2AW1"),
    ])
    def test_parse_valid_titles(self, truba_config, title, expected_abbrev):
        """Should parse BOSS titles into window abbrevs."""
        client = TrubaClient(truba_config)
        assert client._parse_title(title) == expected_abbrev

    @pytest.mark.parametrize("title,expected_abbrev", [
        ("BOSS Incoming Freshmen Round 1 Window 1", "R1FW1"),
        ("BOSS Incoming Freshmen Rnd 1 Win 4", "R1FW4"),
        ("BOSS Incoming Exchange Round 1C Window 1", "R1CW1"),
        ("BOSS Incoming Exchange Rnd 2 Win 3", "R2W3"),
    ])
    def test_parse_incoming_titles(self, truba_config, title, expected_abbrev):
        """Incoming Freshmen/Exchange titles should keep their distinguishing suffixes."""
        client = TrubaClient(truba_config)
        assert client._parse_title(title) == expected_abbrev

    def test_parse_unusual_format_returns_none(self, truba_config):
        """Unparseable titles should return None, not a bogus R1W1."""
        client = TrubaClient(truba_config)
        assert client._parse_title("BOSS Event") is None
        assert client._parse_title("BOSS Course Registration") is None
        assert client._parse_title("File for Graduation") is None


# ==============================================================================
# EventBridge Scheduler Tests
# ==============================================================================

class TestEventBridgeScheduler:
    """Tests for EventBridgeScheduler class from Lambda."""

    def test_schedule_exists_true(self, mock_eventbridge_client):
        """schedule_exists should return True when schedule exists."""
        mock_eventbridge_client.get_schedule.side_effect = None
        mock_eventbridge_client.get_schedule.return_value = {"Name": "test"}

        lambda_module = _load_lambda_module()
        scheduler = lambda_module.EventBridgeScheduler(
            region="ap-southeast-1",
            cluster_arn="arn:ecs:cluster",
            task_def_arn="arn:ecs:task",
            scheduler_role_arn="arn:iam:role",
            subnets=["subnet-1"],
            security_groups=["sg-1"]
        )
        scheduler._scheduler = mock_eventbridge_client

        result = scheduler.schedule_exists("test-schedule")
        assert result is True

    def test_schedule_exists_false(self, mock_eventbridge_client):
        """schedule_exists should return False when schedule not found."""
        lambda_module = _load_lambda_module()
        scheduler = lambda_module.EventBridgeScheduler(
            region="ap-southeast-1",
            cluster_arn="arn:ecs:cluster",
            task_def_arn="arn:ecs:task",
            scheduler_role_arn="arn:iam:role",
            subnets=["subnet-1"],
            security_groups=["sg-1"]
        )
        scheduler._scheduler = mock_eventbridge_client

        result = scheduler.schedule_exists("nonexistent-schedule")
        assert result is False

    def test_create_schedule_calls_api(self, mock_eventbridge_client):
        """create_schedule should create primary + 2 retry schedules with correct params."""
        lambda_module = _load_lambda_module()
        scheduler = lambda_module.EventBridgeScheduler(
            region="ap-southeast-1",
            cluster_arn="arn:ecs:cluster",
            task_def_arn="arn:ecs:task",
            scheduler_role_arn="arn:iam:role",
            subnets=["subnet-1"],
            security_groups=["sg-1"]
        )
        scheduler._scheduler = mock_eventbridge_client

        scrape_time = datetime(2026, 7, 6, 10, 0)
        scheduler.create_schedule(
            schedule_name="bidlysmu-pipeline-AY202627T1-R1W1",
            scrape_time=scrape_time,
            term="AY202627T1",
            abbrev="R1W1",
            results_datetime="2026-07-08T14:00:00"
        )

        # Primary + retry1 (+2h) + retry2 (+4h)
        assert mock_eventbridge_client.create_schedule.call_count == 3
        calls = mock_eventbridge_client.create_schedule.call_args_list

        # Primary: SGT scrape_time (10:00) shifted -8h for EventBridge at() (UTC)
        primary = calls[0].kwargs
        assert primary["Name"] == "bidlysmu-pipeline-AY202627T1-R1W1"
        assert primary["ScheduleExpression"] == "at(2026-07-06T02:00:00)"
        primary_input = json.loads(primary["Target"]["Input"])
        assert primary_input["retry_attempt"] == 0
        assert primary_input["window"] == "R1W1"
        assert primary_input["acad_term_id"] == "AY202627T1"

        retry1 = calls[1].kwargs
        assert retry1["Name"] == "bidlysmu-pipeline-AY202627T1-R1W1-retry1"
        assert retry1["ScheduleExpression"] == "at(2026-07-06T04:00:00)"
        assert json.loads(retry1["Target"]["Input"])["retry_attempt"] == 1

        retry2 = calls[2].kwargs
        assert retry2["Name"] == "bidlysmu-pipeline-AY202627T1-R1W1-retry2"
        assert retry2["ScheduleExpression"] == "at(2026-07-06T06:00:00)"
        assert json.loads(retry2["Target"]["Input"])["retry_attempt"] == 2


# ==============================================================================
# Bidding Schedule Manager Tests
# ==============================================================================

class TestBiddingScheduleManager:
    """Tests for BiddingScheduleManager class from Lambda."""

    def test_merge_with_deduplication_no_duplicates(self, mock_supabase_client):
        """Should add all events when no duplicates exist."""
        lambda_module = _load_lambda_module()
        manager = lambda_module.BiddingScheduleManager(mock_supabase_client)

        existing = {}
        new_events = [
            BossEvent(term="AY202627T1", start_dt="2026-07-08T14:00:00", abbrev="R1W1", title="Round 1 Window 1", is_results=True),
            BossEvent(term="AY202627T1", start_dt="2026-07-10T14:00:00", abbrev="R1AW1", title="Round 1A Window 1", is_results=True),
        ]

        result = manager.merge_with_deduplication(existing, new_events)

        assert "AY202627T1" in result
        assert len(result["AY202627T1"]) == 2

    def test_merge_skips_duplicates(self, mock_supabase_client, sample_bidding_schedules):
        """Should skip events that already exist."""
        lambda_module = _load_lambda_module()
        manager = lambda_module.BiddingScheduleManager(mock_supabase_client)

        new_events = [
            BossEvent(term="AY202627T1", start_dt="2026-07-08T14:00:00", abbrev="R1W1", title="Round 1 Window 1", is_results=True),
            BossEvent(term="AY202627T1", start_dt="2026-07-15T14:00:00", abbrev="R1BW1", title="Round 1B Window 1", is_results=True),
        ]

        # Capture the count BEFORE merging — merge_with_deduplication appends to
        # the existing list (a shallow copy shares the inner list with the fixture).
        original_count = len(sample_bidding_schedules["AY202627T1"])
        result = manager.merge_with_deduplication(sample_bidding_schedules.copy(), new_events)

        assert len(result["AY202627T1"]) == original_count + 1
        assert result["AY202627T1"][-1][2] == "R1BW1"

    def test_merge_sorts_by_datetime(self, mock_supabase_client):
        """Should sort events by datetime within each term."""
        lambda_module = _load_lambda_module()
        manager = lambda_module.BiddingScheduleManager(mock_supabase_client)

        existing = {"AY202627T1": [["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1"]]}
        new_events = [
            BossEvent(term="AY202627T1", start_dt="2026-07-08T14:00:00", abbrev="R1W1", title="Round 1 Window 1", is_results=True),
        ]

        result = manager.merge_with_deduplication(existing, new_events)

        assert result["AY202627T1"][0][2] == "R1W1"
        assert result["AY202627T1"][1][2] == "R1AW1"


# ==============================================================================
# Schedule Tracker Tests
# ==============================================================================

class TestScheduleTracker:
    """Tests for ScheduleTracker class from Lambda."""

    def test_is_tracked_true(self, mock_supabase_client):
        """is_tracked should return True for tracked schedule."""
        lambda_module = _load_lambda_module()
        tracker = lambda_module.ScheduleTracker(mock_supabase_client)

        tracking = {
            "AY202627T1": {
                "bidlysmu-pipeline-AY202627T1-R1W1": {
                    "created_at": "2026-05-01T10:00:00",
                    "scrape_time": "2026-06-22T10:00:00",
                    "results_datetime": "2026-07-08T14:00:00"
                }
            }
        }

        result = tracker.is_tracked(tracking, "AY202627T1", "bidlysmu-pipeline-AY202627T1-R1W1")
        assert result is True

    def test_is_tracked_false(self, mock_supabase_client):
        """is_tracked should return False for untracked schedule."""
        lambda_module = _load_lambda_module()
        tracker = lambda_module.ScheduleTracker(mock_supabase_client)

        tracking = {"AY202627T1": {}}

        result = tracker.is_tracked(tracking, "AY202627T1", "nonexistent-schedule")
        assert result is False

    def test_add_to_tracking(self, mock_supabase_client):
        """add_to_tracking should update tracking dict."""
        lambda_module = _load_lambda_module()
        tracker = lambda_module.ScheduleTracker(mock_supabase_client)

        tracking = {}
        scrape_time = datetime(2026, 6, 22, 10, 0)

        result = tracker.add_to_tracking(
            tracking, "AY202627T1", "bidlysmu-pipeline-AY202627T1-R1W1",
            scrape_time, "2026-07-08T14:00:00"
        )

        assert "AY202627T1" in result
        assert "bidlysmu-pipeline-AY202627T1-R1W1" in result["AY202627T1"]


# ==============================================================================
# Calculate Scrape Time Tests
# ==============================================================================

class TestCalculateScrapeTime:
    """Tests for calculate_scrape_time function from Lambda."""

    def test_first_window_scrape_time(self):
        """First window (R1W1) should scrape 2 weeks before opens_at."""
        lambda_module = _load_lambda_module()
        windows = [
            ["2026-07-08T14:00:00", "Round 1 Window 1", "R1W1", "2026-07-06T10:00:00", "2026-07-08T10:00:00"],
            ["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1"],
        ]

        result = lambda_module.calculate_scrape_time(windows, 0)

        # opens_at 2026-07-06 10:00 minus 2 weeks
        expected = datetime(2026, 6, 22, 10, 0)
        assert result == expected

    def test_subsequent_window_scrape_time(self):
        """Subsequent windows (legacy format) should scrape 1 hour after previous results."""
        lambda_module = _load_lambda_module()
        windows = [
            ["2026-07-08T14:00:00", "Round 1 Window 1", "R1W1"],
            ["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1"],
        ]

        result = lambda_module.calculate_scrape_time(windows, 1)

        expected = datetime(2026, 7, 8, 15, 0)
        assert result == expected

    def test_third_window_scrape_time(self):
        """Third window (legacy format) should scrape 1 hour after second results."""
        lambda_module = _load_lambda_module()
        windows = [
            ["2026-07-08T14:00:00", "Round 1 Window 1", "R1W1"],
            ["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1"],
            ["2026-07-13T14:00:00", "Round 1A Window 2", "R1AW2"],
        ]

        result = lambda_module.calculate_scrape_time(windows, 2)

        expected = datetime(2026, 7, 10, 15, 0)
        assert result == expected


    def test_subsequent_window_extended_format_uses_opens_at(self):
        """Subsequent windows with extended format should scrape 1 hour before opens_at."""
        lambda_module = _load_lambda_module()
        windows = [
            ["2026-07-08T14:00:00", "Round 1 Window 1", "R1W1", "2026-07-06T10:00:00", "2026-07-08T10:00:00"],
            ["2026-07-10T14:00:00", "Round 1A Window 1", "R1AW1", "2026-07-08T17:00:00", "2026-07-10T10:00:00"],
        ]

        result = lambda_module.calculate_scrape_time(windows, 1)

        # opens_at 17:00 - 1h
        expected = datetime(2026, 7, 8, 16, 0)
        assert result == expected