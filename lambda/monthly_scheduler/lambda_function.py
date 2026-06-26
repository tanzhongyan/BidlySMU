"""
Monthly scheduler Lambda function.

Uses Truba JSON API to fetch BOSS bidding events without web scraping.
No authentication or Selenium required - just HTTP requests.

Architecture:
1. Calls Truba JSON API to fetch calendar events (via TrubaClient)
2. Extracts BOSS Results events
3. Updates bidding_schedules.json in Supabase Storage (with deduplication)
4. Creates EventBridge schedules for each bidding window (with deduplication)

Usage:
    This Lambda is deployed as a container image.
    It runs monthly to:
    1. Fetch BOSS events from Truba JSON API
    2. Update bidding_schedules.json in Supabase Storage (with deduplication)
    3. Create EventBridge schedules for each bidding window (with deduplication)
"""
import json
import boto3
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add src to path for Lambda container
import sys
sys.path.insert(0, "/app")

from src.scraper.trumba_client import TrubaClient, TrubaConfig

# Supabase
from supabase import create_client

# Configuration from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
TRUBA_API_URL = os.environ.get("TRUBA_API_URL", "https://www.trumba.com/calendars/SMU_RO_Acad.json")
MONTHS_AHEAD = int(os.environ.get("MONTHS_AHEAD", "12"))

# AWS configuration from environment
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ECS_CLUSTER_ARN = os.environ.get("ECS_CLUSTER_ARN")
ECS_TASK_DEF_ARN = os.environ.get("ECS_TASK_DEF_ARN")
SCHEDULER_ROLE_ARN = os.environ.get("SCHEDULER_ROLE_ARN")
SUBNETS = os.environ.get("SUBNETS", "").split(",") if os.environ.get("SUBNETS") else []
SECURITY_GROUPS = os.environ.get("SECURITY_GROUPS", "").split(",") if os.environ.get("SECURITY_GROUPS") else []
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "bidlysmu-pipeline")


def convert_term_to_acad_term_id(term: str) -> str:
    """
    Convert Truba term format to BOSS database ACAD_TERM_ID format.

    Args:
        term: Truba format (e.g., "2026-27_T1", "2025-26_T3A")

    Returns:
        BOSS database format (e.g., "AY202627T1", "AY202526T3A")

    Examples:
        "2026-27_T1" -> "AY202627T1"
        "2025-26_T3A" -> "AY202526T3A"
        "2025-26_T3B" -> "AY202526T3B"
    """
    # Handle format like '2025-26_T1' or '2025-26_T3A'
    match = re.match(r'(\d{4})-(\d{2})_T(\d+)([A-B]?)', term)
    if match:
        start_year = match.group(1)  # 2025
        end_year_suffix = match.group(2)  # 26
        term_num = match.group(3)  # 1, 2, 3
        term_suffix = match.group(4)  # A, B, or empty

        # Construct database format: AY + start_year + end_year_suffix + T + term_num + suffix
        return f"AY{start_year}{end_year_suffix}T{term_num}{term_suffix}"

    # If already in correct format or unknown format, return as-is
    return term


class BiddingScheduleManager:
    """
    Manages bidding_schedules.json in Supabase Storage with deduplication.
    """

    def __init__(self, supabase_client, bucket: str = "bidlysmu-files"):
        self._supabase = supabase_client
        self._bucket = bucket

    def download_existing_schedules(self) -> Dict:
        """Download existing bidding_schedules.json from Supabase Storage."""
        try:
            data = self._supabase.storage.from_(self._bucket).download(
                "schedules/bidding_schedules.json"
            )
            schedules = json.loads(data.decode('utf-8'))
            logger.info(f"Downloaded {len(schedules)} terms from existing schedules")
            return schedules
        except Exception as e:
            logger.warning(f"Could not download existing schedules: {e}")
            return {}

    def upload_schedules(self, schedules: Dict) -> None:
        """Upload updated bidding_schedules.json to Supabase Storage."""
        content = json.dumps(schedules, indent=2).encode('utf-8')
        self._supabase.storage.from_(self._bucket).upload(
            "schedules/bidding_schedules.json",
            content,
            file_options={"content-type": "application/json", "upsert": "true"}
        )
        logger.info(f"Uploaded schedules: {len(schedules)} terms")

    def merge_with_deduplication(self, existing: Dict, new_events: List) -> Dict:
        """Merge new events into existing schedules with deduplication."""
        # Build lookup set of existing abbreviations per term
        existing_lookup = {}
        for term, windows in existing.items():
            existing_lookup[term] = {window[2] for window in windows}

        added_count = 0
        # Filter out duplicates and merge
        for event in new_events:
            term = event.term
            abbrev = event.abbrev

            if term not in existing_lookup or abbrev not in existing_lookup[term]:
                if term not in existing:
                    existing[term] = []
                existing[term].append([
                    event.datetime,
                    event.title,
                    event.abbrev
                ])
                logger.info(f"Added new event: {term} | {abbrev}")
                added_count += 1
            else:
                logger.debug(f"Skipping duplicate: {term} | {abbrev}")

        # Sort by datetime within each term
        for term in existing:
            existing[term].sort(key=lambda x: x[0])

        logger.info(f"Merge complete: {added_count} new events added")
        return existing


class EventBridgeScheduler:
    """Manages EventBridge schedule creation with deduplication."""

    def __init__(
        self,
        region: str,
        cluster_arn: str,
        task_def_arn: str,
        scheduler_role_arn: str,
        subnets: List[str],
        security_groups: List[str]
    ):
        self._scheduler = boto3.client("scheduler", region_name=region)
        self._cluster_arn = cluster_arn
        self._task_def_arn = task_def_arn
        self._scheduler_role_arn = scheduler_role_arn
        self._subnets = subnets
        self._security_groups = security_groups

    def schedule_exists(self, schedule_name: str) -> bool:
        """Check if schedule exists in EventBridge."""
        try:
            self._scheduler.get_schedule(Name=schedule_name)
            return True
        except self._scheduler.exceptions.ResourceNotFoundException:
            return False
        except Exception:
            return False

    def create_schedule(
        self,
        schedule_name: str,
        scrape_time: datetime,
        term: str,
        abbrev: str,
        results_datetime: str
    ) -> None:
        """Create a one-time EventBridge schedule for pipeline execution."""
        # Convert term to ACAD_TERM_ID format
        acad_term_id = convert_term_to_acad_term_id(term)

        # Create schedule with environment variables for ECS task
        self._scheduler.create_schedule(
            Name=schedule_name,
            ScheduleExpression=f"at({scrape_time.strftime('%Y-%m-%dT%H:%M:%S')})",
            FlexibleTimeWindow={"Mode": "OFF"},
            Target={
                "Arn": self._cluster_arn,
                "RoleArn": self._scheduler_role_arn,
                "EcsParameters": {
                    "TaskDefinitionArn": self._task_def_arn,
                    "LaunchType": "FARGATE",
                    "NetworkConfiguration": {
                        "awsvpcConfiguration": {
                            "Subnets": self._subnets,
                            "SecurityGroups": self._security_groups,
                            "AssignPublicIp": "DISABLED"
                        }
                    },
                    # Pass environment variables to ECS container
                    "Overrides": {
                        "ContainerOverrides": [
                            {
                                "Name": ECS_CONTAINER_NAME,
                                "Environment": [
                                    {"Name": "ACAD_TERM_ID", "Value": acad_term_id},
                                    {"Name": "CURRENT_WINDOW_NAME", "Value": abbrev},
                                    {"Name": "RESULTS_DATETIME", "Value": results_datetime}
                                ]
                            }
                        ]
                    }
                },
                "Input": json.dumps({
                    "term": term,
                    "acad_term_id": acad_term_id,
                    "window": abbrev,
                    "results_datetime": results_datetime
                })
            },
            ActionAfterCompletion="DELETE"
        )
        logger.info(f"Created schedule: {schedule_name} with ACAD_TERM_ID={acad_term_id}")


class ScheduleTracker:
    """Tracks created EventBridge schedules in Supabase Storage."""

    def __init__(self, supabase_client, bucket: str = "bidlysmu-files"):
        self._supabase = supabase_client
        self._bucket = bucket

    def download_tracking_file(self) -> Dict:
        """Download existing_schedules.json tracking file."""
        try:
            data = self._supabase.storage.from_(self._bucket).download(
                "schedules/existing_schedules.json"
            )
            return json.loads(data.decode('utf-8'))
        except Exception:
            return {}

    def upload_tracking_file(self, tracking: Dict) -> None:
        """Upload updated tracking file."""
        content = json.dumps(tracking, indent=2).encode('utf-8')
        self._supabase.storage.from_(self._bucket).upload(
            "schedules/existing_schedules.json",
            content,
            file_options={"content-type": "application/json", "upsert": "true"}
        )

    def is_tracked(self, tracking: Dict, term: str, schedule_name: str) -> bool:
        """Check if schedule is already in tracking file."""
        return term in tracking and schedule_name in tracking.get(term, {})

    def add_to_tracking(
        self,
        tracking: Dict,
        term: str,
        schedule_name: str,
        scrape_time: datetime,
        results_datetime: str
    ) -> Dict:
        """Add schedule to tracking file."""
        if term not in tracking:
            tracking[term] = {}

        tracking[term][schedule_name] = {
            "created_at": datetime.now().isoformat(),
            "scrape_time": scrape_time.isoformat(),
            "results_datetime": results_datetime
        }
        return tracking


def calculate_scrape_time(windows: List, window_index: int) -> datetime:
    """
    Calculate when to scrape based on timing logic.

    - R1W1 (first window): Scrape 2 weeks before window starts
    - Subsequent windows: Scrape 3 hours after previous results
    """
    current_results = datetime.fromisoformat(windows[window_index][0])

    if window_index == 0:
        window_start = current_results - timedelta(days=2)
        return window_start - timedelta(weeks=2)
    else:
        previous_results = datetime.fromisoformat(windows[window_index - 1][0])
        return previous_results + timedelta(hours=3)


def lambda_handler(event, context):
    """
    Main Lambda handler - fetches BOSS events from Truba API.

    Uses TrubaClient for all API interaction (DRY principle).
    """
    try:
        # Initialize Supabase client
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Create TrubaClient (shared implementation)
        truba_config = TrubaConfig(
            api_url=TRUBA_API_URL,
            months_ahead=MONTHS_AHEAD
        )
        truba_client = TrubaClient(truba_config)

        # Fetch BOSS events
        logger.info(f"Fetching Truba API: {TRUBA_API_URL}")
        boss_events = truba_client.fetch_boss_events()
        logger.info(f"Extracted {len(boss_events)} BOSS bidding events")

        if not boss_events:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No new BOSS events found",
                    "events_found": 0
                })
            }

        # Update bidding_schedules.json with deduplication
        schedule_manager = BiddingScheduleManager(supabase)
        existing_schedules = schedule_manager.download_existing_schedules()
        updated_schedules = schedule_manager.merge_with_deduplication(
            existing_schedules, boss_events
        )
        schedule_manager.upload_schedules(updated_schedules)

        # Create EventBridge schedules with deduplication
        schedules_created = []

        if ECS_CLUSTER_ARN and ECS_TASK_DEF_ARN and SCHEDULER_ROLE_ARN:
            scheduler = EventBridgeScheduler(
                region=AWS_REGION,
                cluster_arn=ECS_CLUSTER_ARN,
                task_def_arn=ECS_TASK_DEF_ARN,
                scheduler_role_arn=SCHEDULER_ROLE_ARN,
                subnets=SUBNETS,
                security_groups=SECURITY_GROUPS
            )

            tracker = ScheduleTracker(supabase)
            tracking_data = tracker.download_tracking_file()

            for term, windows in updated_schedules.items():
                for i, window in enumerate(windows):
                    abbrev = window[2]
                    schedule_name = f"bidlysmu-pipeline-{term}-{abbrev}"

                    # Level 1: Check tracking file (fast)
                    if tracker.is_tracked(tracking_data, term, schedule_name):
                        continue

                    # Level 2: Check EventBridge API (authoritative)
                    if scheduler.schedule_exists(schedule_name):
                        continue

                    # Calculate scrape time
                    scrape_time = calculate_scrape_time(windows, i)

                    # Skip if in the past
                    if scrape_time < datetime.now():
                        continue

                    # Create schedule
                    scheduler.create_schedule(
                        schedule_name=schedule_name,
                        scrape_time=scrape_time,
                        term=term,
                        abbrev=abbrev,
                        results_datetime=window[0]
                    )

                    # Update tracking
                    tracking_data = tracker.add_to_tracking(
                        tracking_data, term, schedule_name, scrape_time, window[0]
                    )
                    schedules_created.append(schedule_name)

            # Upload updated tracking file
            tracker.upload_tracking_file(tracking_data)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Successfully updated schedules",
                "events_found": len(boss_events),
                "schedules_created": schedules_created,
                "total_windows": sum(len(w) for w in updated_schedules.values())
            })
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
