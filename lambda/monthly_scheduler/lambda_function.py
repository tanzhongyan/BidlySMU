"""
BidlySMU Lambda Scheduler — Dual-mode handler.

Mode 1 — Monthly calendar fetch (trigger: "monthly-schedule" or default):
  1. Fetch BOSS events from Trumba JSON API
  2. Update bidding_schedules.json in Supabase Storage
  3. Create one-time EventBridge schedules that invoke THIS Lambda in Mode 2

Mode 2 — Pipeline run trigger (trigger: "run_pipeline"):
  1. Receive acad_term_id + window from the schedule payload
  2. Call ecs.run_task() with environment overrides for that term/window
"""
import json
import boto3
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import sys
sys.path.insert(0, "/app")

from src.scraper.trumba_client import TrubaClient, TrubaConfig
from src.config import dash_format_to_acad_term_id
from supabase import create_client

# --- Environment configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
TRUBA_API_URL = os.environ.get("TRUBA_API_URL", "https://www.trumba.com/calendars/SMU_RO_Acad.json")
MONTHS_AHEAD = int(os.environ.get("MONTHS_AHEAD", "12"))

AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ECS_CLUSTER_ARN = os.environ.get("ECS_CLUSTER_ARN")
ECS_TASK_DEF_ARN = os.environ.get("ECS_TASK_DEF_ARN")
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "bidlysmu-pipeline")
SUBNETS = os.environ.get("SUBNETS", "").split(",") if os.environ.get("SUBNETS") else []
SECURITY_GROUPS = os.environ.get("SECURITY_GROUPS", "").split(",") if os.environ.get("SECURITY_GROUPS") else []
ASSIGN_PUBLIC_IP = os.environ.get("ASSIGN_PUBLIC_IP", "ENABLED")
LAMBDA_INVOKE_ROLE_ARN = os.environ.get("LAMBDA_INVOKE_ROLE_ARN", "")


# =============================================================================
# Term format conversion
# =============================================================================



# =============================================================================
# Scrape time calculation
# =============================================================================

def calculate_scrape_time(windows: List, window_index: int) -> datetime:
    """Calculate when to scrape based on timing logic.

    - First window: 2 weeks before window starts
    - Subsequent windows: 1 hour before the window opens (using actual
      opens_at from Trumba), or 1 hour after previous results as fallback.
      This guarantees the ECS pipeline completes before bidding starts.
    """
    current_entry = windows[window_index]
    current_results = datetime.fromisoformat(current_entry[0])

    if window_index == 0:
        # First window: 2 weeks before it opens (gives time to fix issues)
        window_start = current_results - timedelta(days=2)
        return window_start - timedelta(weeks=2)

    # Subsequent windows: prefer 1 hour before opens_at (extended format index 3)
    if len(current_entry) >= 5 and current_entry[3]:
        opens_at = datetime.fromisoformat(current_entry[3])
        return opens_at - timedelta(hours=1)

    # Fallback: 1 hour after previous window's results
    previous_results = datetime.fromisoformat(windows[window_index - 1][0])
    return previous_results + timedelta(hours=1)


# =============================================================================
# BiddingScheduleManager — deduplication for bidding_schedules.json
# =============================================================================

class BiddingScheduleManager:
    """Manages bidding_schedules.json in Supabase Storage."""

    def __init__(self, supabase_client, bucket: str = "bidlysmu-files"):
        self._supabase = supabase_client
        self._bucket = bucket

    def download_existing_schedules(self) -> Dict:
        try:
            data = self._supabase.storage.from_(self._bucket).download(
                "schedules/bidding_schedules.json"
            )
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not download existing schedules: {e}")
            return {}

    def upload_schedules(self, schedules: Dict) -> None:
        content = json.dumps(schedules, indent=2).encode('utf-8')
        self._supabase.storage.from_(self._bucket).upload(
            "schedules/bidding_schedules.json", content,
            file_options={"content-type": "application/json", "upsert": "true"}
        )
        logger.info("Uploaded bidding_schedules.json to Supabase Storage")

    def merge_with_deduplication(self, existing: Dict, new_windows: List) -> Dict:
        """Merge BossWindow objects into schedules, deduplicating by term + abbrev.

        New format (extended, backward-compatible):
            [results_at, title, abbrev, opens_at, closes_at]
        Old code reading index 0-2 still gets results/title/abbrev.
        """
        existing_lookup = {}
        for term, windows in existing.items():
            for window in windows:
                abbrev = window[2] if len(window) >= 3 else None
                if abbrev:
                    existing_lookup.setdefault(term, set()).add(abbrev)

        added_count = 0
        for bw in new_windows:
            term = bw.term
            abbrev = bw.abbrev
            if term not in existing_lookup or abbrev not in existing_lookup.get(term, set()):
                if term not in existing:
                    existing[term] = []
                existing[term].append(bw.to_schedule_entry())
                added_count += 1

        logger.info(f"Merged {len(new_windows)} windows: {added_count} new, {len(new_windows) - added_count} duplicates")
        return existing


# =============================================================================
# ScheduleTracker — tracks which schedules have been created
# =============================================================================

class ScheduleTracker:
    """Tracks created EventBridge schedules in Supabase Storage."""

    def __init__(self, supabase_client, bucket: str = "bidlysmu-files"):
        self._supabase = supabase_client
        self._bucket = bucket

    def download_tracking_file(self) -> Dict:
        try:
            data = self._supabase.storage.from_(self._bucket).download(
                "schedules/existing_schedules.json"
            )
            return json.loads(data.decode('utf-8'))
        except Exception:
            return {}

    def upload_tracking_file(self, tracking: Dict) -> None:
        content = json.dumps(tracking, indent=2).encode('utf-8')
        self._supabase.storage.from_(self._bucket).upload(
            "schedules/existing_schedules.json", content,
            file_options={"content-type": "application/json", "upsert": "true"}
        )

    def is_tracked(self, tracking: Dict, term: str, schedule_name: str) -> bool:
        return term in tracking and schedule_name in tracking.get(term, {})

    def add_to_tracking(self, tracking: Dict, term: str, schedule_name: str,
                        scrape_time: datetime, results_datetime: str) -> Dict:
        if term not in tracking:
            tracking[term] = {}
        tracking[term][schedule_name] = {
            "created_at": datetime.now().isoformat(),
            "scrape_time": scrape_time.isoformat(),
            "results_datetime": results_datetime
        }
        return tracking


# =============================================================================
# Dual-mode Lambda handler
# =============================================================================

def lambda_handler(event, context):
    """Dual-mode handler: monthly calendar fetch OR pipeline run trigger."""
    try:
        trigger = event.get("trigger", "monthly-schedule")

        # =============================================================
        # MODE 2: Pipeline run — start ECS task with overrides
        # =============================================================
        if trigger == "run_pipeline":
            acad_term_id = event.get("acad_term_id")
            window = event.get("window")
            results_datetime = event.get("results_datetime", "")

            if not acad_term_id or not window:
                raise ValueError("run_pipeline trigger requires acad_term_id and window")

            logger.info(f"Mode 2: Starting ECS task for {acad_term_id}/{window}")

            ecs = boto3.client("ecs", region_name=AWS_REGION)
            resp = ecs.run_task(
                cluster=ECS_CLUSTER_ARN,
                taskDefinition=ECS_TASK_DEF_ARN,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": SUBNETS,
                        "securityGroups": SECURITY_GROUPS,
                        "assignPublicIp": ASSIGN_PUBLIC_IP
                    }
                },
                overrides={
                    "containerOverrides": [{
                        "name": ECS_CONTAINER_NAME,
                        "environment": [
                            {"name": "ACAD_TERM_ID", "value": acad_term_id},
                            {"name": "CURRENT_WINDOW_NAME", "value": window},
                            {"name": "RESULTS_DATETIME", "value": results_datetime}
                        ]
                    }]
                }
            )

            task_arn = resp["tasks"][0]["taskArn"] if resp.get("tasks") else "unknown"
            logger.info(f"ECS task started: {task_arn}")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": f"ECS task started for {acad_term_id}/{window}",
                    "task_arn": task_arn
                })
            }

        # =============================================================
        # MODE 1: Monthly calendar fetch + schedule creation
        # =============================================================
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Fetch BOSS windows from Trumba (paired: opens + closes + results)
        truba = TrubaClient(TrubaConfig(
            api_url=TRUBA_API_URL, months_ahead=MONTHS_AHEAD
        ))
        logger.info(f"Fetching Trumba: {TRUBA_API_URL}")
        boss_windows = truba.fetch_boss_windows()
        logger.info(f"Found {len(boss_windows)} BOSS windows")

        if not boss_windows:
            return {"statusCode": 200, "body": json.dumps({"message": "No BOSS windows found", "events_found": 0})}

        # Update bidding_schedules.json
        mgr = BiddingScheduleManager(supabase)
        existing = mgr.download_existing_schedules()
        updated = mgr.merge_with_deduplication(existing, boss_windows)
        mgr.upload_schedules(updated)

        # Create one-time EventBridge schedules → invoke THIS Lambda in Mode 2
        schedules_created = []
        lambda_arn = (context.invoked_function_arn if context
                      else os.environ.get("AWS_LAMBDA_FUNCTION_ARN", ""))

        if LAMBDA_INVOKE_ROLE_ARN:
            sched = boto3.client("scheduler", region_name=AWS_REGION)
            tracker = ScheduleTracker(supabase)
            tracking = tracker.download_tracking_file()

            for term, windows in updated.items():
                for i, window in enumerate(windows):
                    abbrev = window[2]
                    name = f"bidlysmu-pipeline-{term}-{abbrev}".replace(" ", "-").replace("/", "-")

                    if tracker.is_tracked(tracking, term, name):
                        continue
                    try:
                        sched.get_schedule(Name=name)
                        continue
                    except sched.exceptions.ResourceNotFoundException:
                        pass

                    scrape_time = calculate_scrape_time(windows, i)
                    if scrape_time < datetime.now():
                        continue

                    acad_term_id = dash_format_to_acad_term_id(term)
                    sched.create_schedule(
                        Name=name,
                        ScheduleExpression=f"at({scrape_time.strftime('%Y-%m-%dT%H:%M:%S')})",
                        FlexibleTimeWindow={"Mode": "OFF"},
                        Target={
                            "Arn": lambda_arn,
                            "RoleArn": LAMBDA_INVOKE_ROLE_ARN,
                            "Input": json.dumps({
                                "trigger": "run_pipeline",
                                "acad_term_id": acad_term_id,
                                "window": abbrev,
                                "results_datetime": window[0]
                            })
                        },
                        ActionAfterCompletion="DELETE"
                    )
                    tracking = tracker.add_to_tracking(tracking, term, name, scrape_time, window[0])
                    schedules_created.append(name)
                    logger.info(f"Schedule: {name} at {scrape_time.isoformat()}")

            if schedules_created:
                tracker.upload_tracking_file(tracking)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Processed {len(boss_windows)} windows",
                "events_found": len(boss_windows),
                "schedules_created": len(schedules_created)
            })
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
