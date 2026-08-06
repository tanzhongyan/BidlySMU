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
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import sys
sys.path.insert(0, "/app")

from src.scraper.trumba_client import TrubaClient, TrubaConfig
# Timezone helpers are centralized in src.config; alias the public
# to_sgt_aware / to_utc_str to the historical private names so the call
# sites below stay unchanged.
from src.config import SGT, _parse_sgt, to_sgt_aware as _to_sgt_aware, to_utc_str as _to_utc_str
from supabase import create_client

# --- Environment configuration ---
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")


# Supabase credentials are injected as plain Lambda env vars (see
# deploy/terraform/lambda.tf — SUPABASE_URL / SUPABASE_SERVICE_KEY map from
# var.supabase_url / var.supabase_service_key).
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
TRUBA_API_URL = os.environ.get("TRUBA_API_URL", "https://www.trumba.com/calendars/SMU_RO_Acad.json")
MONTHS_AHEAD = int(os.environ.get("MONTHS_AHEAD", "12"))

ECS_CLUSTER_ARN = os.environ.get("ECS_CLUSTER_ARN")
ECS_TASK_DEF_ARN = os.environ.get("ECS_TASK_DEF_ARN")
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "bidlysmu-pipeline")
SUBNETS = os.environ.get("SUBNETS", "").split(",") if os.environ.get("SUBNETS") else []
SECURITY_GROUPS = os.environ.get("SECURITY_GROUPS", "").split(",") if os.environ.get("SECURITY_GROUPS") else []
ASSIGN_PUBLIC_IP = os.environ.get("ASSIGN_PUBLIC_IP", "ENABLED")
LAMBDA_INVOKE_ROLE_ARN = os.environ.get("LAMBDA_INVOKE_ROLE_ARN", "")


# =============================================================================
# Timezone helpers
# =============================================================================
# Canonical in-memory schedule datetimes are SGT-aware (tzinfo=SGT, UTC+8).
# Naive values that come from Trumba / bidding_schedules.json are SGT
# wall-clock. EventBridge at() needs UTC, so SGT->UTC conversion happens only
# when formatting the schedule expression.
#
# The helpers themselves are centralized in src.config (SGT, _parse_sgt,
# to_sgt_aware, to_utc_str) and imported above.


# =============================================================================
# Scrape time calculation
# =============================================================================

def calculate_scrape_time(windows: List, window_index: int) -> datetime:
    """Calculate when to scrape based on timing logic.

    - First window: 2 weeks before the window opens (using the actual
      ``opens_at``, entry index 3).
    - Subsequent windows: 1 hour before the window opens (using actual
      opens_at from Trumba, entry index 3), or 1 hour after previous results
      as fallback. This guarantees the ECS pipeline completes before bidding
      starts.

    Returns NAIVE datetimes (SGT wall-clock); callers convert to SGT-aware /
    UTC as needed.
    """
    current_entry = windows[window_index]

    if window_index == 0:
        # First window: 2 weeks before the window actually opens (entry index 3).
        # Entries are always 5-element [results, title, abbrev, opens, closes];
        # a missing opens_at is a data error.
        opens_at = datetime.fromisoformat(current_entry[3])
        return opens_at - timedelta(weeks=2)

    # Subsequent windows: 1 hour before opens_at (entry index 3); fall back to
    # 1 hour after the previous window's results when opens_at is missing.
    try:
        opens_at = datetime.fromisoformat(current_entry[3])
        return opens_at - timedelta(hours=1)
    except (IndexError, TypeError, ValueError):
        pass

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

        Entries use the canonical 5-element format:
            [results_at, title, abbrev, opens_at, closes_at]
        """
        existing_lookup = {}
        for term, windows in existing.items():
            for window in windows:
                abbrev = window[2]
                if abbrev:
                    existing_lookup.setdefault(term, set()).add(abbrev)

        added_count = 0
        for bw in new_windows:
            term = bw.term
            abbrev = bw.abbrev
            # Skip entries without a results timestamp — they can't be
            # scheduled and would break the chronological sort below.
            results_at = getattr(bw, "results_at", None) or getattr(bw, "start_dt", None)
            if not results_at:
                logger.warning(f"Skipping merge of {abbrev or term}: missing results_at")
                continue
            if term not in existing_lookup or abbrev not in existing_lookup.get(term, set()):
                if term not in existing:
                    existing[term] = []
                existing[term].append(bw.to_schedule_entry())
                added_count += 1

        # Keep windows in chronological order — scheduling logic (first-window
        # detection, previous-window lookups) assumes sorted windows.
        # Sort on parsed datetimes (SGT wall-clock). Entries with missing or
        # unparseable results_at fall back to datetime.min so a null value
        # can't crash the sort with a TypeError.
        def _results_sort_key(entry):
            if not entry or not entry[0]:
                return datetime.min
            try:
                return _parse_sgt(entry[0]).replace(tzinfo=None)
            except (TypeError, ValueError):
                return datetime.min

        for term in existing:
            existing[term].sort(key=_results_sort_key)

        # Canonical titles carry no "BOSS " prefix and no " Results" suffix;
        # normalize on every merge so the stored file stays consistent.
        for term in existing:
            for entry in existing[term]:
                if entry[1]:
                    entry[1] = entry[1].replace(" Results", "").replace("BOSS ", "", 1).strip()

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

    def remove_from_tracking(self, tracking: Dict, term: str, schedule_name: str) -> Dict:
        """Remove a schedule from the tracking dict (used when recreating)."""
        if term in tracking and schedule_name in tracking.get(term, {}):
            del tracking[term][schedule_name]
        return tracking


# =============================================================================
# EventBridge Scheduler — wraps boto3 EventBridge Scheduler client
# =============================================================================

class EventBridgeScheduler:
    """Wraps the boto3 EventBridge Scheduler client for creating one-time schedules.

    Each schedule invokes this Lambda in Mode 2 (run_pipeline) at the computed
    scrape time for a specific term/window pair.
    """

    def __init__(self, region, cluster_arn, task_def_arn,
                 scheduler_role_arn, subnets, security_groups,
                 lambda_arn=None):
        self._region = region
        self._cluster_arn = cluster_arn
        self._task_def_arn = task_def_arn
        self._scheduler_role_arn = scheduler_role_arn
        self._subnets = subnets
        self._security_groups = security_groups
        self._lambda_arn = lambda_arn
        self._scheduler = boto3.client("scheduler", region_name=region)

    def schedule_exists(self, name):
        """Check whether an EventBridge schedule already exists."""
        try:
            self._scheduler.get_schedule(Name=name)
            return True
        except self._scheduler.exceptions.ResourceNotFoundException:
            return False

    def get_schedule_payload(self, name):
        """Return the input payload of an existing schedule, or None."""
        try:
            resp = self._scheduler.get_schedule(Name=name)
            return json.loads(resp["Target"]["Input"])
        except (self._scheduler.exceptions.ResourceNotFoundException, KeyError, json.JSONDecodeError):
            return None

    def delete_schedule(self, name):
        """Delete an EventBridge schedule and its retry schedules by name.
        No-op if they don't exist.
        """
        for schedule_name in [name, f"{name}-retry1", f"{name}-retry2"]:
            try:
                self._scheduler.delete_schedule(Name=schedule_name)
            except self._scheduler.exceptions.ResourceNotFoundException:
                pass

    def create_schedule(self, schedule_name, scrape_time, term, abbrev,
                        results_datetime, previous_window=None):
        """Create a one-time EventBridge schedule that triggers a pipeline run.

        Also creates 2 retry schedules (+2h and +4h) that check for success
        before re-running.
        """
        base_input = {
            "trigger": "run_pipeline",
            "acad_term_id": term,
            "window": abbrev,
            "results_datetime": results_datetime,
            "previous_window": previous_window,
        }

        # Primary schedule
        self._scheduler.create_schedule(
            Name=schedule_name,
            # All times from bidding_schedules.json / Trumba are SGT wall-clock.
            # EventBridge at() requires UTC; _to_utc_str() converts SGT -> UTC.
            ScheduleExpression=f"at({_to_utc_str(scrape_time)})",
            FlexibleTimeWindow={"Mode": "OFF"},
            Target={
                "Arn": self._lambda_arn or "",
                "RoleArn": self._scheduler_role_arn,
                "Input": json.dumps({**base_input, "retry_attempt": 0}),
            },
            ActionAfterCompletion="DELETE",
        )

        # Retry schedules (+2h and +4h)
        # NOTE: timedelta is imported at module level (from datetime import
        # datetime, timedelta). A local import here would shadow it and make
        # it function-local, causing UnboundLocalError at the primary
        # schedule line above which runs before this statement executes.
        for retry_idx, hours_offset in enumerate([2, 4], start=1):
            retry_time = scrape_time + timedelta(hours=hours_offset)
            retry_name = f"{schedule_name}-retry{retry_idx}"
            self._scheduler.create_schedule(
                Name=retry_name,
                ScheduleExpression=f"at({_to_utc_str(retry_time)})",
                FlexibleTimeWindow={"Mode": "OFF"},
                Target={
                    "Arn": self._lambda_arn or "",
                    "RoleArn": self._scheduler_role_arn,
                    "Input": json.dumps({**base_input, "retry_attempt": retry_idx}),
                },
                ActionAfterCompletion="DELETE",
            )


# =============================================================================
# Dual-mode Lambda handler
# =============================================================================

def _pipeline_already_succeeded(acad_term_id: str, window: str) -> bool:
    """Check if the pipeline already produced output for this term/window."""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        supabase.storage.from_("bidlysmu-files").download(
            f"output/{acad_term_id}/{window}/_SUCCESS"
        )
        return True
    except Exception:
        return False

def _pipeline_in_progress(acad_term_id: str, window: str) -> bool:
    """Check if a pipeline run is currently in progress for this term/window.

    The pipeline writes a `_STARTED` marker at the start of a run and only
    writes `_SUCCESS` once every step has completed.  A `_STARTED` marker
    without a `_SUCCESS` marker therefore means the run has started but not
    finished — retries must not spawn a concurrent ECS task.
    """
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        supabase.storage.from_("bidlysmu-files").download(
            f"output/{acad_term_id}/{window}/_STARTED"
        )
        return True
    except Exception:
        return False

def lambda_handler(event, context):
    """Dual-mode handler: monthly calendar fetch OR pipeline run trigger."""
    try:
        trigger = event.get("trigger", "monthly-schedule")

        # =============================================================
        # MODE 2: Pipeline run — start ECS task with overrides
        # =============================================================
        if trigger == "run_pipeline":
            acad_term_id = event.get("acad_term_id")
            window = event.get("window")  # may be None for cleanup runs
            previous_window = event.get("previous_window")
            results_datetime = event.get("results_datetime", "")

            if not acad_term_id:
                raise ValueError("run_pipeline trigger requires acad_term_id")

            retry_attempt = event.get("retry_attempt", 0)

            # If this is a retry, skip when the run already succeeded (_SUCCESS)
            # OR is still in progress (_STARTED without _SUCCESS).  This keeps
            # +2h/+4h retries from spawning concurrent ECS tasks that would
            # duplicate DB writes while the primary run is still going.
            check_window = window or previous_window
            if retry_attempt > 0 and check_window:
                if _pipeline_already_succeeded(acad_term_id, check_window):
                    logger.info(
                        f"Pipeline already succeeded for {acad_term_id}/{check_window}, "
                        f"skipping retry attempt {retry_attempt}."
                    )
                    return {
                        "statusCode": 200,
                        "body": json.dumps({
                            "message": f"Pipeline already succeeded for {acad_term_id}/{check_window}, skipped retry",
                            "skipped": True,
                        })
                    }
                if _pipeline_in_progress(acad_term_id, check_window):
                    logger.info(
                        f"Pipeline still in progress for {acad_term_id}/{check_window}, "
                        f"skipping retry attempt {retry_attempt}."
                    )
                    return {
                        "statusCode": 200,
                        "body": json.dumps({
                            "message": f"Pipeline still in progress for {acad_term_id}/{check_window}, skipped retry",
                            "skipped": True,
                        })
                    }

            logger.info(
                f"Mode 2: Starting ECS task for {acad_term_id}/"
                f"current={window or '(none)'} prev={previous_window or '(none)'} "
                f"(attempt {retry_attempt + 1})"
            )

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
                            # Canonical abbrev is the single source of truth for the window identity.
                            # ECS/config derive the full name from it; never round-trip abbrev->full->abbrev.
                            {"name": "TARGET_CURRENT_WINDOW", "value": window or ""},
                            {"name": "TARGET_PREVIOUS_WINDOW", "value": previous_window or ""},
                            {"name": "RESULTS_DATETIME", "value": results_datetime},
                            {"name": "RETRY_ATTEMPT", "value": str(retry_attempt)}
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
            sched = EventBridgeScheduler(
                region=AWS_REGION,
                cluster_arn=ECS_CLUSTER_ARN,
                task_def_arn=ECS_TASK_DEF_ARN,
                scheduler_role_arn=LAMBDA_INVOKE_ROLE_ARN,
                subnets=SUBNETS,
                security_groups=SECURITY_GROUPS,
                lambda_arn=lambda_arn,
            )
            tracker = ScheduleTracker(supabase)
            tracking = tracker.download_tracking_file()

            for term, windows in updated.items():
                for i, window in enumerate(windows):
                    try:
                        abbrev = window[2]
                        previous_abbrev = windows[i-1][2] if i > 0 else None
                        name = f"bidlysmu-pipeline-{term}-{abbrev}".replace(" ", "-").replace("/", "-")

                        if tracker.is_tracked(tracking, term, name):
                            continue
                        if sched.schedule_exists(name):
                            continue

                        scrape_time = calculate_scrape_time(windows, i)
                        # Canonical in-memory datetimes are SGT-aware;
                        # calculate_scrape_time returns naive SGT wall-clock.
                        scrape_time = _to_sgt_aware(scrape_time)
                        if scrape_time < datetime.now(SGT):
                            continue

                        sched.create_schedule(
                            schedule_name=name,
                            scrape_time=scrape_time,
                            term=term,
                            abbrev=abbrev,
                            results_datetime=window[0],
                            previous_window=previous_abbrev,
                        )
                        # Store the SGT wall-clock form (naive) to keep the
                        # tracking file format unchanged.
                        tracking = tracker.add_to_tracking(
                            tracking, term, name,
                            scrape_time.astimezone(SGT).replace(tzinfo=None),
                            window[0],
                        )
                        schedules_created.append(name)
                        logger.info(f"Schedule: {name} at {scrape_time.isoformat()}")
                    except Exception as e:
                        # One bad window must not abort the rest of the run.
                        logger.error(
                            f"Failed to create schedule for {term}/{window[2] if len(window) > 2 else '?'}: {e}",
                            exc_info=True,
                        )
                        continue

                # Schedule a final cleanup run after the last window's results
                # are released.  If new windows have been added since the last
                # monthly scrape, delete the old cleanup and recreate it pointing
                # at the new last window.
                if windows:
                    try:
                        last = windows[-1]
                        last_abbrev = last[2]
                        last_results_at = _to_sgt_aware(datetime.fromisoformat(last[0]))
                        cleanup_time = last_results_at + timedelta(hours=1)

                        cleanup_name = f"bidlysmu-pipeline-{term}-CLEANUP".replace(" ", "-").replace("/", "-")

                        needs_cleanup = False
                        if sched.schedule_exists(cleanup_name):
                            existing_payload = sched.get_schedule_payload(cleanup_name)
                            existing_prev = existing_payload.get('previous_window') if existing_payload else None
                            if existing_prev != last_abbrev:
                                logger.info(
                                    f"Cleanup target changed: {existing_prev} -> {last_abbrev}, "
                                    f"deleting old schedule and recreating"
                                )
                                sched.delete_schedule(cleanup_name)
                                tracker.remove_from_tracking(tracking, term, cleanup_name)
                                needs_cleanup = True
                        elif not tracker.is_tracked(tracking, term, cleanup_name):
                            needs_cleanup = True

                        if needs_cleanup and cleanup_time > datetime.now(SGT):
                            sched.create_schedule(
                                schedule_name=cleanup_name,
                                scrape_time=cleanup_time,
                                term=term,
                                abbrev=None,
                                results_datetime=last[0],
                                previous_window=last_abbrev,
                            )
                            tracking = tracker.add_to_tracking(
                                tracking, term, cleanup_name,
                                cleanup_time.astimezone(SGT).replace(tzinfo=None),
                                last[0],
                            )
                            schedules_created.append(cleanup_name)
                            logger.info(f"Schedule (cleanup): {cleanup_name} at {cleanup_time.isoformat()}")
                    except Exception as e:
                        # Cleanup scheduling failure must not abort the run.
                        logger.error(f"Failed to create cleanup schedule for {term}: {e}", exc_info=True)

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
