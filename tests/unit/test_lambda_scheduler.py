"""Tests for lambda/monthly_scheduler/lambda_function.py (Mode 2 handoff + schedule payload)."""
import importlib
import json
import pytest
from datetime import datetime

MODULE = importlib.import_module('lambda.monthly_scheduler.lambda_function')


class TestMode2EcsHandoff:
    """Mode 2 must pass the canonical abbrev window + AY term through unchanged."""

    def test_env_overrides_use_canonical_abbrev(self, monkeypatch):
        captured = {}

        class FakeECS:
            def run_task(self, **kw):
                captured.update(kw)
                return {'tasks': [{'taskArn': 'arn:aws:ecs:ap-southeast-1:1:task/x'}]}

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeECS())
        monkeypatch.setattr(MODULE, 'ECS_CLUSTER_ARN', 'cluster')
        monkeypatch.setattr(MODULE, 'ECS_TASK_DEF_ARN', 'td')
        monkeypatch.setattr(MODULE, 'SUBNETS', [])
        monkeypatch.setattr(MODULE, 'SECURITY_GROUPS', [])
        monkeypatch.setattr(MODULE, 'ASSIGN_PUBLIC_IP', 'ENABLED')

        MODULE.lambda_handler({
            'trigger': 'run_pipeline',
            'acad_term_id': 'AY202627T1',
            'window': 'R1FW4',
            'previous_window': 'R1FW3',
            'retry_attempt': 0,
        }, None)

        env = {e['name']: e['value']
               for e in captured['overrides']['containerOverrides'][0]['environment']}
        assert env['TARGET_CURRENT_WINDOW'] == 'R1FW4'
        assert env['TARGET_PREVIOUS_WINDOW'] == 'R1FW3'
        assert env['ACAD_TERM_ID'] == 'AY202627T1'


class TestCreateSchedulePayload:
    """Mode 1 payload uses the AY term unchanged (term is already AY)."""

    def test_payload_acad_term_id_is_ay(self, monkeypatch):
        captured = {}

        class FakeScheduler:
            def create_schedule(self, **kw):
                captured['name'] = kw['Name']
                captured['input'] = json.loads(kw['Target']['Input'])

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeScheduler())
        sched = MODULE.EventBridgeScheduler(
            region='ap-southeast-1', cluster_arn='c', task_def_arn='t',
            scheduler_role_arn='r', subnets=[], security_groups=[], lambda_arn='arn:lambda',
        )
        sched.create_schedule(
            schedule_name='bidlysmu-pipeline-AY202627T1-R1FW4',
            scrape_time=datetime(2026, 7, 8, 10, 0),
            term='AY202627T1', abbrev='R1FW4',
            results_datetime='2026-07-04T14:00:00', previous_window='R1FW3',
        )
        assert captured['input']['acad_term_id'] == 'AY202627T1'
        assert captured['input']['window'] == 'R1FW4'


class TestCalculateScrapeTime:
    """First window uses the actual opens_at (entry index 3); no legacy branch."""

    def test_first_window_uses_opens_at(self):
        windows = [[
            '2026-07-04T14:00:00', 'Round 1 Window 1', 'R1W1',
            '2026-07-02T10:00:00', '2026-07-04T10:00:00',
        ]]
        result = MODULE.calculate_scrape_time(windows, 0)
        # opens_at 2026-07-02 10:00 minus 2 weeks = 2026-06-18 10:00
        assert result == datetime(2026, 6, 18, 10, 0)

    def test_missing_opens_at_raises(self):
        windows = [['2026-07-04T14:00:00', 'Round 1 Window 1', 'R1W1']]
        with pytest.raises(IndexError):
            MODULE.calculate_scrape_time(windows, 0)


class TestMode2DedupAllAttempts:
    """Mode 2 must dedup EVERY attempt (primary included) against markers."""

    def test_attempt1_skips_when_success_marker_exists(self, monkeypatch):
        class FakeECS:
            def run_task(self, **kw):
                raise AssertionError("run_task must not be called when _SUCCESS exists")

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeECS())
        monkeypatch.setattr(MODULE, '_pipeline_already_succeeded', lambda t, w: True)
        monkeypatch.setattr(MODULE, '_pipeline_in_progress', lambda t, w: False)

        resp = MODULE.lambda_handler({
            'trigger': 'run_pipeline', 'acad_term_id': 'AY202627T1',
            'window': 'R1FW1', 'previous_window': None, 'retry_attempt': 0,
        }, None)

        body = json.loads(resp['body'])
        assert resp['statusCode'] == 200
        assert body['skipped'] is True
        assert 'succeeded' in body['message']

    def test_attempt1_skips_when_in_progress(self, monkeypatch):
        class FakeECS:
            def run_task(self, **kw):
                raise AssertionError("run_task must not be called when _STARTED exists")

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeECS())
        monkeypatch.setattr(MODULE, '_pipeline_already_succeeded', lambda t, w: False)
        monkeypatch.setattr(MODULE, '_pipeline_in_progress', lambda t, w: True)

        resp = MODULE.lambda_handler({
            'trigger': 'run_pipeline', 'acad_term_id': 'AY202627T1',
            'window': 'R1FW1', 'previous_window': None, 'retry_attempt': 0,
        }, None)

        body = json.loads(resp['body'])
        assert body['skipped'] is True
        assert 'in progress' in body['message']

    def test_attempt1_starts_task_when_no_markers(self, monkeypatch):
        captured = {}

        class FakeECS:
            def run_task(self, **kw):
                captured['called'] = True
                return {'tasks': [{'taskArn': 'arn:aws:ecs:ap-southeast-1:1:task/x'}]}

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeECS())
        monkeypatch.setattr(MODULE, '_pipeline_already_succeeded', lambda t, w: False)
        monkeypatch.setattr(MODULE, '_pipeline_in_progress', lambda t, w: False)
        monkeypatch.setattr(MODULE, 'ECS_CLUSTER_ARN', 'cluster')
        monkeypatch.setattr(MODULE, 'ECS_TASK_DEF_ARN', 'td')
        monkeypatch.setattr(MODULE, 'SUBNETS', [])
        monkeypatch.setattr(MODULE, 'SECURITY_GROUPS', [])
        monkeypatch.setattr(MODULE, 'ASSIGN_PUBLIC_IP', 'ENABLED')

        resp = MODULE.lambda_handler({
            'trigger': 'run_pipeline', 'acad_term_id': 'AY202627T1',
            'window': 'R1FW1', 'retry_attempt': 0,
        }, None)

        assert captured.get('called') is True
        assert json.loads(resp['body'])['message'].startswith('ECS task started')

    def test_cleanup_primary_not_guarded_by_last_window_markers(self, monkeypatch):
        # Cleanup (window=None) runs after the last window on purpose — its
        # primary must NOT be suppressed even if the last window succeeded or
        # is still marked in progress.
        called = {'marker_checked': False, 'task_started': False}

        def fake_succeeded(t, w):
            called['marker_checked'] = True
            return True

        class FakeECS:
            def run_task(self, **kw):
                called['task_started'] = True
                return {'tasks': [{'taskArn': 'arn:aws:ecs:ap-southeast-1:1:task/x'}]}

        monkeypatch.setattr(MODULE.boto3, 'client', lambda *a, **k: FakeECS())
        monkeypatch.setattr(MODULE, '_pipeline_already_succeeded', fake_succeeded)
        monkeypatch.setattr(MODULE, '_pipeline_in_progress', lambda t, w: True)
        monkeypatch.setattr(MODULE, 'ECS_CLUSTER_ARN', 'cluster')
        monkeypatch.setattr(MODULE, 'ECS_TASK_DEF_ARN', 'td')
        monkeypatch.setattr(MODULE, 'SUBNETS', [])
        monkeypatch.setattr(MODULE, 'SECURITY_GROUPS', [])
        monkeypatch.setattr(MODULE, 'ASSIGN_PUBLIC_IP', 'ENABLED')

        MODULE.lambda_handler({
            'trigger': 'run_pipeline', 'acad_term_id': 'AY202627T1',
            'window': None, 'previous_window': 'R2AW3', 'retry_attempt': 0,
        }, None)

        assert called['marker_checked'] is False
        assert called['task_started'] is True


class TestMode1SkipsCompletedWindows:
    """Mode 1 must not re-create schedules for windows whose pipeline already succeeded."""

    def test_completed_window_not_rescheduled(self, monkeypatch):
        created = []

        class FakeTruba:
            def fetch_boss_windows(self):
                return [object()]

        class FakeMgr:
            def __init__(self, *a, **k):
                pass

            def download_existing_schedules(self):
                return {}

            def merge_with_deduplication(self, existing, new):
                return {
                    'AY202627T1': [[
                        '2026-08-11T14:00:00', 'Incoming Freshmen Round 1 Window 1',
                        'R1FW1', '2026-08-14T10:00:00', '2026-08-16T10:00:00',
                    ]]
                }

            def upload_schedules(self, s):
                pass

        class FakeTracker:
            def __init__(self, *a, **k):
                pass

            def download_tracking_file(self):
                return {}

            def is_tracked(self, *a):
                return False

            def add_to_tracking(self, *a, **k):
                return {}

            def upload_tracking_file(self, t):
                pass

        class FakeScheduler:
            def __init__(self, *a, **k):
                pass

            def schedule_exists(self, name):
                return False

            def get_schedule_payload(self, name):
                return None

            def delete_schedule(self, name):
                pass

            def create_schedule(self, **kw):
                # EventBridgeScheduler is mocked, so the handler's call passes
                # schedule_name (the real class's own parameter name).
                created.append(kw['schedule_name'])

        monkeypatch.setattr(MODULE, 'SUPABASE_URL', 'https://x.supabase.co')
        monkeypatch.setattr(MODULE, 'SUPABASE_SERVICE_KEY', 'svc')
        monkeypatch.setattr(MODULE, 'LAMBDA_INVOKE_ROLE_ARN', 'role')
        monkeypatch.setattr(MODULE, 'create_client', lambda *a, **k: object())
        monkeypatch.setattr(MODULE, 'TrubaClient', lambda *a, **k: FakeTruba())
        monkeypatch.setattr(MODULE, 'TrubaConfig', lambda *a, **k: None)
        monkeypatch.setattr(MODULE, 'BiddingScheduleManager', FakeMgr)
        monkeypatch.setattr(MODULE, 'ScheduleTracker', FakeTracker)
        monkeypatch.setattr(MODULE, 'EventBridgeScheduler', FakeScheduler)
        # R1FW1 already completed -> must not be re-scheduled.
        monkeypatch.setattr(MODULE, '_pipeline_already_succeeded', lambda term, w: w == 'R1FW1')

        ctx = type('Ctx', (), {'invoked_function_arn': 'arn:aws:lambda:ap-southeast-1:1:function:x'})()
        resp = MODULE.lambda_handler({'trigger': 'monthly-schedule'}, ctx)

        assert 'bidlysmu-pipeline-AY202627T1-R1FW1' not in created
        # The term's cleanup run is still scheduled (it is not a window run).
        assert 'bidlysmu-pipeline-AY202627T1-CLEANUP' in created
        assert json.loads(resp['body'])['schedules_created'] == 1
