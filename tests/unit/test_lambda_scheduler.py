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
