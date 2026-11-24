import json
import importlib.util
import os
import subprocess
import unittest
from unittest.mock import patch


class ContinueResearchTests(unittest.TestCase):
    def check_dispatch(self, records, statuses, expected, current_run_id=None):
        self.assertIsNotNone(importlib.util.find_spec('mlkit_next'), 'Continuation controller missing')
        from mlkit_next import dispatch_next
        calls = []

        def gh(args, **kwargs):
            calls.append(args)
            if args[1:3] == ['run', 'list']:
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([{'databaseId': i + 100, 'status': s} for i, s in enumerate(statuses)]))
            if args[1:3] == ['workflow', 'run']:
                return subprocess.CompletedProcess(args, 0)
            self.fail(f'Unexpected command: {args}')

        with patch.dict(os.environ, {'GITHUB_RUN_ID': str(current_run_id or '')}):
            self.assertEqual(dispatch_next(records, gh), expected)
        dispatches = [c for c in calls if c[1:3] == ['workflow', 'run']]
        self.assertEqual(dispatches, [['gh', 'workflow', 'run', 'mlkit.yml', '--ref', 'main', '--repo', 'xxammaxx/mindloop-ocr-lab']] if expected else [])

    def test_rejected_result_starts_next_candidate(self):
        self.check_dispatch([], ['completed'], 'region-crop-retry')

    def test_candidate_result_still_continues(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'LAB_CANDIDATE'}], [], 'region-crop-retry')

    def test_infrastructure_failure_stops(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'INCONCLUSIVE_INFRA'}], [], None)

    def test_exhausted_catalog_stops(self):
        self.check_dispatch([{'candidate': 'region-crop-retry', 'status': 'REJECTED_DEV'}], [], None)

    def test_active_research_prevents_duplicate(self):
        for status in ('queued', 'in_progress', 'waiting', 'requested', 'pending'):
            with self.subTest(status=status):
                self.check_dispatch([], ['completed', status], None)

    def test_finishing_workflow_does_not_block_its_own_successor(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'REJECTED_DEV'},
                             {'candidate': 'sharpen', 'status': 'REJECTED_DEV'}],
                            ['completed', 'in_progress'], 'region-crop-retry', current_run_id=101)

    def test_other_queued_run_still_blocks_successor(self):
        self.check_dispatch([], ['in_progress', 'queued'], None, current_run_id=100)
