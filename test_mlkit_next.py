import json
import importlib.util
import subprocess
import unittest


class ContinueResearchTests(unittest.TestCase):
    def check_dispatch(self, records, statuses, expected):
        self.assertIsNotNone(importlib.util.find_spec('mlkit_next'), 'Continuation controller missing')
        from mlkit_next import dispatch_next
        calls = []

        def gh(args, **kwargs):
            calls.append(args)
            if args[1:3] == ['run', 'list']:
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([{'status': s} for s in statuses]))
            if args[1:3] == ['workflow', 'run']:
                return subprocess.CompletedProcess(args, 0)
            self.fail(f'Unexpected command: {args}')

        self.assertEqual(dispatch_next(records, gh), expected)
        dispatches = [c for c in calls if c[1:3] == ['workflow', 'run']]
        self.assertEqual(dispatches, [['gh', 'workflow', 'run', 'mlkit.yml', '--ref', 'main', '--repo', 'xxammaxx/mindloop-ocr-lab']] if expected else [])

    def test_rejected_result_starts_next_candidate(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'REJECTED_DEV'}], ['completed'], 'sharpen')

    def test_candidate_result_still_continues(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'LAB_CANDIDATE'}], [], 'sharpen')

    def test_infrastructure_failure_stops(self):
        self.check_dispatch([{'candidate': 'contrast1', 'status': 'INCONCLUSIVE_INFRA'}], [], None)

    def test_exhausted_catalog_stops(self):
        self.check_dispatch([{'candidate': c, 'status': 'REJECTED_DEV'} for c in ('contrast1', 'sharpen', 'contrast', 'threshold', 'scale2')], [], None)

    def test_active_research_prevents_duplicate(self):
        for status in ('queued', 'in_progress', 'waiting', 'requested', 'pending'):
            with self.subTest(status=status):
                self.check_dispatch([], ['completed', status], None)
