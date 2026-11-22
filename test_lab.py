import tempfile
import unittest
from pathlib import Path

import lab


class LabTests(unittest.TestCase):
    def test_character_error_rate_counts_umlauts(self):
        self.assertEqual(lab.edit_distance('Müller', 'Muller'), 1)
        self.assertEqual(lab.edit_distance('12,50', '1250'), 1)

    def test_critical_token_regression_blocks_promotion(self):
        baseline = {'cer': 0.15, 'critical_accuracy': 1.0}
        self.assertFalse(lab.improves(baseline, {'cer': 0.05, 'critical_accuracy': 0.9}))
        self.assertTrue(lab.improves(baseline, {'cer': 0.05, 'critical_accuracy': 1.0}))

    def test_plan_skips_completed_candidates_and_is_bounded(self):
        ids = [c['id'] for c in lab.CANDIDATES]
        plan = lab.plan([{'candidate': ids[0]}], 2)
        self.assertEqual(len(plan), 2)
        self.assertNotIn(ids[0], [c['id'] for c in plan])
        self.assertEqual(lab.plan([{'candidate': x} for x in ids], 2), [])
        with self.assertRaises(ValueError):
            lab.plan([], 100)

    def test_fixtures_are_deterministic_and_separate(self):
        dev = lab.fixtures('dev')
        validation = lab.fixtures('validation')
        self.assertTrue(set(x['id'] for x in dev).isdisjoint(x['id'] for x in validation))
        self.assertEqual(dev[0]['image'].tobytes(), lab.fixtures('dev')[0]['image'].tobytes())
        self.assertIn('ü', dev[0]['truth'])

    def test_empty_or_different_context_does_not_reuse_ledger(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base)/'ledger.json'
            self.assertEqual(lab.load_ledger(path, 'new')['records'], [])
            lab.save_ledger(path, {'schema': 1, 'context': 'old', 'records': [{'candidate': 'x'}]})
            with self.assertRaises(ValueError):
                lab.load_ledger(path, 'new')


if __name__ == '__main__':
    unittest.main()
