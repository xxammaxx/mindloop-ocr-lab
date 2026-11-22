import tempfile
import unittest
from unittest.mock import patch
import subprocess
from pathlib import Path

import lab
from PIL import Image


class LabTests(unittest.TestCase):
    def test_character_error_rate_counts_umlauts(self):
        self.assertEqual(lab.edit_distance('Müller', 'Muller'), 1)
        self.assertEqual(lab.edit_distance('12,50', '1250'), 1)

    def test_critical_token_regression_blocks_promotion(self):
        baseline = {'cer': 0.15, 'critical_accuracy': 1.0}
        self.assertFalse(lab.improves(baseline, {'cer': 0.05, 'critical_accuracy': 0.9}))
        self.assertTrue(lab.improves(baseline, {'cer': 0.05, 'critical_accuracy': 1.0}))

    def test_real_improvement_can_qualify_at_perfect_synthetic_baseline(self):
        perfect = {'cer': 0.0, 'critical_accuracy': 1.0,
                   'cohorts': {'form': {'cer': 0.0, 'critical_accuracy': 1.0}}}
        self.assertTrue(hasattr(lab, 'eligible_dev'))
        self.assertTrue(lab.eligible_dev(perfect, perfect, paired_gain=True))
        self.assertFalse(lab.eligible_dev(perfect, perfect, paired_gain=False))
        self.assertFalse(lab.eligible_dev(perfect, perfect))
        worse = {'cer': 0.1, 'critical_accuracy': 1.0,
                 'cohorts': {'form': {'cer': 0.1, 'critical_accuracy': 1.0}}}
        self.assertFalse(lab.eligible_dev(perfect, worse, paired_gain=True))

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

    def test_german_unicode_probe_present_and_renderable_in_each_split(self):
        expected = set('ÄÖÜäöüßẞ€²³§')
        for split in ('dev', 'validation'):
            truth = ''.join(case['truth'] for case in lab.fixtures(split))
            self.assertTrue(expected <= set(truth), expected - set(truth))
        font = lab.ImageFont.truetype(str(lab.FONT), 29)
        missing = font.getmask('\U0010ffff').tobytes() if hasattr(font.getmask('\U0010ffff'), 'tobytes') else bytes(font.getmask('\U0010ffff'))
        for glyph in expected:
            self.assertNotEqual(bytes(font.getmask(glyph)), missing, glyph)

    def test_one_percent_autocontrast_excludes_sparse_outliers(self):
        image = Image.new('L', (1000, 1))
        image.putdata([0] + [40] * 499 + [190] * 499 + [255])
        self.assertIn('contrast1-psm6', [c['id'] for c in lab.CANDIDATES])
        output = lab.prepare(image, 'contrast1')
        self.assertEqual(output.size, image.size)
        self.assertEqual((output.getpixel((500, 0)), output.getpixel((10, 0))), (255, 0))

    def test_empty_or_different_context_does_not_reuse_ledger(self):
        with tempfile.TemporaryDirectory() as base:
            path = Path(base)/'ledger.json'
            self.assertEqual(lab.load_ledger(path, 'new')['records'], [])
            lab.save_ledger(path, {'schema': 1, 'context': 'old', 'records': [{'candidate': 'x'}]})
            with self.assertRaises(ValueError):
                lab.load_ledger(path, 'new')

    def test_failed_candidate_is_recorded_and_next_candidate_runs(self):
        baseline = {'cer': 0.1, 'critical_accuracy': 1.0, 'cohorts': {'form': {'cer': 0.1, 'critical_accuracy': 1.0}}}
        improved = {'cer': 0.0, 'critical_accuracy': 1.0, 'cohorts': {'form': {'cer': 0.0, 'critical_accuracy': 1.0}}}
        failure = subprocess.TimeoutExpired('private-path', 25)
        with tempfile.TemporaryDirectory() as folder, patch.object(lab, 'context', return_value=('ctx', {})), patch.object(lab, 'fixtures', return_value=[]), patch.object(lab, 'evaluate', side_effect=[baseline, baseline, failure, improved, improved]):
            lab.run(Path(folder), 'eng', 2)
            records = lab.load_ledger(Path(folder)/'ctx'/'ledger.json', 'ctx')['records']
            self.assertEqual([r['status'] for r in records], ['INCONCLUSIVE_TIMEOUT', 'LAB_CANDIDATE'])
            self.assertEqual(records[0]['stage'], 'dev')
            self.assertNotIn('private-path', str(records))

    def test_paired_and_validation_failures_have_stage_receipts(self):
        score = {'cer': 0.0, 'critical_accuracy': 1.0, 'cohorts': {'form': {'cer': 0.0, 'critical_accuracy': 1.0}}}
        fail = subprocess.CalledProcessError(1, 'private-path')
        with tempfile.TemporaryDirectory() as folder, patch.object(lab, 'context', return_value=('ctx', {})), patch.object(lab, 'fixtures', return_value=[]), patch.object(lab, 'evaluate', side_effect=[score, score, score, score, fail]), patch('document_benchmark.evaluate_corpus', side_effect=[[], fail, []]), patch('document_benchmark.paired_improves', return_value=True):
            lab.run(Path(folder), 'eng', 2, corpus=Path(folder))
            records = lab.load_ledger(Path(folder)/'ctx'/'ledger.json', 'ctx')['records']
            self.assertEqual([r['stage'] for r in records], ['paired', 'validation'])
            self.assertEqual([r['status'] for r in records], ['INCONCLUSIVE_ERROR', 'INCONCLUSIVE_ERROR'])
            self.assertNotIn('private-path', str(records))


if __name__ == '__main__':
    unittest.main()
