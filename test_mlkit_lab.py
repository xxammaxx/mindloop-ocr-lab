import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlkit_lab as m


def observation(variant, path, id, text):
    return {'variant': variant, 'pipeline': path, 'id': id, 'text': text,
            'elapsed_ms': 10, 'line_count': 1, 'status': 'ok'}


class MlKitLabTests(unittest.TestCase):
    def test_catalog_advances_and_does_not_repeat_rejections(self):
        self.assertEqual(m.pending([]), 'contrast1')
        self.assertEqual(m.pending([{'candidate': 'contrast1', 'status': 'REJECTED_DEV'}]), 'sharpen')
        self.assertIsNone(m.pending([{'candidate': key} for key in m.CANDIDATES]))
        self.assertIsNone(m.pending([{'candidate': 'contrast1', 'status': 'INCONCLUSIVE_INFRA'}]))

    def test_gate_detects_per_field_regression_and_is_pipeline_specific(self):
        truth = [{'id': 'photo-1', 'modality': 'photo', 'page': 1, 'fields': ['11,00', '22,00']},
                 {'id': 'scan-1', 'modality': 'scan', 'page': 1, 'fields': ['11,00', '22,00']},
                 {'id': 'dev-0', 'modality': 'dev', 'truth': 'Müller 12,00', 'fields': ['Müller']},
                 {'id': 'validation-0', 'modality': 'validation', 'truth': 'Straße 13,00', 'fields': ['Straße']}]
        rows = []
        for path in m.PIPELINES:
            for variant in ('baseline', 'contrast1'):
                texts = ['11,00', '11,00', 'Müller 12,00', 'Straße 13,00']
                if variant == 'contrast1':
                    texts[0] = '11,00 22,00' if path == 'umliva-nv21' else '22,00'
                rows.extend(observation(variant, path, case['id'], text) for case, text in zip(truth, texts))
        result = m.score(truth, rows, 'contrast1')
        self.assertEqual(result['app_statuses'], {'umliva-nv21': 'LAB_CANDIDATE', 'feldexa-gray-bitmap': 'REJECTED_DEV'})
        self.assertNotIn('Müller', str(result))
        self.assertNotIn('11,00', str(result))
        self.assertEqual(result['status'], 'LAB_CANDIDATE')
        with self.assertRaises(ValueError):
            m.score(truth, rows[:-1], 'contrast1')
        with self.assertRaises(ValueError):
            m.score(truth, rows + [rows[0]], 'contrast1')
        rows[0]['status'] = 'error'
        with self.assertRaises(ValueError):
            m.score(truth, rows, 'contrast1')

    def test_empty_recognition_is_not_a_candidate(self):
        truth = [{'id': 'p', 'modality': 'photo', 'page': 1, 'fields': ['11,00']}]
        rows = [observation(v, p, 'p', '') for v in ('baseline', 'contrast1') for p in m.PIPELINES]
        self.assertTrue(all(x == 'REJECTED_DEV' for x in m.score(truth, rows, 'contrast1')['app_statuses'].values()))

    def test_source_signature_changes_when_input_manifest_changes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'manifest.json').write_text('{}')
            (path / 'SHA256SUMS').write_text('first')
            a = m.signature(path)
            (path / 'SHA256SUMS').write_text('second')
            self.assertNotEqual(a, m.signature(path))


if __name__ == '__main__':
    unittest.main()
