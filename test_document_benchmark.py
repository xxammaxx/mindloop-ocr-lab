import tempfile
import unittest
from pathlib import Path

import document_benchmark as benchmark
from document_benchmark import field_score, load_manifest


class DocumentBenchmarkTests(unittest.TestCase):
    def test_exact_token_not_substring(self):
        result = field_score('Summe 563,43 und 1.248,44', ['563,43', '1.248,44', '63,43'])
        self.assertEqual(result, {'matched': 2, 'total': 3})

    def test_punctuation_is_not_part_of_a_field_but_numeric_separators_are(self):
        self.assertEqual(field_score('Summe 563,43. Name Müller, Ende 1.563,43', ['563,43', 'Müller', '43', '63,43']), {'matched': 2, 'total': 4})
        self.assertEqual(field_score('1.563,43 und 563,431', ['563,43']), {'matched': 0, 'total': 1})

    def test_scan_inventory_requires_one_raster_per_physical_page(self):
        self.assertTrue(benchmark.valid_scan_inventory('header\n---\n1 0 image\n2 1 image\n', 2))
        self.assertFalse(benchmark.valid_scan_inventory('header\n---\n1 0 image\n1 1 image\n', 2))
        self.assertFalse(benchmark.valid_scan_inventory('header\n---\n1 0 image\n2 1 mask\n', 2))

    def test_hit_indices_distinguish_lost_and_new_fields(self):
        self.assertTrue(hasattr(benchmark, 'field_hits'), 'missing per-field recognition evidence')
        self.assertEqual(benchmark.field_hits('Summe 563,43 und 1.248,44', ['563,43', '63,43', '1.248,44']), [0, 2])

    def test_paired_gate_rejects_lost_field_even_when_count_increases(self):
        def row(modality, hits):
            return {'document': 'one', 'page': 1, 'modality': modality, 'total': 4, 'matched': len(hits), 'hits': hits}
        baseline = [row('scan', [0, 1]), row('photo', [0])]
        self.assertTrue(hasattr(benchmark, 'paired_improves'), 'missing paired regression gate')
        self.assertFalse(benchmark.paired_improves(baseline, [row('scan', [0, 2, 3]), row('photo', [0, 1])]))
        self.assertFalse(benchmark.paired_improves(baseline, [row('scan', [0, 1, 2])]))
        self.assertFalse(benchmark.paired_improves(baseline, baseline))
        self.assertTrue(benchmark.paired_improves(baseline, [row('scan', [0, 1, 2]), row('photo', [0])]))

    def test_same_document_modalities_stay_together(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text('{"schema":1,"documents":[{"id":"x","scan":"scan.pdf","pages":[{"photo":"p.png","fields":["test"]}]}]}')
            (root / 'scan.pdf').touch()
            (root / 'p.png').touch()
            doc = load_manifest(root)[0]
            self.assertEqual(doc['id'], 'x')
            self.assertEqual(len(doc['pages']), 1)

    def test_no_external_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text('{"schema":1,"documents":[{"id":"x","scan":"../outside.pdf","pages":[{"photo":"p.png","fields":["test"]}]}]}')
            with self.assertRaises(ValueError):
                load_manifest(root)
