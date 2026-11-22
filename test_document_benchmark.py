import tempfile
import unittest
from pathlib import Path

from document_benchmark import field_score, load_manifest


class DocumentBenchmarkTests(unittest.TestCase):
    def test_exact_token_not_substring(self):
        result = field_score('Summe 563,43 und 1.248,44', ['563,43', '1.248,44', '63,43'])
        self.assertEqual(result, {'matched': 2, 'total': 3})

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
