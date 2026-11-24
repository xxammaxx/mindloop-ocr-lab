import unittest
from pathlib import Path

from mlkit_fusion import fuse, retry_regions, restore_line, should_retry
from feldexa_forms import form_cases
import mlkit_lab as lab


def line(text, bounds):
    return {'text': text, 'box': bounds}


class FusionTests(unittest.TestCase):
    def test_only_spatially_new_lines_are_added_and_conflicts_kept(self):
        first = {'text': 'Name\nBetrag 13,00', 'lines': [line('Name', [10, 10, 100, 30]), line('Betrag 13,00', [10, 80, 150, 100])]}
        second = {'lines': [line('Name', [12, 11, 102, 31]), line('Datum', [10, 45, 100, 65]), line('Betrag 18,00', [10, 80, 150, 100])]}
        result = fuse(first, second)
        self.assertEqual(result['text'], 'Name\nDatum\nBetrag 13,00')
        self.assertEqual(result['added_lines'], 1)
        self.assertEqual(result['conflicts'], 1)
        self.assertNotIn('18,00', result['text'])

    def test_no_duplicate_or_shifted_line(self):
        first = {'text': 'Straße', 'lines': [line('Straße', [10, 10, 80, 30])]}
        result = fuse(first, {'lines': [line('Straße', [11, 32, 81, 52])]})
        self.assertEqual(result['text'], 'Straße')
        self.assertEqual(result['added_lines'], 0)

    def test_empty_and_malformed_geometry(self):
        self.assertEqual(fuse({'text': '', 'lines': []}, {'lines': [line('Ort', [10, 10, 60, 30])]})['text'], 'Ort')
        with self.assertRaises(ValueError):
            fuse({'text': '', 'lines': []}, {'lines': [line('bad', [10, 10, 0, 2])]})

    def test_trigger_does_not_use_truth(self):
        self.assertTrue(should_retry({'text': 'Name', 'lines': [line('Name', [1, 1, 10, 10])]}))
        rows = [line('Normaler Feldtext', [10, 20*i, 100, 20*i+10]) for i in range(8)]
        self.assertFalse(should_retry({'text': '\n'.join(x['text'] for x in rows), 'lines': rows}))

    def test_region_policy_is_bounded_clipped_and_truth_independent(self):
        lines = [line('?', [1, 1, 11, 21]), line('X', [2, 2, 12, 22]),
                 line('Müller', [30, 50, 100, 70]), line('ß', [200, 200, 210, 220])]
        regions = retry_regions(lines, 220, 215)
        self.assertEqual(regions, [[0, 0, 35, 33], [176, 188, 220, 215]])
        self.assertEqual(retry_regions(lines, 220, 215, limit=0), [])
        self.assertEqual(len(retry_regions(lines + [line('?', [1, 100, 11, 120]),
                                                   line('?', [1, 150, 11, 170]),
                                                   line('?', [1, 180, 11, 200])], 220, 215)), 4)
        self.assertEqual(restore_line(line('Datum', [2, 3, 20, 13]), [100, 200, 150, 250])['box'], [102, 203, 120, 213])
        with self.assertRaises(ValueError):
            retry_regions(lines, 220, 215, limit=5)
        with self.assertRaises(ValueError):
            restore_line(line('Datum', [2, 3, 51, 13]), [100, 200, 150, 250])

    def test_four_feldexa_forms_render_all_thirteen_pages(self):
        cases = list(form_cases(Path('fixtures/feldexa')))
        self.assertEqual(len(cases), 13)
        self.assertEqual(len({c['id'] for c, _ in cases}), 13)
        self.assertTrue(all(c['modality'] == 'feldexa-form' and c['fields'] and im.width > 0 for c, im in cases))
        self.assertEqual(sum(len(c['fields']) for c, _ in cases), 63)

    def test_paired_fusion_scores_full_and_selective_without_raw_text(self):
        truth = [{'id': 'p', 'modality': 'feldexa-form', 'fields': ['Name', 'Datum']}]
        rows = []
        for p in lab.PIPELINES:
            for variant, lines in [('baseline', [line('Name', [10, 10, 80, 30])]), ('contrast1-fusion', [line('Datum', [10, 80, 80, 100])])]:
                rows.append({'id': 'p', 'pipeline': p, 'variant': variant, 'lines': lines, 'text': lines[0]['text'], 'line_count': 1, 'status': 'ok', 'elapsed_ms': 10, 'heap_used_bytes': 100})
        result = lab.score_fusion(truth, rows)
        self.assertEqual(result['app_statuses']['feldexa-gray-bitmap'], 'LAB_CANDIDATE')
        self.assertEqual(result['selective_replay']['app_statuses']['feldexa-gray-bitmap'], 'LAB_CANDIDATE')
        self.assertNotIn('Datum', str(result))
        self.assertEqual(result['rows'][1]['elapsed_ms'], 20)

    def test_region_retry_scores_only_merged_full_page_results(self):
        truth = [{'id': 'p', 'modality': 'feldexa-form', 'fields': ['Datum']}]
        rows = []
        for pipeline in lab.PIPELINES:
            rows.extend([
                {'id': 'p', 'pipeline': pipeline, 'variant': 'baseline', 'text': '?',
                 'lines': [line('?', [10, 10, 20, 30])], 'line_count': 1,
                 'status': 'ok', 'elapsed_ms': 10},
                {'id': 'p', 'pipeline': pipeline, 'variant': 'region-crop-retry', 'text': 'Datum',
                 'lines': [line('Datum', [25, 2, 40, 18])], 'line_count': 1,
                 'retry_regions': 1, 'raster_width': 100, 'raster_height': 100,
                 'status': 'ok', 'elapsed_ms': 6},
            ])
        result = lab.score_region_retry(truth, rows)
        self.assertEqual(result['app_statuses'], {p: 'LAB_CANDIDATE' for p in lab.PIPELINES})
        self.assertEqual(result['diagnostics'][0]['retry_regions'], 1)
        self.assertNotIn('Datum', str(result))
        rows[1]['retry_regions'] = 0
        with self.assertRaises(ValueError):
            lab.score_region_retry(truth, rows)
        rows[1]['retry_regions'] = 1
        rows[1]['lines'] = [line('Datum', [25, 2, 101, 18])]
        with self.assertRaises(ValueError):
            lab.score_region_retry(truth, rows)
