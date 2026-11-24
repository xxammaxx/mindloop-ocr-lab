"""Finite, paired research on the same bundled ML Kit SDK as both consumers."""
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINES = ('umliva-nv21', 'feldexa-gray-bitmap')
CANDIDATES = ('region-v1',)
SDK = 'com.google.mlkit:text-recognition:16.0.1'


def pending(records):
    if any(row.get('status') == 'INCONCLUSIVE_INFRA' for row in records):
        return None
    done = {row['candidate'] for row in records}
    return next((candidate for candidate in CANDIDATES if candidate not in done), None)


def signature(corpus):
    h = hashlib.sha256()
    paths = [ROOT / n for n in ('mlkit_lab.py', 'mlkit_fusion.py', 'feldexa_forms.py', 'lab.py', 'document_benchmark.py', '.github/workflows/mlkit.yml')]
    paths += sorted(p for p in (ROOT / 'mlkit-android').rglob('*') if p.is_file() and ('src' in p.parts or p.suffix in ('.gradle', '.properties', '.sh')) and 'assets' not in p.parts and 'build' not in p.parts and '.gradle' not in p.parts)
    paths += [corpus / 'manifest.json', corpus / 'SHA256SUMS']
    paths += sorted(p for p in (corpus / 'feldexa').glob('*') if p.is_file())
    for p in paths:
        name = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name
        h.update(name.encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def ledger(root, corpus):
    key = signature(corpus)
    path = root / key / 'ledger.json'
    state = json.loads(path.read_text()) if path.exists() else {'schema': 1, 'context': key, 'environment': {'engine': SDK, 'scope': 'android-emulator', 'lang': 'latin'}, 'records': []}
    if state['context'] != key or state['environment']['engine'] != SDK:
        raise ValueError('Invalid ML Kit experiment context')
    return path, state


def score(truth, observations, candidate):
    from document_benchmark import field_hits
    from lab import edit_distance, normalize
    cases = {c['id']: c for c in truth}
    if not cases or len(cases) != len(truth):
        raise ValueError('Missing or duplicate input case')
    expected = {(v, p, i) for v in ('baseline', candidate) for p in PIPELINES for i in cases}
    keys = [(r['variant'], r['pipeline'], r['id']) for r in observations]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Missing, duplicate or unexpected observation')
    rows = []
    for r in observations:
        if r['status'] != 'ok' or not isinstance(r['text'], str) or not isinstance(r['elapsed_ms'], (int, float)) or not math.isfinite(r['elapsed_ms']) or r['elapsed_ms'] < 0:
            raise ValueError('Unsuccessful OCR observation')
        c = cases[r['id']]
        hits = field_hits(r['text'], c['fields'])
        row = {k: r[k] for k in ('variant', 'pipeline', 'id', 'elapsed_ms', 'line_count')}
        row.update(modality=c['modality'], hits=hits, matched=len(hits), total=len(c['fields']))
        if 'page' in c:
            row['page'] = c['page']
        if 'truth' in c:
            raw_truth = normalize(c['truth'])
            row.update(edits=edit_distance(raw_truth, normalize(r['text'])), chars=len(raw_truth))
        rows.append(row)
    statuses = {}
    for pipeline in PIPELINES:
        before = {r['id']: r for r in rows if r['pipeline'] == pipeline and r['variant'] == 'baseline'}
        after = {r['id']: r for r in rows if r['pipeline'] == pipeline and r['variant'] == candidate}
        nonregressing = all(set(b['hits']) <= set(after[id]['hits']) and after[id].get('edits', 0) <= b.get('edits', 0) for id,b in before.items())
        gain = any(b['modality'] in ('photo', 'scan', 'feldexa-form') and set(b['hits']) < set(after[id]['hits']) for id,b in before.items())
        statuses[pipeline] = 'LAB_CANDIDATE' if nonregressing and gain else 'REJECTED_DEV'
    return {'candidate': candidate, 'config': {'preprocess': candidate, 'sdk': SDK, 'max_source_side': 2400}, 'status': 'LAB_CANDIDATE' if 'LAB_CANDIDATE' in statuses.values() else 'REJECTED_DEV', 'app_statuses': statuses, 'rows': rows}


def score_fusion(truth, observations):
    """Score the measured first pass plus at most one actual regional second pass.

    Selection and coordinates are produced on-device from baseline geometry only.
    No ground-truth access is allowed during triggering or fusion.
    """
    from mlkit_fusion import fuse
    candidate = 'region-v1'
    by_key = {(r['pipeline'], r['id'], r['variant']): r for r in observations}
    if len(by_key) != len(observations):
        raise ValueError('Duplicate input to regional fusion')
    full, diagnostics = [], []
    for r in observations:
        if r['variant'] == 'baseline':
            full.append(r)
            continue
        if r['variant'] != candidate or r.get('retry_regions') not in (0, 1):
            raise ValueError('Invalid or unbounded regional retry measurement')
        before = by_key.get((r['pipeline'], r['id'], 'baseline'))
        if before is None or r.get('line_count') != len(r.get('lines', [])):
            raise ValueError('Missing baseline or malformed regional retry')
        if r['retry_regions'] == 0 and (r['lines'] or r['text'].strip()):
            raise ValueError('No-retry observation must not invent lines')
        fused = fuse(before, r)
        full.append({**r, 'text': fused['text'],
                     'line_count': before['line_count'] + fused['added_lines'],
                     'elapsed_ms': before['elapsed_ms'] + r['elapsed_ms']})
        diagnostics.append({'pipeline': r['pipeline'], 'id': r['id'],
                            'retry_regions': r['retry_regions'],
                            'added_lines': fused['added_lines'],
                            'conflicts': fused['conflicts'],
                            'baseline_heap_snapshot_bytes': before['heap_used_bytes'],
                            'second_heap_snapshot_bytes': r['heap_used_bytes']})
    result = score(truth, full, candidate)
    result['config'].update(preprocess='region-v1', fusion='spatial-add-only',
                            max_regional_retries=1, magnification=2,
                            trigger='first-pass short/replacement-glyph line only')
    result['diagnostics'] = diagnostics
    result['evidence_limit'] = ('Actual bounded emulator crop re-OCR, but not production '
                                'app acceptance or field geometry/role correctness; '
                                'elapsed_ms is OCR-only and heap snapshots are not peak RAM.')
    return result


def prepare_assets(corpus, assets, truth_path, candidate):
    from PIL import Image, ImageOps
    from lab import fixtures, prepare
    from document_benchmark import load_manifest, valid_scan_inventory
    from feldexa_forms import form_cases
    assets.mkdir(parents=True, exist_ok=True)
    truth, index = [], []

    def add(case, source):
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        truth.append(case)
        for variant in ('baseline', candidate):
            out = image if variant == 'baseline' or candidate == 'region-v1' else prepare(image, variant)
            filename = f'{case["id"]}-{variant}.png'
            out.save(assets / filename)
            index.append({'id': case['id'], 'variant': variant, 'file': filename})

    for split in ('dev', 'validation'):
        for case in fixtures(split):
            add({'id': case['id'], 'modality': split, 'truth': case['truth'], 'fields': case['critical']}, case['image'])
    for case, image in form_cases(corpus / 'feldexa'):
        add(case, image)
    for doc in load_manifest(corpus):
        scan = corpus / doc['scan']
        with tempfile.TemporaryDirectory() as tmp:
            inventory = subprocess.check_output(['pdfimages', '-list', str(scan)], text=True, timeout=60)
            info = subprocess.check_output(['pdfinfo', str(scan)], text=True, timeout=30)
            if not valid_scan_inventory(inventory, len(doc['pages'])) or not any(line.split() == ['Pages:', str(len(doc['pages']))] for line in info.splitlines()):
                raise ValueError('Invalid physical PDF page inventory')
            subprocess.run(['pdfimages', '-png', str(scan), str(Path(tmp) / 'page')], check=True, capture_output=True, timeout=60)
            sources = sorted(Path(tmp).glob('page-*.png'))
            if len(sources) != len(doc['pages']):
                raise ValueError('Invalid extracted page count')
            for page, (spec, raster) in enumerate(zip(doc['pages'], sources), 1):
                for modality, path in [('photo', corpus / spec['photo']), ('scan', raster)]:
                    with Image.open(path) as image:
                        add({'id': f'{doc["id"]}-{modality}-{page}', 'modality': modality, 'page': page, 'fields': spec['fields']}, image)
    (assets / 'index.json').write_text(json.dumps(index))
    truth_path.write_text(json.dumps(truth, ensure_ascii=False))
    print(f'Prepared {len(truth)} cases, {len(index)} image variants; no raw OCR output')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pending', 'prepare', 'score', 'error'])
    parser.add_argument('--corpus', type=Path, default=Path('fixtures'))
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--error-stage', choices=['setup', 'emulator-or-score'])
    args = parser.parse_args()
    path, state = ledger(args.results_dir, args.corpus)
    if args.command == 'pending':
        print(pending(state['records']) or ('BLOCKED' if any(r.get('status') == 'INCONCLUSIVE_INFRA' for r in state['records']) else 'EXHAUSTED'))
        return
    if args.candidate != pending(state['records']):
        raise ValueError('Candidate is not next in the finite catalog')
    if args.command == 'prepare':
        prepare_assets(args.corpus, args.assets, args.truth, args.candidate)
        return
    if args.command == 'error':
        record = {'candidate': args.candidate, 'status': 'INCONCLUSIVE_INFRA', 'stage': args.error_stage}
    else:
        runtime = json.loads(args.runtime.read_text())
        if runtime['sdk'] != SDK or runtime['api'] != '35' or not runtime['aar_hashes']:
            raise ValueError('Unqualified emulator / ML Kit artifact')
        record = score_fusion(json.loads(args.truth.read_text()), json.loads(args.observations.read_text()))
        record['runtime'] = runtime
    state['records'].append(record)
    state['complete'] = record['status'] != 'INCONCLUSIVE_INFRA' and pending(state['records']) is None
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temp.replace(path)
    for pipeline, status in record.get('app_statuses', {}).items():
        print(pipeline, status)
        for variant in ('baseline', args.candidate):
            for modality in ('photo', 'scan', 'feldexa-form'):
                rows = [r for r in record['rows'] if r['pipeline'] == pipeline and r['variant'] == variant and r['modality'] == modality]
                print(variant, modality, sum(r['matched'] for r in rows), '/', sum(r['total'] for r in rows))


if __name__ == '__main__':
    main()
