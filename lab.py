"""Bounded, CPU-only OCR experiments on generated non-personal documents."""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BASELINE = {'id': 'baseline', 'preprocess': 'gray', 'psm': 6}
CANDIDATES = [
    {'id': f'{mode}-psm{psm}', 'preprocess': mode, 'psm': psm}
    for mode in ('contrast1', 'contrast', 'sharpen', 'threshold', 'scale2', 'gray')
    for psm in (6, 11) if (mode, psm) != ('gray', 6)
]


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, a_char in enumerate(a, 1):
        row = [i]
        for j, b_char in enumerate(b, 1):
            row.append(min(prev[j]+1, row[-1]+1, prev[j-1]+(a_char != b_char)))
        prev = row
    return prev[-1]


def improves(baseline, candidate):
    return (candidate['cer'] < baseline['cer'] - 1e-9
            and candidate['critical_accuracy'] >= baseline['critical_accuracy'])


def plan(records, limit):
    if not 1 <= limit <= 2:
        raise ValueError('Each experiment batch can evaluate 1 or 2 candidates')
    done = {record['candidate'] for record in records}
    pending = [c for c in CANDIDATES if c['id'] not in done]
    scored = [r for r in records if 'dev' in r and not r.get('status', '').startswith('INCONCLUSIVE')]
    if scored:
        best = min(scored, key=lambda r: (-sum(x['matched'] for x in r.get('paired', [])), r['dev']['cer'], r['candidate']))
        family = best['candidate'].rsplit('-psm', 1)[0]
        pending.sort(key=lambda c: c['preprocess'] != family)
    return pending[:limit]


def fixtures(split):
    if split not in ('dev', 'validation'):
        raise ValueError('Unknown split')
    rng = random.Random(8243 if split == 'dev' else 37612)
    font = ImageFont.truetype(str(FONT), 29)
    cases = []
    for index in range(5):
        form = index % 2
        subject = ('Müller', 'Jäger', 'Straße', 'Größe', 'Gebühr')[index] if split == 'dev' else ('Grün', 'Bär', 'Öl', 'Maß', 'Größe')[index]
        amount = f'{rng.randrange(12, 900)},{rng.randrange(100):02d}'
        year = '2025' if split == 'dev' else '2024'
        lines = ([f'Wohngeldantrag {year}', f'Beispielname: {subject}', f'Monatsmiete: {amount} EUR', 'Hausgröße und Fläche'] if form else
                 [f'Nebenkostenabrechnung {year}', f'Beispielobjekt: {subject}', f'Heizkosten: {amount} EUR', 'Wärme und Gebühren'])
        critical = [subject, amount, year]
        if index == 4:
            lines = (['Änderung: ÖL und ÜBERTRAG', 'Straße, Höhe, Gebühr, ẞ', 'Fläche 32 m², Wasser 12 m³', '§ 5 – Betrag 18,45 €'] if split == 'dev' else
                     ['ÄRGER ÜBER ÖL: Fläche', 'Größe, Gebühr, Maße, ẞ', 'Wohnraum 48 m², Gas 20 m³', '§ 7 – Betrag 27,61 €'])
            critical = ['ẞ', 'm²', 'm³', '§', '€', 'Fläche', 'Gebühr']
        image = Image.new('L', (840, 245), 246)
        draw = ImageDraw.Draw(image)
        for row, line in enumerate(lines):
            draw.text((20, 18 + row * 47), line, font=font, fill=30)
        angle = rng.uniform(-1.9, 1.9)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=246)
        if index >= 2:
            image = image.filter(ImageFilter.GaussianBlur(1.1)).resize((630, 184))
        image = Image.eval(image, lambda p: int(p * 0.68 + 75))
        cases.append({'id': f'{split}-{index}', 'cohort': 'form' if form else 'statement',
                      'truth': '\n'.join(lines), 'critical': critical, 'image': image})
    return cases


def prepare(image, mode):
    gray = ImageOps.grayscale(image)
    if mode == 'gray':
        return gray
    if mode == 'contrast':
        return ImageOps.autocontrast(gray)
    if mode == 'contrast1':
        return ImageOps.autocontrast(gray, cutoff=1)
    if mode == 'sharpen':
        return ImageOps.autocontrast(gray).filter(ImageFilter.UnsharpMask(2, 180, 2))
    if mode == 'threshold':
        return ImageOps.autocontrast(gray).point(lambda p: 255 if p > 150 else 0)
    if mode == 'scale2':
        return gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
    raise ValueError('Unknown preprocessing mode')


def recognize(image, config, lang):
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / 'synthetic.png'
        prepare(image, config['preprocess']).save(image_path)
        result = subprocess.run(['tesseract', str(image_path), 'stdout', '-l', lang, '--psm', str(config['psm'])],
                                check=True, capture_output=True, text=True, timeout=25,
                                env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
        return normalize(result.stdout)


def evaluate(cases, config, lang):
    rows = []
    for case in cases:
        truth, observed = normalize(case['truth']), recognize(case['image'], config, lang)
        correct = sum(bool(re.search(r'(?<!\w)' + re.escape(token) + r'(?!\w)', observed)) for token in case['critical'])
        rows.append({'id': case['id'], 'cohort': case['cohort'], 'edits': edit_distance(truth, observed),
                     'chars': len(truth), 'critical_correct': correct, 'critical_total': len(case['critical'])})

    def aggregate(items):
        return {'cer': sum(r['edits'] for r in items) / sum(r['chars'] for r in items),
                'critical_accuracy': sum(r['critical_correct'] for r in items) / sum(r['critical_total'] for r in items)}
    return {**aggregate(rows), 'cohorts': {c: aggregate([r for r in rows if r['cohort'] == c]) for c in ('form', 'statement')}, 'rows': rows}


def nonregressing(baseline, candidate):
    return all(candidate['cohorts'][c]['cer'] <= b['cer'] and
               candidate['cohorts'][c]['critical_accuracy'] >= b['critical_accuracy']
               for c, b in baseline['cohorts'].items())


def eligible_dev(baseline, measured, paired_gain=None):
    return nonregressing(baseline, measured) and (improves(baseline, measured) if paired_gain is None else paired_gain)


def load_ledger(path, context):
    if not path.exists():
        return {'schema': 1, 'context': context, 'records': []}
    state = json.loads(path.read_text())
    if state.get('schema') != 1 or state.get('context') != context or not isinstance(state.get('records'), list):
        raise ValueError('Ledger context/schema mismatch; preserve original evidence')
    ids = [r['candidate'] for r in state['records']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate experiment receipts')
    return state


def save_ledger(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def context(lang, corpus=None):
    version = subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0]
    languages = subprocess.check_output(['tesseract', '--list-langs'], text=True).splitlines()
    if lang not in languages[1:]:
        raise RuntimeError(f'Required OCR language not installed: {lang}')
    tessdir = Path(languages[0].split('"')[1])
    metadata = {'engine': version, 'lang': lang, 'model_sha256': hashlib.sha256((tessdir / (lang + '.traineddata')).read_bytes()).hexdigest(),
                'pillow': PIL.__version__, 'font_sha256': hashlib.sha256(FONT.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if corpus is not None:
        from document_benchmark import corpus_digest
        metadata['corpus_sha256'] = corpus_digest(corpus)
        metadata['benchmark_sha256'] = hashlib.sha256(Path(__file__).with_name('document_benchmark.py').read_bytes()).hexdigest()
        pdfimages = subprocess.run(['pdfimages', '-v'], capture_output=True, text=True, check=True)
        metadata['pdfimages'] = (pdfimages.stderr or pdfimages.stdout).splitlines()[0]
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(), metadata


def run(root, lang, limit, corpus=None):
    key, metadata = context(lang, corpus)
    path = root / key / 'ledger.json'
    state = load_ledger(path, key)
    state['environment'] = metadata
    dev, validation = fixtures('dev'), fixtures('validation')
    candidates = plan(state['records'], limit)
    if not candidates:
        print('EXHAUSTED: no novel candidate in this context')
        return
    if 'baseline' not in state:
        state['baseline'] = {'config': BASELINE, 'dev': evaluate(dev, BASELINE, lang), 'validation': evaluate(validation, BASELINE, lang)}
        if corpus is not None:
            from document_benchmark import evaluate_corpus
            state['baseline']['paired'] = evaluate_corpus(corpus, BASELINE, lang)
        save_ledger(path, state)
    for candidate in candidates:
        record = {'candidate': candidate['id'], 'config': candidate, 'status': 'REJECTED_DEV'}
        stage = 'dev'
        try:
            measured = evaluate(dev, candidate, lang)
            record['dev'] = measured
            paired_gain = None
            if corpus is not None:
                from document_benchmark import evaluate_corpus, paired_improves
                stage = 'paired'
                record['paired'] = evaluate_corpus(corpus, candidate, lang)
                paired_gain = paired_improves(state['baseline']['paired'], record['paired'])
                record['paired_strict_gain'] = paired_gain
            if eligible_dev(state['baseline']['dev'], measured, paired_gain):
                stage = 'validation'
                holdout = evaluate(validation, candidate, lang)
                record['validation'] = holdout
                valid = nonregressing(state['baseline']['validation'], holdout) and (corpus is not None or improves(state['baseline']['validation'], holdout))
                record['status'] = 'LAB_CANDIDATE' if valid else 'REJECTED_VALIDATION'
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            record['status'] = 'INCONCLUSIVE_TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'INCONCLUSIVE_ERROR'
            record['stage'] = stage
            record['error_type'] = type(exc).__name__
        state['records'].append(record)
        state['complete'] = not plan(state['records'], 1)
        save_ledger(path, state)
        print(candidate['id'], record['status'])
        if 'dev' in record:
            print('dev CER', record['dev']['cer'], 'critical', record['dev']['critical_accuracy'])
        if 'paired' in record:
            for modality in ('photo', 'scan'):
                scores = [r for r in record['paired'] if r['modality'] == modality]
                print(modality, sum(r['matched'] for r in scores), '/', sum(r['total'] for r in scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--lang', choices=['deu', 'eng'], default='deu')
    parser.add_argument('--max-candidates', type=int, choices=[1, 2], default=2)
    parser.add_argument('--corpus', type=Path, help='Explicit approved paired document corpus; fail if unavailable')
    arguments = parser.parse_args()
    run(arguments.results_dir, arguments.lang, arguments.max_candidates, arguments.corpus)
