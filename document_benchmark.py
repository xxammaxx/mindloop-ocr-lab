"""Regression gate for paired photographs and scanned PDFs; no source text logged."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from lab import BASELINE, CANDIDATES, normalize, recognize


def field_hits(observed, expected):
    observed = normalize(observed)
    return [i for i, field in enumerate(expected)
            if re.search(r'(?<!\w)(?<!\d[.,])' + re.escape(normalize(field)) + r'(?!\w)(?![.,]\d)', observed)]


def field_score(observed, expected):
    return {'matched': len(field_hits(observed, expected)), 'total': len(expected)}


def paired_improves(baseline, candidate):
    def index(rows):
        return {(r['document'], r['page'], r['modality']): (r['total'], set(r['hits'])) for r in rows}
    before, after = index(baseline), index(candidate)
    if not before or before.keys() != after.keys() or len(before) != len(baseline) or len(after) != len(candidate):
        return False
    if not all(before[k][0] == after[k][0] and before[k][1] <= after[k][1] for k in before):
        return False
    return any(before[k][1] < after[k][1] for k in before)


def local_file(root, name, suffix):
    path = (root / name).resolve()
    if path.parent != root.resolve() or path.suffix.lower() not in suffix or not path.is_file():
        raise ValueError('Invalid corpus file')
    return path


def load_manifest(root):
    payload = json.loads((root / 'manifest.json').read_text())
    if payload.get('schema') != 1 or not payload.get('documents'):
        raise ValueError('Invalid corpus manifest')
    documents = payload['documents']
    ids = set()
    photos = set()
    for doc in documents:
        if not re.fullmatch(r'[a-z0-9-]+', doc['id']) or doc['id'] in ids or not doc['pages']:
            raise ValueError('Invalid or duplicate document identifier')
        ids.add(doc['id'])
        local_file(root, doc['scan'], {'.pdf'})
        for page in doc['pages']:
            photo = local_file(root, page['photo'], {'.png', '.jpg', '.jpeg'})
            if photo in photos:
                raise ValueError('Duplicate photo in paired manifest')
            photos.add(photo)
            if not page['fields'] or not all(isinstance(f, str) and normalize(f) for f in page['fields']):
                raise ValueError('Every page requires independently transcribed expected fields')
    return documents


def valid_scan_inventory(inventory, expected_pages):
    rows = [line.split() for line in inventory.splitlines()[2:] if line.strip()]
    return [row[:3] for row in rows] == [[str(i), str(i - 1), 'image'] for i in range(1, expected_pages + 1)]


def evaluate_corpus(root, config, lang):
    rows = []
    for doc in load_manifest(root):
        with tempfile.TemporaryDirectory() as folder:
            inventory = subprocess.check_output(['pdfimages', '-list', str(root / doc['scan'])], text=True, timeout=60)
            pdfinfo = subprocess.check_output(['pdfinfo', str(root / doc['scan'])], text=True, timeout=30)
            if not re.search(r'^Pages:\s+' + str(len(doc['pages'])) + r'\s*$', pdfinfo, re.MULTILINE) or not valid_scan_inventory(inventory, len(doc['pages'])):
                raise ValueError('Expected exactly one raster per physical PDF page')
            # Fixture PDFs are raster scans; extraction avoids new rasterization artifacts.
            subprocess.run(['pdfimages', '-png', str(root / doc['scan']), str(Path(folder) / 'page')],
                           check=True, capture_output=True, timeout=60)
            pages = sorted(Path(folder).glob('page-*.png'))
            if len(pages) != len(doc['pages']):
                raise ValueError('Number of scanned images and photo pages differ')
            for i, (page, scan) in enumerate(zip(doc['pages'], pages), 1):
                for modality, source in [('scan', scan), ('photo', root / page['photo'])]:
                    with Image.open(source) as image:
                        hits = field_hits(recognize(image, config, lang), page['fields'])
                    rows.append({'document': doc['id'], 'page': i, 'modality': modality,
                                 'matched': len(hits), 'total': len(page['fields']), 'hits': hits})
    return rows


def corpus_digest(root):
    files = {root / 'manifest.json'}
    for doc in load_manifest(root):
        files.add(root / doc['scan'])
        files.update(root / p['photo'] for p in doc['pages'])
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.name.encode() + b'\0' + hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--lang', choices=['deu', 'eng'], default='deu')
    parser.add_argument('--candidate', choices=[c['id'] for c in [BASELINE, *CANDIDATES]], default='baseline')
    args = parser.parse_args()
    config = next(c for c in [BASELINE, *CANDIDATES] if c['id'] == args.candidate)
    rows = evaluate_corpus(args.corpus, config, args.lang)
    print(json.dumps({'schema': 1, 'corpus_sha256': corpus_digest(args.corpus), 'config': config,
                      'rows': rows, 'modalities': {m: {'matched': sum(r['matched'] for r in rows if r['modality'] == m),
                                                       'total': sum(r['total'] for r in rows if r['modality'] == m)}
                                                  for m in ('photo', 'scan')}}, indent=2))


if __name__ == '__main__':
    main()
