"""Pinned, rights-cleared Feldexa development forms, rendered as images for real OCR."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def form_cases(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['schema'] != 1:
        raise ValueError('Invalid form manifest')
    for doc in manifest['forms']:
        path = root / (doc['id'] + '.pdf')
        if hashlib.sha256(path.read_bytes()).hexdigest() != doc['sha256']:
            raise ValueError('Feldexa fixture digest mismatch')
        info = subprocess.check_output(['pdfinfo', str(path)], text=True, timeout=30)
        attrs = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
        if attrs['Form'].strip() != 'none' or attrs['Encrypted'].strip() != 'no' or int(attrs['Pages']) != len(doc['pages']):
            raise ValueError('Expected unencrypted, static PDF with pinned page inventory')
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'page'
            subprocess.run(['pdftoppm', '-scale-to', '2400', '-png', str(path), str(prefix)],
                           check=True, capture_output=True, timeout=90)
            sources = sorted(Path(tmp).glob('page-*.png'), key=lambda p: int(p.stem.split('-')[-1]))
            if len(sources) != len(doc['pages']):
                raise ValueError('Incomplete form raster inventory')
            for page, source in zip(doc['pages'], sources):
                if page['index'] != int(source.stem.split('-')[-1]) - 1 or not page['labels']:
                    raise ValueError('Invalid label or page mapping')
                with Image.open(source) as im:
                    yield {'id': f'feldexa-{doc["id"]}-{page["index"]+1}', 'modality': 'feldexa-form',
                           'page': page['index']+1, 'fields': page['labels']}, im.convert('RGB')
