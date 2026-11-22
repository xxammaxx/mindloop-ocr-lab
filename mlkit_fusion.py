"""Conservative, truth-independent, same-raster OCR line fusion for research."""
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def checked(line):
    b = line['box']
    if len(b) != 4 or not all(isinstance(x, int) for x in b) or b[2] <= b[0] or b[3] <= b[1] or not normalized(line['text']):
        raise ValueError('Invalid OCR line geometry')
    return line


def overlaps(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return x*y >= 0.15*area


def near(a, b):
    height = max(a[3]-a[1], b[3]-b[1])
    return abs(a[0]-b[0]) <= height and abs(a[1]-b[1]) <= 1.5*height


def should_retry(first):
    lines = [checked(x) for x in first['lines']]
    return len(lines) <= 4 or any(len(normalized(x['text'])) <= 2 or '\ufffd' in x['text'] for x in lines)


def fuse(first, second):
    original = [checked(x) for x in first['lines']]
    added, conflicts = [], 0
    for line in map(checked, second['lines']):
        possible = original + added
        matches = [x for x in possible if overlaps(x['box'], line['box']) or
                   (normalized(x['text']) == normalized(line['text']) and near(x['box'], line['box']))]
        if matches:
            if not any(normalized(x['text']) == normalized(line['text']) for x in matches):
                conflicts += 1  # retain original, no forced correction from correlated evidence
            continue
        added.append(line)
    text = first['text']
    if added:
        if normalized(text) == normalized('\n'.join(x['text'] for x in original)):
            ordered = original[:]
            for line in sorted(added, key=lambda x: (x['box'][1], x['box'][0])):
                index = next((i for i, x in enumerate(ordered) if (x['box'][1], x['box'][0]) > (line['box'][1], line['box'][0])), len(ordered))
                ordered.insert(index, line)
            text = '\n'.join(x['text'] for x in ordered)
        else:
            text = '\n'.join([text] + [x['text'] for x in added]).strip()
    return {'text': text, 'added_lines': len(added), 'conflicts': conflicts}
