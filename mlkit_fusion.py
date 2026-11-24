"""Conservative, truth-independent, same-raster OCR retries for research."""
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


def retry_regions(lines, width, height, limit=4):
    """Select bounded crop rectangles from first-pass OCR alone.

    Short/replacement-character lines are the only triggers. Rectangles are
    expanded around the first-pass box, clipped to the raster and de-duplicated
    by overlap. Ground truth, labels and expected field locations are not read.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError('Invalid raster dimensions')
    if not isinstance(limit, int) or limit < 0 or limit > 4:
        raise ValueError('Retry region limit must be between 0 and 4')
    if limit == 0:
        return []
    selected = []
    ordered = sorted((checked(x) for x in lines), key=lambda x: (x['box'][1], x['box'][0]))
    for line in ordered:
        text = normalized(line['text'])
        if len(text) > 2 and '\ufffd' not in text:
            continue
        x1, y1, x2, y2 = line['box']
        line_height = y2-y1
        pad_x = max(24, line_height)
        pad_y = max(12, line_height//2)
        region = [max(0, x1-pad_x), max(0, y1-pad_y),
                  min(width, x2+pad_x), min(height, y2+pad_y)]
        if region[2] <= region[0] or region[3] <= region[1]:
            raise ValueError('Retry region outside raster')
        if any(overlaps(existing, region) for existing in selected):
            continue
        selected.append(region)
        if len(selected) == limit:
            break
    return selected


def restore_line(line, crop):
    """Map a crop-relative OCR box back to the immutable source raster."""
    line = checked(line)
    x1, y1, x2, y2 = crop
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid crop geometry')
    b = line['box']
    if b[0] < 0 or b[1] < 0 or b[2] > x2-x1 or b[3] > y2-y1:
        raise ValueError('OCR box lies outside crop')
    restored = dict(line)
    restored['box'] = [b[0]+x1, b[1]+y1, b[2]+x1, b[3]+y1]
    return checked(restored)


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
