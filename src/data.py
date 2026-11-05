"""Manifest validation with split, duplicate and acquisition-group safeguards."""
import csv
import hashlib
from pathlib import Path

FIELDS = ['id', 'path', 'crop', 'label', 'domain', 'group', 'split',
          'leaf_mask', 'lesion_mask', 'source', 'license']


def read_manifest(path, check_hash=True):
    """Read CSV, resolve project-relative paths, reject leakage and bad schema."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Manifest columns must include: ' + ','.join(FIELDS))
        rows = list(reader)
    ids, groups, hashes = set(), {}, {}
    for r in rows:
        if r['id'] in ids or not r['id'] or not r['label'] or not r['group']:
            raise ValueError('Empty label/group or duplicate/empty id: ' + r['id'])
        ids.add(r['id'])
        if r['crop'] not in ('rice', 'coffee') or r['domain'] not in ('plain', 'field'):
            raise ValueError('Invalid crop/domain: ' + r['id'])
        if r['split'] not in ('train', 'test', 'segval', 'segtest'):
            raise ValueError('Invalid split: ' + r['id'])
        group = r['crop'] + ':' + r['group']
        if group in groups and groups[group] != r['split']:
            raise ValueError('Acquisition group crosses splits: ' + group)
        groups[group] = r['split']
        image = Path(r['path'])
        if not image.is_file():
            raise ValueError('Missing image: ' + str(image))
        if check_hash:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError('Duplicate file content: ' + str(image) + ' / ' + hashes[digest])
            hashes[digest] = str(image)
        for col in ('leaf_mask', 'lesion_mask'):
            if r[col] and not Path(r[col]).is_file():
                raise ValueError('Missing annotation: ' + r[col])
    return rows


def write_manifest(path, rows):
    """Write manifest in UTF-8 with fixed column order."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
