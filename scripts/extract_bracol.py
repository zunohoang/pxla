"""Salvage BRACOL leaf images (truncated inner zip, no central directory) into data/raw/coffee.

Source: BRACOL v1, https://doi.org/10.17632/yy2k5y8mxg.1 (CC BY 4.0).
Labels follow dataset.csv predominant_stress 0-4; code 5 is undocumented and skipped.
group = image id (dataset has no plant id); 75/25 stratified split, seed 42.

Usage (from project root):
    python scripts/extract_bracol.py <path to yy2k5y8mxg-1.zip>
    python scripts/import_folders.py data/raw/coffee --crop coffee --domain plain
        --source https://doi.org/10.17632/yy2k5y8mxg.1 --license "CC BY 4.0"
        --group-map data/coffee_group_map.csv --out data/manifest.csv
"""
import csv, io, random, struct, sys, zipfile, zlib, collections
from pathlib import Path
from PIL import Image

LABELS = {'0': 'healthy', '1': 'leaf_miner', '2': 'rust', '3': 'brown_spot', '4': 'cercospora'}  # 5 unknown: skipped
outer = zipfile.ZipFile(sys.argv[1])
d = outer.read(outer.namelist()[0])
files, p = {}, 0
while d[p:p+4] == b'PK\x03\x04':
    m, = struct.unpack('<H', d[p+8:p+10]); crc, cs, us, nl, el = struct.unpack('<IIIHH', d[p+14:p+30])
    name = d[p+30:p+30+nl].decode(); body = d[p+30+nl+el:p+30+nl+el+cs]
    p += 30+nl+el+cs
    if name.endswith('/') or len(body) < cs: continue
    try:
        data = zlib.decompress(body, -15) if m == 8 else body
    except zlib.error: continue
    if zlib.crc32(data) == crc: files[name.split('/')[-1]] = data
rows = list(csv.DictReader(io.StringIO(files.pop('dataset.csv').decode())))
root = Path('data/raw/coffee'); out = []
for r in rows:
    label, data = LABELS.get(r['predominant_stress']), files.get(r['id'] + '.jpg')
    if not label or data is None: continue
    try: Image.open(io.BytesIO(data)).load()
    except Exception: continue
    dest = root / label / (r['id'] + '.jpg'); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    out.append((f'coffee_{label}_{r["id"]}', label))
random.seed(42); bylab = collections.defaultdict(list)
for ident, label in out: bylab[label].append(ident)
with open('data/coffee_group_map.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id', 'group', 'split'])
    for label, ids in sorted(bylab.items()):
        random.shuffle(ids); cut = round(len(ids) * 0.75)
        for i, ident in enumerate(ids): w.writerow([ident, ident, 'train' if i < cut else 'test'])  # group=image: BRACOL gives no plant ID
print(len(out), {k: len(v) for k, v in bylab.items()})
