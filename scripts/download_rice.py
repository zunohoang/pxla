"""Download UCI 486 CC BY 4.0 images; no Kaggle credentials required."""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import write_manifest

URL = 'https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip'
SOURCE = 'https://doi.org/10.24432/C5R013'


def main():
    """Create fixed stratified 90/30 split and record downloaded archive SHA256."""
    root = Path('data/raw/rice')
    root.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={'User-Agent': 'LeafLab academic project'})
    with urllib.request.urlopen(request, timeout=120) as response:
        blob = response.read(100_000_000)
    rows = []
    mapping = {'Bacterial leaf blight': 'bacterial_leaf_blight',
               'Brown spot': 'brown_spot', 'Leaf smut': 'leaf_smut'}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for item in sorted(z.namelist()):
            p = Path(item)
            if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            label = next((v for k, v in mapping.items() if k in item), None)
            if label is None:
                continue
            dest = root / label / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(item))
            ident = label + '_' + p.stem
            rows.append(dict(id=ident, path=dest.as_posix(), crop='rice', label=label,
                             domain='plain', group=ident, split='train', leaf_mask='',
                             lesion_mask='', source=SOURCE, license='CC BY 4.0'))
    rng = np.random.default_rng(42)
    for label in sorted(mapping.values()):
        subset = [r for r in rows if r['label'] == label]
        if len(subset) != 40:
            raise ValueError(f'Expected 40 images of {label}, found {len(subset)}')
        for idx in rng.permutation(len(subset))[:10]:
            subset[idx]['split'] = 'test'
    write_manifest('data/manifest.csv', rows)
    Path('data/rice_provenance.json').write_text(json.dumps(dict(
        url=URL, sha256=hashlib.sha256(blob).hexdigest(), images=len(rows),
        seed=42, train=90, test=30,
        group_limitation='UCI has no plant/session IDs; image IDs are proxies. '
        'Manual near-duplicate and acquisition audit required.'), indent=2), encoding='utf-8')
    print(f'Downloaded {len(rows)} images. No healthy class. No field photos or lesion GT.')


if __name__ == '__main__':
    main()
