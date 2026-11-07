"""Import manually acquired class folders; inspect groups before experiments."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import merge_manifest


def main():
    """Generate a manifest for root/class/image; all images start in train split."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--crop', choices=['rice', 'coffee'], required=True)
    p.add_argument('--domain', choices=['plain', 'field'], required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--license', required=True)
    p.add_argument('--out', default='data/imported.csv')
    p.add_argument('--group-map', type=Path, help='CSV with id,group,split; required for field photos')
    args = p.parse_args()
    if args.domain == 'field' and args.group_map is None:
        p.error('Field photos require --group-map with real plant/session group IDs and splits')
    mapping = {}
    if args.group_map:
        with args.group_map.open(encoding='utf-8-sig', newline='') as f:
            mapping = {r['id']: r for r in csv.DictReader(f)}
    rows = []
    for image in sorted(args.root.glob('*/*')):
        if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        ident = f'{args.crop}_{image.parent.name}_{image.stem}'
        if args.group_map and ident not in mapping:
            raise ValueError('Missing group mapping for '+ident)
        rows.append(dict(id=ident, path=image.as_posix(), crop=args.crop,
                         label=image.parent.name, domain=args.domain,
                         group=mapping.get(ident, {}).get('group', ident),
                         split=mapping.get(ident, {}).get('split', 'train'),
                         leaf_mask='', lesion_mask='', source=args.source,
                         license=args.license))
    merge_manifest(args.out, rows)
    print(f'{len(rows)} records. REQUIRED: replace group with plant/session IDs and set splits.')


if __name__ == '__main__':
    main()
