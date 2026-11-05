"""Import manually acquired class folders; inspect groups before experiments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import write_manifest


def main():
    """Generate a manifest for root/class/image; all images start in train split."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--crop', choices=['rice', 'coffee'], required=True)
    p.add_argument('--domain', choices=['plain', 'field'], required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--license', required=True)
    p.add_argument('--out', default='data/imported.csv')
    args = p.parse_args()
    rows = []
    for image in sorted(args.root.glob('*/*')):
        if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        ident = f'{args.crop}_{image.parent.name}_{image.stem}'
        rows.append(dict(id=ident, path=image.as_posix(), crop=args.crop,
                         label=image.parent.name, domain=args.domain, group=ident,
                         split='train', leaf_mask='', lesion_mask='', source=args.source,
                         license=args.license))
    write_manifest(args.out, rows)
    print(f'{len(rows)} records. REQUIRED: replace group with plant/session IDs and set splits.')


if __name__ == '__main__':
    main()
