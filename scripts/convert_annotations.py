"""Convert LeafLab/LabelMe polygon JSON into original-resolution binary masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def convert(path, out):
    """Rasterize leaf/lesion polygons; reject lesion pixels outside leaf."""
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    size = (record['imageWidth'], record['imageHeight'])
    masks = {name: Image.new('L', size) for name in ['leaf', 'lesion']}
    for shape in record['shapes']:
        if shape['label'] not in masks or shape.get('shape_type', 'polygon') != 'polygon':
            raise ValueError('Only leaf/lesion polygons supported')
        points = [tuple(p) for p in shape['points']]
        if len(points) < 3:
            raise ValueError('Polygon needs >=3 vertices')
        ImageDraw.Draw(masks[shape['label']]).polygon(points, fill=255)
    leaf, lesion = np.array(masks['leaf']) > 0, np.array(masks['lesion']) > 0
    if not leaf.any() or np.any(lesion & ~leaf):
        raise ValueError('Need nonempty leaf and lesion subset of leaf: ' + str(path))
    Path(out).mkdir(parents=True, exist_ok=True)
    for key, image in masks.items():
        image.save(Path(out) / (Path(path).stem + '_' + key + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_files', nargs='+')
    parser.add_argument('--out', default='data/masks')
    args = parser.parse_args()
    for source in args.json_files:
        convert(source, args.out)
