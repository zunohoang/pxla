"""Qualitative leaf-method comparison on TRAIN only, never for test tuning."""
from pathlib import Path
import sys
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import read_manifest
from src.pipeline import Config, leaf_mask, load_rgb
from scripts.reference_methods import grabcut_reference


def main():
    """Show one training image per class and five explicit comparators."""
    rows = read_manifest('data/manifest.csv')
    selected = {}
    for row in rows:
        if row['split'] == 'train':
            selected.setdefault((row['crop'],row['label']), row)
    if not selected:
        return
    chosen = list(selected.values())[:6]
    methods = ['hybrid','hsv','exg_otsu','exg_exr','grabcut_reference']
    fig, axes = plt.subplots(len(chosen), 6, figsize=(15, 3*len(chosen)), squeeze=False)
    for y, row in enumerate(chosen):
        rgb = load_rgb(row['path'])
        axes[y,0].imshow(rgb)
        axes[y,0].set_title('TRAIN '+row['label'], fontsize=8)
        for x, method in enumerate(methods, 1):
            try:
                mask = grabcut_reference(rgb) if method == 'grabcut_reference' else leaf_mask(
                    rgb, replace(Config(), leaf_method=method))
                axes[y,x].imshow(mask, cmap='gray', vmin=0,vmax=1)
            except ValueError:
                axes[y,x].text(.5,.5,'Processing failed',ha='center')
            axes[y,x].set_title(method, fontsize=8)
    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    Path('results').mkdir(exist_ok=True)
    fig.savefig('results/leaf_methods_train.png', dpi=150)
    plt.close(fig)
    print('Saved results/leaf_methods_train.png; no IoU without manual ground truth.')


if __name__ == '__main__':
    main()
