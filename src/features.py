"""Versioned lesion-ROI feature schema shared by training, inference and reports."""
import numpy as np
from .algorithms import (components, distance_transform, glcm, glcm_properties,
                         lbp8, morphology, region_shape, rgb_to_hsv, rgb_to_lab)

FEATURE_SCHEMA = 'leaflab-2'
COLOR_NAMES = tuple(f'{s}_{c}_bin_{b}' for s in ['hsv', 'lab']
                    for c in range(3) for b in range(16)) + tuple(
                        f'lab_{s}_{c}' for s in ['mean', 'std'] for c in range(3))
TEXTURE_NAMES = tuple(f'glcm_{p}' for p in ['contrast', 'dissimilarity', 'homogeneity',
                                          'asm', 'energy', 'correlation']) + tuple(
                                              f'lbp_r{r}_bin_{b}' for r in [1, 2] for b in range(10))
SHAPE_NAMES = ('lesion_count', 'mean_area_fraction', 'max_area_fraction',
               'mean_eccentricity', 'mean_solidity', 'edge_distance_normalized', 'lesion_ratio')
FEATURE_NAMES = COLOR_NAMES + TEXTURE_NAMES + SHAPE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_GROUPS = {'color': np.arange(len(COLOR_NAMES)),
                  'texture': np.arange(len(COLOR_NAMES), len(COLOR_NAMES)+len(TEXTURE_NAMES)),
                  'shape': np.arange(len(COLOR_NAMES)+len(TEXTURE_NAMES), FEATURE_DIM),
                  'all': np.arange(FEATURE_DIM)}


def features(rgb, leaf, lesion, valid=None):
    """135 lesion-ROI features: color 102, texture 26, shape 7.

    No image-quadrant features. ROI below 1% of leaf returns fixed zeros.
    """
    if not leaf.any():
        raise ValueError('Nonempty leaf required')
    roi = lesion & leaf
    if valid is not None:
        roi &= valid
    if roi.sum()/leaf.sum() < .01:
        return np.zeros(FEATURE_DIM)
    hsv, lab = rgb_to_hsv(rgb), rgb_to_lab(rgb)
    color = []
    for space, ranges in [(hsv, [(0, 360), (0, 1), (0, 1)]),
                          (lab, [(0, 100), (-128, 128), (-128, 128)])]:
        for c, bounds in enumerate(ranges):
            hist, _ = np.histogram(space[..., c][roi], bins=16, range=bounds)
            color.extend(hist/max(hist.sum(), 1))
    color.extend(lab[roi].mean(0))
    color.extend(lab[roi].std(0))
    gray = np.round(rgb @ np.array([.299, .587, .114])).astype(np.uint8)
    texture = list(np.mean([glcm_properties(glcm(gray, roi, dy, dx))
                            for dy, dx in [(0, 1), (1, 0), (1, 1), (-1, 1)]], axis=0))
    vals = np.arange(256, dtype=np.uint16)
    bits = (vals[:, None] >> np.arange(8)) & 1
    transitions = (bits != np.roll(bits, 1, axis=1)).sum(1)
    lookup = np.where(transitions <= 2, bits.sum(1), 9)
    for radius in [1, 2]:
        valid_centers = morphology(roi, 'erode', 2*radius+1)
        codes = lbp8(gray, radius)
        hist = np.bincount(lookup[codes[valid_centers]], minlength=10).astype(float)
        texture.extend(hist/max(hist.sum(), 1))
    labels, areas = components(roi)
    eccentricities, solidities = [], []
    for label in range(1, len(areas)):
        yy, xx = np.nonzero(labels == label)
        local = labels[yy.min():yy.max()+1, xx.min():xx.max()+1] == label
        ecc, solidity = region_shape(local)
        eccentricities.append(ecc)
        solidities.append(solidity)
    distance = distance_transform(leaf)
    shape = [len(areas)-1, areas[1:].mean()/leaf.sum(), areas[1:].max()/leaf.sum(),
             np.mean(eccentricities), np.mean(solidities),
             distance[roi].mean()/np.sqrt(leaf.sum()), roi.sum()/leaf.sum()]
    result = np.r_[color, texture, shape]
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError('Invalid feature vector')
    return result
