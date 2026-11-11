"""Versioned feature schema (lesion ROI + whole leaf) shared by training, inference and reports."""
import numpy as np
from .algorithms import (components, distance_transform, glcm, glcm_properties,
                         lbp8, morphology, region_shape, rgb_to_hsv, rgb_to_lab)

FEATURE_SCHEMA = 'leaflab-3'
COLOR_NAMES = tuple(f'{s}_{c}_bin_{b}' for s in ['hsv', 'lab']
                    for c in range(3) for b in range(16)) + tuple(
                        f'lab_{s}_{c}' for s in ['mean', 'std'] for c in range(3))
TEXTURE_NAMES = tuple(f'glcm_{p}' for p in ['contrast', 'dissimilarity', 'homogeneity',
                                          'asm', 'energy', 'correlation']) + tuple(
                                              f'lbp_r{r}_bin_{b}' for r in [1, 2] for b in range(10))
SHAPE_NAMES = ('lesion_count', 'mean_area_fraction', 'max_area_fraction',
               'mean_eccentricity', 'mean_solidity', 'edge_distance_normalized', 'lesion_ratio')
LESION_DIM = len(COLOR_NAMES) + len(TEXTURE_NAMES) + len(SHAPE_NAMES)
LEAF_CHANNELS = ('L', 'a', 'b')
LEAF_NAMES = (tuple(f'leaf_ab_hist_a{i}_b{j}' for i in range(10) for j in range(10))
              + tuple(f'leaf_lbp_{c}_r{r}_bin_{b}' for c in LEAF_CHANNELS for r in [1, 2, 3] for b in range(10))
              + tuple(f'leaf_grad_{c}_bin_{b}' for c in LEAF_CHANNELS for b in range(8))
              + tuple(f'leaf_glcm_{c}_d{d}_{p}' for c in LEAF_CHANNELS for d in [1, 2, 4]
                      for p in ['contrast', 'dissimilarity', 'homogeneity', 'asm', 'energy', 'correlation']))
FEATURE_NAMES = COLOR_NAMES + TEXTURE_NAMES + SHAPE_NAMES + LEAF_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_GROUPS = {'color': np.arange(len(COLOR_NAMES)),
                  'texture': np.arange(len(COLOR_NAMES), len(COLOR_NAMES)+len(TEXTURE_NAMES)),
                  'shape': np.arange(len(COLOR_NAMES)+len(TEXTURE_NAMES), LESION_DIM),
                  'lesion': np.arange(LESION_DIM),
                  'leaf': np.arange(LESION_DIM, FEATURE_DIM),
                  'all': np.arange(FEATURE_DIM)}
_BITS = (np.arange(256)[:, None] >> np.arange(8)) & 1
_UNIFORM = np.where((_BITS != np.roll(_BITS, 1, axis=1)).sum(1) <= 2, _BITS.sum(1), 9)
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (-1, 1)]


def _to_uint8(values, low, high):
    """Linearly quantize [low, high] to 0..255 for LBP/GLCM on a Lab channel."""
    return np.clip(np.round((values-low)*255/(high-low)), 0, 255).astype(np.uint8)


def leaf_features(processed, raw, leaf):
    """268 whole-leaf features that do not depend on the lesion mask.

    Colour: 10x10 joint a*b* histogram of the preprocessed image (lightness is
    ignored on purpose). Texture, on the raw image's L*, a*, b* channels: uniform
    LBP at radius 1, 2, 3; gradient-magnitude histogram; GLCM properties at
    distance 1, 2, 4 averaged over four directions. Histogram values are stored
    as square roots (Hellinger map) so that large bins do not dominate distances.
    """
    ab = rgb_to_lab(processed)[leaf][:, 1:]
    joint, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=10, range=[(-50, 50), (-30, 70)])
    colour = np.sqrt(joint.ravel()/max(joint.sum(), 1))
    lab = rgb_to_lab(raw)
    inner = morphology(leaf, 'erode', 3)
    lbp, gradient, cooccurrence = [], [], []
    for index, (low, high, top) in enumerate([(0, 100, 30), (-64, 64, 20), (-64, 64, 20)]):
        channel = _to_uint8(lab[..., index], low, high)
        for radius in [1, 2, 3]:
            centers = morphology(leaf, 'erode', 2*radius+1)
            hist = np.bincount(_UNIFORM[lbp8(channel, radius)[centers]], minlength=10)
            lbp.append(np.sqrt(hist/max(centers.sum(), 1)))
        gy, gx = np.gradient(lab[..., index])
        hist, _ = np.histogram(np.hypot(gy, gx)[inner], bins=8, range=(0, top))
        gradient.append(np.sqrt(hist/max(hist.sum(), 1)))
        for distance in [1, 2, 4]:
            cooccurrence.append(np.mean([glcm_properties(glcm(channel, leaf, dy*distance, dx*distance))
                                         for dy, dx in _DIRECTIONS], axis=0))
    return np.concatenate([colour, *lbp, *gradient, *cooccurrence])


def features(rgb, leaf, lesion, valid=None, raw=None):
    """403 features: lesion ROI (color 102, texture 26, shape 7) + whole leaf 268.

    `rgb` is the preprocessed image; `raw` is the original image used for leaf
    texture (defaults to `rgb`). Lesion ROI below 1% of leaf gives zeros for the
    lesion block only; the leaf block is always computed.
    """
    if not leaf.any():
        raise ValueError('Nonempty leaf required')
    result = np.r_[lesion_features(rgb, leaf, lesion, valid),
                   leaf_features(rgb, rgb if raw is None else raw, leaf)]
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError('Invalid feature vector')
    return result


def lesion_features(rgb, leaf, lesion, valid=None):
    """135 lesion-ROI features. No image-quadrant features; tiny ROI returns zeros."""
    roi = lesion & leaf
    if valid is not None:
        roi &= valid
    if roi.sum()/leaf.sum() < .01:
        return np.zeros(LESION_DIM)
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
    lookup = _UNIFORM
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
    return np.r_[color, texture, shape]
