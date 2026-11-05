"""Leaf masks, Lab lesion candidates, features, visual overlays."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .algorithms import (cleanup, components, fill_holes, glcm, glcm_properties,
                         kmeans, lbp8, morphology, rgb_to_hsv, rgb_to_lab)


@dataclass(frozen=True)
class Config:
    """Frozen algorithm settings stored with each classifier."""
    max_side: int = 256
    clusters: int = 3
    seed: int = 42
    exg: float = .04
    saturation: float = .15
    lesion_a: float = -3.
    lesion_b: float = 8.
    dark_l: float = 30.

    def to_dict(self):
        """Return JSON-safe settings."""
        return asdict(self)


def load_rgb(source, max_side=256):
    """Decode file/stream, apply EXIF orientation, limit to max_side pixels."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if min(image.size) < 8:
            raise ValueError('Ảnh quá nhỏ; cần ít nhất 8 pixel mỗi chiều.')
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.array(image)


def leaf_mask(rgb, config):
    """HSV plant-color candidates joined to ExG green seeds; keep one component.

    Brown/yellow pixels may join seeds, preserving symptomatic tissue. The
    assumption is one isolated leaf; grass and overlapping leaves violate it.
    """
    hsv = rgb_to_hsv(rgb)
    x = rgb.astype(float) / 255
    norm = x / np.maximum(x.sum(-1, keepdims=True), 1e-8)
    exg = 2 * norm[..., 1] - norm[..., 0] - norm[..., 2]
    seed = (exg > config.exg) & (hsv[..., 1] > config.saturation)
    candidate = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 170)
                 & (hsv[..., 1] > config.saturation) & (hsv[..., 2] > .06))
    labels, areas = components(cleanup(candidate))
    if len(areas) == 1:
        raise ValueError('Không tìm thấy lá; chụp một lá trên nền tương phản.')
    scores = np.bincount(labels[seed], minlength=len(areas)).astype(float)
    scores[0] = 0
    if scores.max() == 0:
        scores = areas.astype(float)
        scores[0] = 0
    mask = fill_holes(labels == scores.argmax())
    if mask.sum() < 40:
        raise ValueError('Vùng lá quá nhỏ; hãy chụp gần hơn.')
    return mask


def lesion_mask(rgb, leaf, config):
    """Cluster only leaf pixels in Lab; flag brown/yellow or dark centers.

    Cluster IDs have no disease meaning. Center color rules are an explicit
    heuristic frozen before test evaluation, not ground-truth lesion labels.
    """
    lab = rgb_to_lab(rgb)
    x = lab[leaf]
    rng = np.random.default_rng(config.seed)
    sample = x[rng.choice(len(x), min(len(x), 8000), replace=False)]
    _, centers, _ = kmeans(sample, min(config.clusters, len(sample)), config.seed)
    ids = ((x[:, None] - centers)**2).sum(-1).argmin(-1)
    bad = (((centers[:, 1] > config.lesion_a) & (centers[:, 2] > config.lesion_b))
           | (centers[:, 0] < config.dark_l))
    result = np.zeros(leaf.shape, bool)
    result[leaf] = bad[ids]
    # Do not open lesion mask: tiny spots are clinically relevant candidates.
    return result & leaf, centers


def features(rgb, leaf):
    """74 dimensions: 48 HSV bins, 6 Lab moments, 6 GLCM, 14 uniform-square LBP.

    LBP uses 10 bins (0..8 ones, 9 nonuniform), plus 4 quadrant means of gray.
    Quadrant means describe spatial color intensity; all values are leaf-only.
    """
    hsv, lab = rgb_to_hsv(rgb), rgb_to_lab(rgb)
    hist = []
    for channel, high in [(0, 360), (1, 1), (2, 1)]:
        h, _ = np.histogram(hsv[..., channel][leaf], bins=16, range=(0, high))
        hist.extend(h / max(h.sum(), 1))
    moments = np.r_[lab[leaf].mean(0), lab[leaf].std(0)]
    gray = np.round(rgb @ np.array([.299, .587, .114])).astype(np.uint8)
    texture = np.mean([glcm_properties(glcm(gray, leaf, dy, dx))
                       for dy, dx in [(0, 1), (1, 0), (1, 1)]], axis=0)
    codes = lbp8(gray)
    valid = morphology(leaf, 'erode')
    vals = np.arange(256, dtype=np.uint16)
    bits = ((vals[:, None] >> np.arange(8)) & 1)
    transitions = (bits != np.roll(bits, 1, axis=1)).sum(1)
    table = np.where(transitions <= 2, bits.sum(1), 9)
    lbp_hist = np.bincount(table[codes[valid]], minlength=10).astype(float)
    lbp_hist /= max(lbp_hist.sum(), 1)
    hm, wm = gray.shape[0] // 2, gray.shape[1] // 2
    quadrants = []
    for ys, xs in [(slice(0, hm), slice(0, wm)), (slice(0, hm), slice(wm, None)),
                   (slice(hm, None), slice(0, wm)), (slice(hm, None), slice(wm, None))]:
        q = gray[ys, xs][leaf[ys, xs]]
        quadrants.append(q.mean() / 255 if len(q) else 0.)
    return np.r_[hist, moments, texture, lbp_hist, quadrants]


def process(rgb, config=Config()):
    """Return leaf/lesion masks, 74 features, projected area fraction and flags."""
    leaf = leaf_mask(rgb, config)
    lesion, centers = lesion_mask(rgb, leaf, config)
    ratio = float(lesion.sum() / leaf.sum())
    level = 'Chưa thấy vùng nghi tổn thương' if ratio < .01 else (
        'Nhẹ' if ratio < .05 else 'Trung bình' if ratio < .20 else 'Nặng')
    flags = []
    fraction = leaf.mean()
    if fraction < .03 or fraction > .90:
        flags.append('Diện tích lá bất thường; kiểm tra lại mặt nạ.')
    if leaf[0].any() or leaf[-1].any() or leaf[:, 0].any() or leaf[:, -1].any():
        flags.append('Lá chạm biên ảnh; tỉ lệ chỉ phản ánh phần lá nhìn thấy.')
    if rgb.std() < 8:
        flags.append('Ảnh ít tương phản; nên chụp lại.')
    overlay = rgb.copy()
    overlay[lesion] = (.5*overlay[lesion] + .5*np.array([255, 40, 40])).astype('uint8')
    return dict(leaf=leaf, lesion=lesion, features=features(rgb, leaf),
                ratio=ratio, severity=level, warnings=flags, centers=centers,
                overlay=overlay)
