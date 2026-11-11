"""Classical DIP pipeline v2 with preprocessing and configurable segmentation."""
from dataclasses import asdict, dataclass
import numpy as np
from PIL import Image, ImageOps
from .algorithms import (clahe, cleanup, components, fill_holes, gray_world,
                         kmeans, lab_to_rgb, median_filter, morphology, otsu,
                         rgb_to_hsv, rgb_to_lab)
from .features import FEATURE_DIM, FEATURE_GROUPS, FEATURE_NAMES, features
from .advice import severity


@dataclass(frozen=True)
class Config:
    """Engineering defaults frozen before test, not tuned against test images."""
    max_side: int = 256
    clusters: int = 3
    seed: int = 42
    exg: float = .04
    saturation: float = .15
    lesion_a: float = -3.
    lesion_b: float = 8.
    dark_l: float = 30.
    preprocessing: bool = True
    white_balance: bool = True
    clahe_clip: float = 2.
    clahe_tiles: int = 8
    median: bool = False
    exclude_glare: bool = True
    leaf_method: str = 'hybrid'
    morph_height: int = 3
    morph_width: int = 3
    lesion_method: str = 'ab_distance'
    lesion_min_distance: float = 8.
    min_lesion_pixels: int = 3
    lesion_close: bool = False

    def to_dict(self):
        """Return JSON-safe pipeline settings."""
        return asdict(self)


def load_rgb(source, max_side=256):
    """Decode RGB with EXIF correction; resize preserving aspect ratio."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if min(image.size) < 8:
            raise ValueError('Ảnh quá nhỏ; cần ít nhất 8 pixel mỗi chiều.')
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.array(image)


def preprocess(rgb, config):
    """Gray-world, CLAHE on L*, optional median; detect glare on raw RGB."""
    hsv = rgb_to_hsv(rgb)
    glare = (hsv[..., 2] >= .98) & (hsv[..., 1] <= .10)
    result = rgb.copy()
    if config.preprocessing:
        if config.white_balance:
            result = gray_world(result)
        lab = rgb_to_lab(result)
        light = np.clip(np.round(lab[..., 0]*255/100), 0, 255).astype(np.uint8)
        lab[..., 0] = clahe(light, (config.clahe_tiles, config.clahe_tiles),
                           config.clahe_clip).astype(float)*100/255
        result = lab_to_rgb(lab)
        if config.median:
            result = median_filter(result)
    return result, glare


def leaf_mask(rgb, config):
    """HSV/ExG variants; edge padding preserves leaves crossing image edges."""
    hsv = rgb_to_hsv(rgb)
    x = rgb.astype(float)/255
    norm = x/np.maximum(x.sum(-1, keepdims=True), 1e-8)
    exg = 2*norm[..., 1]-norm[..., 0]-norm[..., 2]
    seed = (exg > config.exg) & (hsv[..., 1] > config.saturation)
    color = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 170)
             & (hsv[..., 1] > config.saturation) & (hsv[..., 2] > .06))
    if config.leaf_method in ['hybrid', 'hsv']:
        candidate = color
    elif config.leaf_method == 'exg_otsu':
        candidate = exg > otsu(exg)
    elif config.leaf_method == 'exg_exr':
        index = exg-(1.4*norm[..., 0]-norm[..., 1])
        candidate = index > otsu(index)
    else:
        raise ValueError('Unknown NumPy leaf method: '+config.leaf_method)
    shape = (config.morph_height, config.morph_width)
    pad = max(shape)+1
    cleaned = cleanup(np.pad(candidate, pad, mode='edge'), shape)[pad:-pad, pad:-pad]
    labels, areas = components(cleaned)
    if len(areas) == 1:
        raise ValueError('Không tìm thấy lá; chụp một lá trên nền tương phản.')
    scores = np.bincount(labels[seed], minlength=len(areas)).astype(float)
    if config.leaf_method != 'hybrid' or scores[1:].max() == 0:
        scores = areas.astype(float)
    scores[0] = 0
    mask = fill_holes(labels == scores.argmax())
    if mask.sum() < 40:
        raise ValueError('Vùng lá quá nhỏ; hãy chụp gần hơn.')
    return mask


def lesion_mask(rgb, leaf, config, valid=None):
    """Lab baseline or a*b* clustering relative to the largest color cluster.

    Largest-cluster healthy tissue is an assumption. Minimum distance prevents
    forced lesions on uniform leaves. Centers retain L*a*b* columns for the UI.
    """
    lab = rgb_to_lab(rgb)
    roi = leaf if valid is None else leaf & valid
    result = np.zeros(leaf.shape, bool)
    if roi.sum() < config.clusters:
        return result, np.empty((0, 3))
    x = lab[roi]
    rng = np.random.default_rng(config.seed)
    sample = x[rng.choice(len(x), min(len(x), 8000), replace=False)]
    fit = sample if config.lesion_method == 'lab_rule' else sample[:, 1:]
    if config.lesion_method == 'otsu_a':
        result[roi] = x[:, 1] > otsu(x[:, 1])
        centers = np.array([x.mean(0)])
    elif config.lesion_method in ['lab_rule', 'ab_distance']:
        _, fitted, _ = kmeans(fit, min(config.clusters, len(fit)), config.seed)
        all_x = x if config.lesion_method == 'lab_rule' else x[:, 1:]
        ids = ((all_x[:, None]-fitted)**2).sum(-1).argmin(-1)
        centers = np.array([x[ids == j].mean(0) if np.any(ids == j) else x.mean(0)
                            for j in range(len(fitted))])
        if config.lesion_method == 'lab_rule':
            bad = ((centers[:, 1] > config.lesion_a) & (centers[:, 2] > config.lesion_b)) | (centers[:, 0] < config.dark_l)
        else:
            healthy = np.bincount(ids, minlength=len(fitted)).argmax()
            distances = np.sqrt(((fitted-fitted[healthy])**2).sum(1))
            bad = np.zeros(len(fitted), bool)
            farthest = distances.argmax()
            if distances[farthest] >= config.lesion_min_distance:
                bad[farthest] = True
        result[roi] = bad[ids]
    else:
        raise ValueError('Unknown lesion method: '+config.lesion_method)
    if config.lesion_close:
        result = morphology(morphology(result, 'dilate'), 'erode') & roi
    labels, areas = components(result)
    keep = areas >= config.min_lesion_pixels
    keep[0] = False
    return keep[labels] & roi, centers


def process(rgb, config=Config(), leaf_override=None):
    """Stage images, masks, versioned ROI features, ratio and quality flags."""
    processed, glare = preprocess(rgb, config)
    leaf = leaf_mask(rgb, config) if leaf_override is None else np.asarray(leaf_override, bool)
    if leaf.shape != rgb.shape[:2] or not leaf.any():
        raise ValueError('Invalid leaf mask')
    valid = ~glare if config.exclude_glare else np.ones(leaf.shape, bool)
    lesion, centers = lesion_mask(processed, leaf, config, valid)
    ratio = float(lesion.sum()/leaf.sum())
    level = severity(ratio)['label']
    flags = []
    if leaf.mean() < .03 or leaf.mean() > .90:
        flags.append('Diện tích lá bất thường; kiểm tra lại mặt nạ.')
    if leaf[0].any() or leaf[-1].any() or leaf[:, 0].any() or leaf[:, -1].any():
        flags.append('Lá chạm biên ảnh; tỉ lệ chỉ phản ánh phần lá nhìn thấy.')
    glare_fraction = float((glare & leaf).sum()/leaf.sum())
    if glare_fraction > .03:
        flags.append('Có vùng chói trên lá; phần này không được dùng nhận dạng tổn thương.')
    if rgb.std() < 8:
        flags.append('Ảnh ít tương phản; nên chụp lại.')
    overlay = rgb.copy()
    overlay[lesion] = (.5*overlay[lesion]+.5*np.array([255, 40, 40])).astype(np.uint8)
    return dict(leaf=leaf, lesion=lesion, features=features(processed, leaf, lesion, valid, rgb),
                ratio=ratio, severity=level, warnings=flags, centers=centers,
                overlay=overlay, processed=processed, glare=glare,
                glare_fraction=glare_fraction)
