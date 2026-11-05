"""NumPy implementations. Equations and conventions are documented in docs/.

No OpenCV/scikit-image processing calls are used in this module.
"""
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rgb_to_hsv(rgb):
    """Convert uint8 RGB to H in degrees, S/V in [0, 1] (Eq. 2.2)."""
    x = np.asarray(rgb, dtype=float) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    safe = np.where(delta > 0, delta, 1)
    r, g, b = np.moveaxis(x, -1, 0)
    h = np.select(
        [mx == r, mx == g],
        [((g - b) / safe) % 6, (b - r) / safe + 2],
        default=(r - g) / safe + 4,
    ) * 60
    h[delta == 0] = 0
    return np.stack([h, delta / np.where(mx > 0, mx, 1), mx], -1)


def rgb_to_lab(rgb):
    """sRGB D65 to CIE L*a*b*, float64, using IEC sRGB transfer (Eq. 2.2)."""
    x = np.asarray(rgb, dtype=float) / 255
    linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    xyz = (linear @ matrix.T) / np.array([.95047, 1., 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def morphology(mask, operation, size=3):
    """Square binary erosion/dilation with constant False border (Eq. 2.4)."""
    if size < 1 or size % 2 != 1 or operation not in ('erode', 'dilate'):
        raise ValueError('Odd positive size and erode/dilate required')
    windows = sliding_window_view(np.pad(mask.astype(bool), size // 2), (size, size))
    return windows.all((-2, -1)) if operation == 'erode' else windows.any((-2, -1))


def cleanup(mask, size=3):
    """Opening then closing, preserving a fixed zero-border convention."""
    x = morphology(morphology(mask, 'erode', size), 'dilate', size)
    return morphology(morphology(x, 'dilate', size), 'erode', size)


def components(mask):
    """Return 4-connected integer labels and areas by flood fill, O(HW)."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    areas = [0]
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        label = len(areas)
        labels[y, x] = label
        queue = deque([(y, x)])
        area = 0
        while queue:
            i, j = queue.popleft()
            area += 1
            for a, b in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not labels[a, b]:
                    labels[a, b] = label
                    queue.append((a, b))
        areas.append(area)
    return labels, np.array(areas)


def fill_holes(mask):
    """Fill non-border-connected background under 4-connectivity."""
    labels, _ = components(~mask)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return mask | ~np.isin(labels, np.r_[0, border])


def kmeans(x, k=3, seed=42, max_iter=60, init=None):
    """Lloyd K-means with k-means++ seeds; returns labels, centers, SSE (Eq. 2.5).

    Empty clusters keep their previous center. Duplicate samples are supported.
    """
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < k or k < 1:
        raise ValueError('Finite NxD input with N >= K >= 1 required')
    rng = np.random.default_rng(seed)
    if init is None:
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d = ((x[:, None] - np.array(centers)) ** 2).sum(-1).min(-1)
            idx = rng.choice(len(x), p=d / d.sum()) if d.sum() > 0 else 0
            centers.append(x[idx])
        centers = np.array(centers)
    else:
        centers = np.array(init, float).copy()
        if centers.shape != (k, x.shape[1]):
            raise ValueError('Initial centers shape mismatch')
    for _ in range(max_iter):
        labels = ((x[:, None] - centers) ** 2).sum(-1).argmin(-1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j)
                        else centers[j] for j in range(k)])
        if np.max(np.abs(new - centers)) < 1e-6:
            centers = new
            break
        centers = new
    labels = ((x[:, None] - centers) ** 2).sum(-1).argmin(-1)
    return labels, centers, float(((x - centers[labels]) ** 2).sum())


def lbp8(gray):
    """3x3 square LBP, E NE N NW W SW S SE, >= center (Eq. 2.7).

    This is a square-neighbor variant, not circular interpolated skimage LBP.
    Border codes are zero; extraction excludes any neighborhood outside leaf.
    """
    h, w = gray.shape
    result = np.zeros((h, w), np.uint8)
    for bit, (dy, dx) in enumerate([(0, 1), (-1, 1), (-1, 0), (-1, -1),
                                    (0, -1), (1, -1), (1, 0), (1, 1)]):
        result[1:-1, 1:-1] |= ((gray[1+dy:h-1+dy, 1+dx:w-1+dx]
                               >= gray[1:-1, 1:-1]).astype(np.uint8) << bit)
    return result


def glcm(gray, mask, dy=0, dx=1, levels=16):
    """Masked symmetric normalized co-occurrence, quantized uint8 (Eq. 2.8)."""
    if dy < 0 or dx < 0 or dy >= gray.shape[0] or dx >= gray.shape[1]:
        raise ValueError('Nonnegative offsets within image required')
    q = np.minimum(gray.astype(np.int32) * levels // 256, levels - 1)
    h, w = gray.shape
    a, b = q[:h-dy, :w-dx], q[dy:, dx:]
    valid = mask[:h-dy, :w-dx] & mask[dy:, dx:]
    counts = np.bincount((a[valid] * levels + b[valid]), minlength=levels**2)
    p = counts.reshape(levels, levels).astype(float)
    p += p.T.copy()
    return p / p.sum() if p.sum() else p


def glcm_properties(p):
    """Contrast, dissimilarity, homogeneity, ASM, energy, correlation (Eq. 2.9)."""
    i, j = np.indices(p.shape)
    mi, mj = (i*p).sum(), (j*p).sum()
    vi, vj = ((i-mi)**2*p).sum(), ((j-mj)**2*p).sum()
    corr = ((i-mi)*(j-mj)*p).sum() / np.sqrt(vi*vj) if vi*vj > 1e-15 else 1.
    asm = (p*p).sum()
    return np.array([((i-j)**2*p).sum(), (abs(i-j)*p).sum(),
                     (p/(1+(i-j)**2)).sum(), asm, np.sqrt(asm), corr])
