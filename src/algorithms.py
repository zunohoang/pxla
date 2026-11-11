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
    shape = (size, size) if np.isscalar(size) else tuple(size)
    if len(shape) != 2 or any(s < 1 or s % 2 != 1 for s in shape) or operation not in ('erode', 'dilate'):
        raise ValueError('Odd positive size and erode/dilate required')
    windows = sliding_window_view(np.pad(mask.astype(bool),
                                  [(s//2, s//2) for s in shape]), shape)
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


def lbp8(gray, radius=1):
    """3x3 square LBP, E NE N NW W SW S SE, >= center (Eq. 2.7).

    This is a square-neighbor variant, not circular interpolated skimage LBP.
    Border codes are zero; extraction excludes any neighborhood outside leaf.
    """
    h, w = gray.shape
    if radius < 1 or 2*radius >= min(h, w):
        raise ValueError('LBP radius must fit image')
    result = np.zeros((h, w), np.uint8)
    for bit, (dy, dx) in enumerate([(0, 1), (-1, 1), (-1, 0), (-1, -1),
                                    (0, -1), (1, -1), (1, 0), (1, 1)]):
        dy, dx = dy*radius, dx*radius
        result[radius:-radius, radius:-radius] |= ((
            gray[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
            >= gray[radius:-radius, radius:-radius]).astype(np.uint8) << bit)
    return result


def glcm(gray, mask, dy=0, dx=1, levels=16):
    """Masked symmetric normalized co-occurrence, quantized uint8 (Eq. 2.8)."""
    if abs(dy) >= gray.shape[0] or abs(dx) >= gray.shape[1] or not 2 <= levels <= 256:
        raise ValueError('Offsets must fit image; levels in [2,256]')
    q = np.minimum(gray.astype(np.int32) * levels // 256, levels - 1)
    h, w = gray.shape
    y0, y1 = max(0, -dy), min(h, h-dy)
    x0, x1 = max(0, -dx), min(w, w-dx)
    a, b = q[y0:y1, x0:x1], q[y0+dy:y1+dy, x0+dx:x1+dx]
    valid = mask[y0:y1, x0:x1] & mask[y0+dy:y1+dy, x0+dx:x1+dx]
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


def gray_world(rgb):
    """Gray-world white balance with bounded gains; preserves mean brightness."""
    x = np.asarray(rgb, float)
    means = x.mean((0, 1))
    gains = np.clip(means.mean()/np.maximum(means, 1e-8), .5, 2.)
    return np.clip(np.round(x*gains), 0, 255).astype(np.uint8)


def lab_to_rgb(lab):
    """Invert CIE Lab D65 to clipped 8-bit sRGB (same constants as rgb_to_lab)."""
    fy = (lab[..., 0]+16)/116
    f = np.stack([fy+lab[..., 1]/500, fy, fy-lab[..., 2]/200], -1)
    d = 6/29
    xyz = np.where(f > d, f**3, 3*d*d*(f-4/29))*[.95047, 1., 1.08883]
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    linear = np.maximum(xyz @ np.linalg.inv(matrix).T, 0)
    srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return np.clip(np.round(srgb*255), 0, 255).astype(np.uint8)


def clahe(channel, tiles=(8, 8), clip_limit=2.):
    """8-bit CLAHE with reflect padding, clipped histograms and bilinear LUTs.

    Matches OpenCV's tile-center interpolation and residual redistribution.
    clip_limit is histogram multiples of tile_area/256, not skimage's fraction.
    """
    channel = np.asarray(channel, np.uint8)
    ty, tx = tiles
    if min(ty, tx) < 1 or clip_limit <= 0:
        raise ValueError('Positive tile grid and clip limit required')
    h, w = channel.shape
    # OpenCV pads BOTH axes in the non-divisible case, even if one axis fits.
    py, px = (0, 0) if h % ty == 0 and w % tx == 0 else (ty-h % ty, tx-w % tx)
    extended = np.pad(channel, ((0, py), (0, px)), mode='reflect')
    th, tw = extended.shape[0]//ty, extended.shape[1]//tx
    area = th*tw
    limit = max(1, int(clip_limit*area/256))
    luts = np.empty((ty, tx, 256), np.uint8)
    for iy in range(ty):
        for ix in range(tx):
            hist = np.bincount(extended[iy*th:(iy+1)*th, ix*tw:(ix+1)*tw].ravel(), minlength=256)
            excess = int(np.maximum(hist-limit, 0).sum())
            hist = np.minimum(hist, limit)
            hist += excess//256
            residual = excess % 256
            if residual:
                step = max(256//residual, 1)
                hist[np.arange(0, 256, step)[:residual]] += 1
            luts[iy, ix] = np.clip(np.rint(hist.cumsum()*255/area), 0, 255).astype(np.uint8)
    y = np.arange(h)/th-.5
    x = np.arange(w)/tw-.5
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ay, ax = y-y0, x-x0
    y1, x1 = np.clip(y0+1, 0, ty-1), np.clip(x0+1, 0, tx-1)
    y0, x0 = np.clip(y0, 0, ty-1), np.clip(x0, 0, tx-1)
    top = ((1-ax)*luts[y0[:, None], x0, channel]+ax*luts[y0[:, None], x1, channel])
    bottom = ((1-ax)*luts[y1[:, None], x0, channel]+ax*luts[y1[:, None], x1, channel])
    return np.clip(np.rint((1-ay[:, None])*top+ay[:, None]*bottom), 0, 255).astype(np.uint8)


def median_filter(rgb, size=3):
    """Reflect-padded RGB median filter, NumPy implementation."""
    padded = np.pad(rgb, ((size//2, size//2), (size//2, size//2), (0, 0)), mode='reflect')
    return np.median(sliding_window_view(padded, (size, size), axis=(0, 1)),
                     axis=(-1, -2)).astype(np.uint8)


def otsu(values):
    """Otsu threshold in original value units using 256 bins."""
    values = np.asarray(values, float)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo
    q = np.clip(np.rint((values-lo)*255/(hi-lo)), 0, 255).astype(np.uint8)
    hist = np.bincount(q.ravel(), minlength=256).astype(float)
    weight = hist.cumsum()
    moment = (hist*np.arange(256)).cumsum()
    den = weight*(weight[-1]-weight)
    objective = np.divide((moment[-1]*weight-moment*weight[-1])**2, den,
                          out=np.zeros(256), where=den > 0)
    return lo+int(objective.argmax())*(hi-lo)/255


def distance_transform(mask):
    """Exact Euclidean distance to background including the outside image, O(P).

    Separable lower envelopes of parabolas (Felzenszwalb and Huttenlocher).
    """
    padded = np.pad(np.asarray(mask, bool), 1)
    f = np.where(padded, 1e12, 0.)

    def transform_1d(a):
        n = len(a)
        v = np.zeros(n, int)
        z = np.full(n+1, np.inf)
        z[0] = -np.inf
        k = 0
        for q in range(1, n):
            s = ((a[q]+q*q)-(a[v[k]]+v[k]*v[k]))/(2*(q-v[k]))
            while s <= z[k]:
                k -= 1
                s = ((a[q]+q*q)-(a[v[k]]+v[k]*v[k]))/(2*(q-v[k]))
            k += 1
            v[k], z[k], z[k+1] = q, s, np.inf
        result = np.empty(n)
        k = 0
        for q in range(n):
            while z[k+1] < q:
                k += 1
            result[q] = (q-v[k])**2+a[v[k]]
        return result

    for y in range(f.shape[0]):
        f[y] = transform_1d(f[y])
    for x in range(f.shape[1]):
        f[:, x] = transform_1d(f[:, x])
    return np.sqrt(f[1:-1, 1:-1])


def convex_hull_mask(mask):
    """Raster convex hull using pixel diamond extents and monotone chain."""
    boundary = mask & ~morphology(mask, 'erode')
    coords = np.column_stack(np.nonzero(boundary)).astype(float)
    if not len(coords):
        return np.zeros_like(mask)
    points = np.concatenate([coords+offset for offset in [(-.5, 0), (.5, 0), (0, -.5), (0, .5)]])
    points = sorted(set(map(tuple, points)))

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1]+upper[:-1]
    yy, xx = np.indices(mask.shape)
    inside = np.ones(mask.shape, bool)
    for a, b in zip(hull, hull[1:]+hull[:1]):
        inside &= (b[0]-a[0])*(xx-a[1])-(b[1]-a[1])*(yy-a[0]) >= -1e-9
    return inside


def region_shape(mask):
    """Return eccentricity from second moments and pixel-hull solidity."""
    coords = np.column_stack(np.nonzero(mask)).astype(float)
    if not len(coords):
        return 0., 0.
    centered = coords-coords.mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(centered.T@centered/len(coords)), 0)
    ecc = np.sqrt(1-eigen[0]/eigen[1]) if eigen[1] > 0 else 0.
    solidity = len(coords)/max(convex_hull_mask(mask).sum(), 1)
    return float(ecc), float(solidity)


def lda_projection(x, y, shrinkage=.1):
    """Fisher discriminant axes with a shrunk within-class covariance.

    Sw = sum_c p_c Cov_c is shrunk to (1-s) Sw + s (tr Sw / d) I, then the
    generalized problem Sb v = lambda Sw v is solved by whitening Sw. Returns a
    d x (C-1) matrix whose columns satisfy v' Sw v = 1, largest lambda first.
    """
    x, y = np.asarray(x, float), np.asarray(y)
    classes = np.unique(y)
    if x.ndim != 2 or len(classes) < 2 or not 0 < shrinkage <= 1:
        raise ValueError('NxD input, >=2 classes and shrinkage in (0, 1] required')
    mean = x.mean(0)
    within = np.zeros((x.shape[1], x.shape[1]))
    between = np.zeros_like(within)
    for label in classes:
        rows = x[y == label]
        centered = rows-rows.mean(0)
        within += centered.T@centered
        between += len(rows)*np.outer(rows.mean(0)-mean, rows.mean(0)-mean)
    within, between = within/len(x), between/len(x)
    within = (1-shrinkage)*within+shrinkage*np.trace(within)/len(within)*np.eye(len(within))
    values, vectors = np.linalg.eigh(within)
    whitening = vectors/np.sqrt(values)
    _, axes = np.linalg.eigh(whitening.T@between@whitening)
    return whitening@axes[:, ::-1][:, :len(classes)-1]


def rbf_kernel(a, b, gamma):
    """Gaussian kernel matrix exp(-gamma ||a_i - b_j||^2) between row sets."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    squared = (a**2).sum(1)[:, None]+(b**2).sum(1)[None]-2*a@b.T
    return np.exp(-gamma*np.maximum(squared, 0))


def kernel_ridge_fit(kernel, targets, alpha):
    """Dual ridge coefficients: solve (K + alpha I) W = T for a symmetric kernel K."""
    kernel = np.asarray(kernel, float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or alpha <= 0:
        raise ValueError('Square kernel matrix and positive alpha required')
    return np.linalg.solve(kernel+alpha*np.eye(len(kernel)), np.asarray(targets, float))
