"""Numerical comparison against independent library implementations."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops
from src.algorithms import (glcm, glcm_properties, kmeans, lbp8, morphology,
                            rgb_to_hsv, rgb_to_lab)
from src.model import KNN


def verify():
    """Report max error/MSE; assert tolerances for matching conventions."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    gray = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    mask = gray > 120
    comparisons = []

    def record(name, a, b, tolerance):
        diff = np.asarray(a, float)-np.asarray(b, float)
        row = dict(name=name, max_abs=float(abs(diff).max()), mse=float((diff**2).mean()),
                   tolerance=tolerance)
        row['passed'] = row['max_abs'] <= tolerance
        comparisons.append(row)
        assert row['passed'], row

    hsv = rgb2hsv(rgb) * np.array([360, 1, 1])
    record('HSV', rgb_to_hsv(rgb), hsv, 1e-10)
    record('Lab D65', rgb_to_lab(rgb), rgb2lab(rgb), .03)
    for op, cvop in [('erode', cv2.erode), ('dilate', cv2.dilate)]:
        ref = cvop(mask.astype('uint8'), np.ones((3, 3), 'uint8'),
                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
        record(op, morphology(mask, op), ref, 0)
    quant = gray.astype('int32') * 16 // 256
    ref = graycomatrix(quant.astype('uint8'), [1], [0], levels=16,
                      symmetric=True, normed=True)[:, :, 0, 0]
    ours = glcm(gray, np.ones(gray.shape, bool))
    record('GLCM', ours, ref, 1e-14)
    ref_props = [graycoprops(ref[:, :, None, None], p)[0, 0] for p in
                 ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']]
    record('GLCM properties', glcm_properties(ours), ref_props, 1e-10)
    x = np.r_[rng.normal(0, .1, (40, 2)), rng.normal(4, .1, (40, 2))]
    init = np.array([[0, 0], [4, 4.]])
    a, centers, sse = kmeans(x, 2, init=init)
    ref = KMeans(2, init=init, n_init=1, tol=1e-10, algorithm='lloyd').fit(x)
    record('K-means centers same init', centers, ref.cluster_centers_, 1e-10)
    record('K-means SSE', [sse], [ref.inertia_], 1e-10)
    y = np.array(['a']*40+['b']*40)
    own = KNN(3, [0, 1]).fit(x, y)
    z = rng.normal(1, 1, (10, 2))
    sk = KNeighborsClassifier(3).fit(own.x, y)
    record('k-NN predictions', own.predict(z) == 'a',
           sk.predict((z-own.mean)/own.std) == 'a', 0)
    # Hand example: center=5; E, NE, NW, SW, S set => decimal 107.
    small = np.array([[6, 4, 8], [2, 5, 7], [9, 5, 1]], 'uint8')
    expected = sum((int(v) >= 5) << i for i, v in enumerate([7, 8, 4, 6, 2, 9, 5, 1]))
    record('LBP hand example', [lbp8(small)[1, 1]], [expected], 0)
    return comparisons


if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)
    result = verify()
    Path('results/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
