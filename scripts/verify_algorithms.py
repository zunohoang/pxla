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
from src.algorithms import (kernel_ridge_fit, rbf_kernel, lda_projection, clahe, gray_world, lab_to_rgb, region_shape,
                            distance_transform, otsu, components, fill_holes, median_filter)
from scipy.ndimage import distance_transform_edt, binary_fill_holes, label
from skimage.measure import regionprops


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
    light = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    record('CLAHE OpenCV same 8x8 tiles clip2', clahe(light),
           cv2.createCLAHE(clipLimit=2., tileGridSize=(8,8)).apply(light), 1.)
    unequal = rng.integers(0,256,(64,70),dtype=np.uint8)
    record('CLAHE padded rectangle', clahe(unequal),
           cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(unequal), 1.)
    record('Lab inverse roundtrip', lab_to_rgb(rgb_to_lab(rgb)), rgb, 1.)
    record('Median RGB reflect', median_filter(rgb),
           cv2.medianBlur(np.pad(rgb, ((1,1),(1,1),(0,0)), mode='reflect'), 3)[1:-1,1:-1], 0.)
    record('Gray world hand', gray_world(np.full((8,8,3), [60,100,140], np.uint8)),
           np.full((8,8,3), 100), 0.)
    for dy, dx, angle in [(1,0,np.pi/2), (-1,1,-np.pi/4)]:
        reference = graycomatrix(quant.astype('uint8'), [1], [angle], levels=16,
                                symmetric=True, normed=True)[:,:,0,0]
        record(f'GLCM signed {dy},{dx}', glcm(gray,np.ones(gray.shape,bool),dy,dx), reference, 1e-14)
    shape = np.zeros((24,24), bool)
    shape[3:18,5:9] = True
    shape[14:18,5:20] = True
    ours_ecc, ours_solidity = region_shape(shape)
    props = regionprops(shape.astype(int))[0]
    record('Region eccentricity solidity', [ours_ecc,ours_solidity],
           [props.eccentricity,props.solidity], 1e-10)
    padded = np.pad(shape,1)
    record('Euclidean distance transform', distance_transform(shape),
           distance_transform_edt(padded)[1:-1,1:-1], 1e-10)
    holes = shape.copy()
    holes[5:8,6:8] = False
    record('Fill holes', fill_holes(holes), binary_fill_holes(holes), 0)
    record('Connected components count', [len(components(shape)[1])-1], [label(shape)[1]], 0)
    from skimage.filters import threshold_otsu
    record('Otsu uint8', [otsu(light)], [threshold_otsu(light)], 0)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    data = np.r_[rng.normal(0, 1, (30, 6)), rng.normal(1.5, 1, (30, 6)), rng.normal([3, 0, 0, 0, 0, 0], 1, (30, 6))]
    classes = np.repeat(['a', 'b', 'c'], 30)
    # Axes are defined up to sign and translation, so compare pairwise distances.
    pair = lambda z: np.sqrt(((z[:, None]-z[None])**2).sum(-1))
    reference = LinearDiscriminantAnalysis(solver='eigen').fit(data, classes).transform(data)
    record('LDA no shrinkage vs sklearn eigen', pair(data@lda_projection(data, classes, 1e-12)),
           pair(reference), 1e-6)
    # sklearn's eigen solver also shrinks the total scatter, which is not the
    # textbook form used here; so the shrunk case is checked against SciPy's
    # generalized eigensolver fed with scikit-learn covariance estimates.
    from scipy.linalg import eigh
    from sklearn.covariance import empirical_covariance, shrunk_covariance
    within = sum(np.mean(classes == c)*shrunk_covariance(empirical_covariance(data[classes == c]), .1)
                 for c in np.unique(classes))
    between = empirical_covariance(data)-sum(np.mean(classes == c)*empirical_covariance(data[classes == c])
                                             for c in np.unique(classes))
    _, vectors = eigh(between, within)
    record('LDA shrinkage 0.1 vs SciPy generalized eigh', pair(data@lda_projection(data, classes, .1)),
           pair(data@vectors[:, ::-1][:, :2]), 1e-8)
    from sklearn.kernel_ridge import KernelRidge as SkKernelRidge
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    queries = rng.normal(1, 1.5, (12, 6))
    record('RBF kernel', rbf_kernel(queries, data, .2), sk_rbf(queries, data, gamma=.2), 1e-12)
    targets = (classes[:, None] == np.unique(classes))*2.-1
    reference = SkKernelRidge(alpha=.1, kernel='rbf', gamma=.2).fit(data, targets).predict(queries)
    ours = rbf_kernel(queries, data, .2)@kernel_ridge_fit(rbf_kernel(data, data, .2), targets, .1)
    record('Kernel ridge decision values', ours, reference, 1e-8)
    return comparisons


if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)
    result = verify()
    Path('results/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
