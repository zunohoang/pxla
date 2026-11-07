"""Behavioral tests for numerical invariants, leakage, masks and persistence."""
import csv

import numpy as np
import pytest
from PIL import Image

from src.algorithms import components, fill_holes, kmeans, lbp8
from src.data import FIELDS, read_manifest
from src.evaluation import classification, overlap
from src.model import KNN
from src.pipeline import Config, process
from src.features import FEATURE_DIM


def test_components_and_holes():
    mask = np.zeros((9, 9), bool)
    mask[2:7, 2:7] = True
    mask[4, 4] = False
    filled = fill_holes(mask)
    assert filled.sum() == 25
    assert components(filled)[1].tolist() == [0, 25]


def test_kmeans_degenerate_and_objective():
    x = np.ones((10, 2))
    labels, centers, sse = kmeans(x, 3)
    assert np.isfinite(centers).all() and sse == 0
    x = np.array([[0., 0], [0, 2], [8, 8], [8, 10]])
    _, centers, sse = kmeans(x, 2, init=np.array([[0, 0], [8, 8]]))
    np.testing.assert_allclose(centers, [[0, 1], [8, 9]])
    assert sse == 4
    # Actual 4x4 hand-worked image example required by the course guide.
    small = np.array([[0, 0, 8, 8], [0, 2, 8, 10],
                      [0, 2, 8, 10], [2, 2, 10, 10]], float)
    _, centers, sse = kmeans(small.reshape(-1, 1), 2, init=np.array([[0.], [8.]]))
    np.testing.assert_allclose(centers[:, 0], [1, 9])
    assert sse == 16


def test_square_lbp_hand():
    x = np.array([[6, 4, 8], [2, 5, 7], [9, 5, 1]], np.uint8)
    assert lbp8(x)[1, 1] == 107


def test_metrics_empty_and_fixed_classes():
    empty = np.zeros((3, 3), bool)
    assert overlap(empty, empty)['iou'] == 1
    assert overlap(empty, ~empty)['iou'] == 0
    assert classification(['a'], ['a'], ['a', 'b'])['macro_f1'] == .5
    scored = classification(['a', 'b'], ['a', '__failed__'], ['a', 'b'])
    assert scored['accuracy'] == .5 and scored['macro_f1'] == .5
    assert np.array(scored['confusion']).shape == (2, 3)


def test_pipeline_synthetic_not_real_disease():
    rgb = np.full((96, 96, 3), 240, np.uint8)
    rgb[12:84, 25:71] = [50, 150, 45]
    rgb[35:50, 40:55] = [155, 85, 25]
    out = process(rgb)
    assert len(out['features']) == FEATURE_DIM and np.isfinite(out['features']).all()
    assert 0 < out['ratio'] < 1
    assert not (out['lesion'] & ~out['leaf']).any()
    with pytest.raises(ValueError):
        process(np.zeros((32, 32, 3), np.uint8))


def test_model_roundtrip_and_train_only_scaling(tmp_path):
    x = np.array([[0., 1], [1, 1], [10, 1], [11, 1]])
    model = KNN(1, [0, 1]).fit(x, ['a', 'a', 'b', 'b'])
    np.testing.assert_allclose(model.mean, [5.5, 1])
    path = tmp_path / 'model.npz'
    model.save(path)
    restored, config, crop = KNN.load(path)
    assert restored.predict(x).tolist() == ['a', 'a', 'b', 'b']
    assert crop == 'rice' and config == Config()


def test_group_leakage_rejected(tmp_path):
    rows = []
    for i, split in enumerate(['train', 'test']):
        image = tmp_path / f'{i}.png'
        Image.new('RGB', (12, 12), (i, 10, 20)).save(image)
        rows.append(dict(id=str(i), path=str(image), crop='rice', label='a',
                         domain='field', group='same_plant', split=split,
                         leaf_mask='', lesion_mask='', source='self', license='consent'))
    path = tmp_path / 'manifest.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='crosses splits'):
        read_manifest(path)
