"""Regression coverage for v2 numerical methods, data safety and review findings."""
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from src.algorithms import (clahe, distance_transform, glcm, gray_world,
                            lab_to_rgb, lbp8, morphology, otsu, rgb_to_lab)
from src.advice import RULES, recommendation, severity
from src.data import FIELDS, merge_manifest
from src.features import FEATURE_DIM, FEATURE_GROUPS, features
from src.model import KNN, KernelRidge
from src.pipeline import Config, process
from run_experiments import evaluate_masks


def test_lab_roundtrip_and_gray_world():
    rgb = np.random.default_rng(7).integers(0, 256, (24, 24, 3), dtype=np.uint8)
    np.testing.assert_allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1)
    balanced = gray_world(np.full((16, 16, 3), [60, 100, 140], np.uint8))
    assert balanced.min() == balanced.max() == 100


def test_signed_glcm_mirror_and_rectangular_morphology():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)*4
    mask = np.ones((8, 8), bool)
    np.testing.assert_allclose(glcm(gray, mask, -1, 1), glcm(gray, mask, 1, -1))
    thin = np.zeros((15, 15), bool)
    thin[7, 2:13] = True
    assert morphology(thin, 'erode', (1, 7)).any()
    assert not morphology(thin, 'erode', (3, 3)).any()
    assert 0 <= otsu(np.array([0, 0, 200, 200], float)) < 200


def test_uniform_leaf_no_forced_lesion_and_border_warning():
    image = np.full((64, 64, 3), 240, np.uint8)
    image[:, 20:44] = [50, 150, 45]
    out = process(image, Config(preprocessing=False))
    assert any('chạm biên' in w for w in out['warnings'])
    assert out['ratio'] == 0
    lesion_block = out['features'][FEATURE_GROUPS['lesion']]
    np.testing.assert_array_equal(lesion_block, np.zeros(len(lesion_block)))
    assert out['features'][FEATURE_GROUPS['leaf']].any()


def test_roi_ignores_background_and_rotation_shape():
    image = np.full((64, 64, 3), [30, 150, 40], np.uint8)
    leaf = np.zeros((64, 64), bool)
    leaf[5:59, 12:52] = True
    lesion = np.zeros_like(leaf)
    lesion[20:36, 25:38] = True
    image[lesion] = [150, 80, 30]
    a = features(image, leaf, lesion)
    changed = image.copy()
    changed[~lesion] = [250, 250, 250]
    block = FEATURE_GROUPS['lesion']
    np.testing.assert_allclose(a[block], features(changed, leaf, lesion)[block])
    outside = image.copy()
    outside[~leaf] = [250, 250, 250]
    np.testing.assert_allclose(a, features(outside, leaf, lesion))
    b = features(np.rot90(image), np.rot90(leaf), np.rot90(lesion))
    np.testing.assert_allclose(a[FEATURE_GROUPS['shape']], b[FEATURE_GROUPS['shape']], atol=1e-10)


def mask_case(tmp_path, dark=False, empty_leaf=False):
    """Write one synthetic annotated case; never count it as real field data."""
    image = np.full((48, 48, 3), 240 if not dark else 3, np.uint8)
    leaf = np.zeros((48, 48), np.uint8)
    if not empty_leaf:
        leaf[5:43, 12:36] = 255
    if not dark:
        image[leaf > 0] = [30, 150, 40]
    lesion = np.zeros_like(leaf)
    for name, data in [('image', image), ('leaf', leaf), ('lesion', lesion)]:
        Image.fromarray(data).save(tmp_path/f'{name}.png')
    return dict(id='synthetic', path=str(tmp_path/'image.png'), split='segtest',
                leaf_mask=str(tmp_path/'leaf.png'), lesion_mask=str(tmp_path/'lesion.png'))


def test_segmentation_failure_counted_and_empty_gt_rejected(tmp_path):
    row = mask_case(tmp_path, dark=True)
    scores = evaluate_masks([row], Config())
    assert scores['n'] == 1 and scores['failures'] == 1
    assert scores['records'][0]['leaf_iou'] == 0
    assert scores['records'][0]['iou'] == 0
    row = mask_case(tmp_path, empty_leaf=True)
    with pytest.raises(ValueError, match='Empty GT leaf'):
        evaluate_masks([row], Config())


@pytest.mark.parametrize('threshold', RULES['project']['thresholds'])
def test_severity_boundaries(threshold):
    below = severity(np.nextafter(threshold, 0))
    at = severity(threshold)
    assert at['level'] == below['level']+1
    assert at['irri_score'] is None


def test_advice_changes_with_level_and_rejection():
    assert recommendation('brown_spot', .02) != recommendation('brown_spot', .5)
    assert recommendation('brown_spot', .5, True) == RULES['advice']['uncertain']


def test_manifest_merge_preserves_manual_edits(tmp_path):
    path = tmp_path/'manifest.csv'
    row = dict(zip(FIELDS, ['a','image.jpg','rice','brown_spot','plain','real_plant','test','','','self','own']))
    merge_manifest(path, [row])
    changed = dict(row, split='train', group='image_proxy')
    merge_manifest(path, [changed])
    saved = list(csv.DictReader(path.open(encoding='utf-8')))
    assert saved[0]['split'] == 'test' and saved[0]['group'] == 'real_plant'
    assert list(tmp_path.glob('*.backup-*.csv'))


def test_old_schema_rejected_and_distance_explanation(tmp_path):
    old = tmp_path/'old.npz'
    np.savez(old, schema='leaflab-1')
    with pytest.raises(ValueError, match='không tương thích'):
        KNN.load(old)
    x = np.zeros((4, FEATURE_DIM))
    x[2:] = 1
    model = KNN(1).fit(x, ['a','a','b','b'])
    assert model.explain(x[0])['neighbors'][0]['label'] == 'a'


def test_lbp_radius_two_hand_example():
    gray = np.zeros((7,7), np.uint8)
    gray[3,3] = 5
    for dy, dx in [(0,2),(-2,2),(-2,0),(-2,-2),(0,-2),(2,-2),(2,0),(2,2)]:
        gray[3+dy,3+dx] = 4
    gray[3,5] = 6
    gray[5,3] = 5
    assert lbp8(gray,2)[3,3] == 65


def test_lda_knn_separates_and_roundtrips(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (60, FEATURE_DIM))
    y = np.repeat(['a', 'b', 'c'], 20)
    x[20:40, 0] += 6
    x[40:, 1] += 6
    model = KNN(3, [0, 1, 2, 3], lda=.1).fit(x, y)
    assert model.x.shape == (60, 2)
    assert (model.predict(x) == y).mean() > .95
    fresh = rng.normal(0, 1, (30, FEATURE_DIM))
    fresh[10:20, 0] += 6
    fresh[20:, 1] += 6
    accepted = [not d['rejected'] for d in model.predict_details(fresh)]
    assert np.mean(accepted) > .7  # the gate must not reject ordinary unseen rows
    path = tmp_path/'lda.npz'
    model.save(path)
    restored, _, _ = KNN.load(path)
    assert restored.predict(x).tolist() == model.predict(x).tolist()
    assert restored.explain(x[0])['distance_terms'][0]['feature'].startswith('lda_axis_')


def test_kernel_ridge_separates_and_roundtrips(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (60, FEATURE_DIM))
    y = np.repeat(['a', 'b', 'c'], 20)
    x[20:40, 0] += 6
    x[40:, 1] += 6
    model = KernelRidge(.1, 1., [0, 1, 2, 3]).fit(x, y)
    assert (model.predict(x) == y).mean() > .95
    path = tmp_path/'ridge.npz'
    model.save(path)
    restored, _, _ = KNN.load(path)
    assert isinstance(restored, KernelRidge)
    np.testing.assert_allclose(restored.decision(x), model.decision(x))
    assert restored.explain(x[0])['neighbors'][0]['label'] == 'a'
