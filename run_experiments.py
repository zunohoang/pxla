"""One-command reproducible experiments; no invented field or mask scores."""
import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import time
from dataclasses import replace
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from scripts.verify_algorithms import verify
from src.data import read_manifest
from src.evaluation import classification, overlap
from src.model import KNN, KernelRidge
from src.pipeline import Config, load_rgb, process
from src.features import FEATURE_DIM, FEATURE_GROUPS, FEATURE_NAMES, FEATURE_SCHEMA
from scripts.reference_methods import grabcut_reference


def save_json(path, value):
    """Write readable UTF-8 JSON with no nonstandard NaN values."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def evaluate_masks(rows, config, leaf_reference=False):
    """Evaluate fixed segmentation config on manually supplied segtest masks."""
    records = []
    for r in rows:
        if r['split'] != 'segtest' or not r['lesion_mask'] or not r['leaf_mask']:
            continue
        rgb = load_rgb(r['path'], config.max_side)
        masks = {}
        for key in ['leaf_mask', 'lesion_mask']:
            with Image.open(r[key]) as im:
                with Image.open(r['path']) as original:
                    if im.size != original.size:
                        raise ValueError('Mask must have original image dimensions: ' + r['id'])
                masks[key] = np.array(im.convert('L').resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 127
        if np.any(masks['lesion_mask'] & ~masks['leaf_mask']):
            raise ValueError('GT lesion outside GT leaf: ' + r['id'])
        gt_leaf, gt_lesion = masks['leaf_mask'], masks['lesion_mask']
        if not gt_leaf.any():
            raise ValueError('Empty GT leaf mask: '+r['id'])
        try:
            override = grabcut_reference(rgb) if leaf_reference else None
            out = process(rgb, config, override)
            failure = None
        except ValueError as exc:
            out = dict(leaf=np.zeros_like(gt_leaf), lesion=np.zeros_like(gt_leaf), ratio=0.)
            failure = str(exc)
        # End-to-end IoU uses full image, so leaf omissions count as errors.
        scores = overlap(out['lesion'], gt_lesion) if failure is None else dict(iou=0., dice=0.)
        gt_ratio = gt_lesion.sum()/max(gt_leaf.sum(), 1)
        records.append(dict(id=r['id'], crop=r.get('crop'), domain=r.get('domain'),
                            processing_failed=failure is not None, error=failure, **scores,
                            leaf_iou=overlap(out['leaf'], gt_leaf)['iou'],
                            ratio_abs_error=abs(out['ratio']-float(gt_ratio)),
                            empty_gt=not bool(gt_lesion.any())))
    summary = dict(n=len(records), required=30, status='complete' if len(records) >= 30 else 'pending',
                   failures=sum(r['processing_failed'] for r in records), records=records)
    if records:
        summary['mean_std'] = {k: [float(np.mean([r[k] for r in records])),
                                    float(np.std([r[k] for r in records], ddof=1))
                                    if len(records) > 1 else 0.]
                               for k in ['iou', 'dice', 'leaf_iou', 'ratio_abs_error']}
        nonempty = [r['iou'] for r in records if not r['empty_gt']]
        summary['nonempty_gt_iou'] = float(np.mean(nonempty)) if nonempty else None
        summary['empty_gt_count'] = sum(r['empty_gt'] for r in records)
    return summary


def main():
    """Run grouped CV selection, held-out/domain tests and illumination stress."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifest.csv')
    p.add_argument('--out', default='results')
    p.add_argument('--max-side', type=int, default=256)
    args = p.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    old_metrics = outdir/'metrics.json'
    old_schema = json.loads(old_metrics.read_text(encoding='utf-8')).get('schema') if old_metrics.exists() else FEATURE_SCHEMA
    if old_schema != FEATURE_SCHEMA:
        archive = outdir/('baseline_v1' if not old_schema else 'baseline_'+old_schema)
        archive.mkdir(exist_ok=True)
        for name in ['metrics.json', 'cross_validation.json', 'verification.json', 'predictions.csv']:
            if (outdir/name).exists() and not (archive/name).exists():
                shutil.copy2(outdir/name, archive/name)
    Path('models').mkdir(exist_ok=True)
    save_json(outdir / 'verification.json', verify())
    save_json(outdir / 'environment.json', dict(
        os=platform.platform(), processor=platform.processor(), python=platform.python_version(),
        packages={m: importlib.metadata.version(m) for m in
                  ['numpy', 'Pillow', 'opencv-python-headless', 'scikit-image',
                   'scikit-learn', 'streamlit', 'matplotlib']}))
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError('Manifest empty; download/import dataset first')
    config = Config(max_side=args.max_side)
    report = dict(schema=FEATURE_SCHEMA, feature_dim=FEATURE_DIM,
                  feature_groups={k:len(v) for k,v in FEATURE_GROUPS.items()},
                  seed=42, config=config.to_dict(), crops={},
                  field_photos=len([r for r in rows if r['domain'] == 'field' and r['source'] == 'self']),
                  limitations=['Image-level proxy groups must be audited against plant/session identity.',
                               'Severity cutoffs are project conventions, not an agronomic disease scale.'])
    results, all_cv, preview = [], [], []
    for crop in sorted({r['crop'] for r in rows}):
        subset = [r for r in rows if r['crop'] == crop and r['split'] in ['train', 'test']]
        if not subset:
            continue
        x, timings, failed = [], [], []
        for i, r in enumerate(subset):
            start = time.perf_counter()
            rgb = load_rgb(r['path'], config.max_side)
            try:
                result = process(rgb, config)
            except ValueError as exc:
                if r['split'] == 'train':
                    raise ValueError(f"Train segmentation failed for {r['id']}; review using train only: {exc}") from exc
                failed.append(dict(id=r['id'], error=str(exc)))
                x.append(np.zeros(FEATURE_DIM))
                timings.append(time.perf_counter()-start)
                continue
            x.append(result['features'])
            timings.append(time.perf_counter()-start)
            if len(preview) < 6 and r['split'] == 'test':
                preview.append((rgb, result['leaf'], result['overlay'], r['label']))
            if (i+1) % 20 == 0:
                print(f'{crop}: extracted {i+1}/{len(subset)} images', flush=True)
        x = np.array(x)
        y = np.array([r['label'] for r in subset])
        groups = np.array([r['group'] for r in subset])
        train = np.array([r['split'] == 'train' for r in subset])
        test = ~train
        labels = sorted(np.unique(y[train]))
        if len(labels) < 2 or not test.any() or set(y[test])-set(labels):
            raise ValueError(f'{crop}: need >=2 train classes and held-out test with known labels')
        minimum = min(len(set(groups[train & (y == c)])) for c in labels)
        if minimum < 2:
            raise ValueError('Need >=2 independent training groups per class')
        folds = list(StratifiedGroupKFold(min(5, minimum), shuffle=True, random_state=42)
                     .split(x[train], y[train], groups[train]))
        if any(set(y[train][a]) != set(labels) for a, _ in folds):
            raise ValueError('Some CV training fold lacks a class; collect more independent groups')
        # Use training CV only for all hyperparameter and feature selection.
        cvrows = []
        # Plain k-NN on every group; LDA + k-NN (the improvement) on groups holding leaf features.
        # and RBF kernel ridge on the same groups. All are NumPy implementations in src/.
        grid = [dict(method='knn', features=name, k=k, lda=None) for name in FEATURE_GROUPS for k in [1, 3, 5, 7]]
        grid += [dict(method='lda+knn', features=name, k=k, lda=lda)
                 for name in ['leaf', 'all'] for lda in [.03, .1, .3] for k in [1, 5, 11, 21]]
        grid += [dict(method='kernel_ridge', features=name, k=5, lda=None, gamma=gamma, alpha=alpha)
                 for name in ['leaf', 'all'] for gamma in [.1, .3, 1.] for alpha in [.01, .1, 1.]]

        def build(spec):
            """Unfitted NumPy classifier for one grid row."""
            indices = FEATURE_GROUPS[spec['features']]
            if spec['method'] == 'kernel_ridge':
                return KernelRidge(spec['alpha'], spec['gamma'], indices, spec['k'])
            return KNN(spec['k'], indices, spec['lda'])

        for spec in grid:
            if True:
                scores = []
                for a, b in folds:
                    model = build(spec).fit(x[train][a], y[train][a])
                    scores.append(classification(y[train][b], model.predict(x[train][b]), labels)['macro_f1'])
                row = dict(crop=crop, **spec,
                           macro_f1_mean=float(np.mean(scores)),
                           macro_f1_std=float(np.std(scores, ddof=1)),
                           fold_scores=scores)
                cvrows.append(row)
        best = max(cvrows, key=lambda r: r['macro_f1_mean'])
        model = build(best).fit(x[train], y[train])
        model.save(Path('models') / f'{crop}_knn.npz', config, crop,
                   sample_ids=[r['id'] for r in subset if r['split'] == 'train'])
        pred = model.predict(x[test]).astype(object)
        details = model.predict_details(x[test])
        test_rows = [r for r in subset if r['split'] == 'test']
        failed_ids = {r['id'] for r in failed}
        for i, r in enumerate(test_rows):
            if r['id'] in failed_ids:
                pred[i] = '__segmentation_failed__'
                details[i] = dict(label='__segmentation_failed__', vote=0., distance=None, rejected=True)
        metrics = classification(y[test], pred, labels)
        keep = np.array([not d['rejected'] for d in details])
        metrics['coverage'] = float(keep.mean())
        metrics['accepted_accuracy'] = float((pred[keep] == y[test][keep]).mean()) if keep.any() else None
        # SVM library comparator is not the deployed NumPy core classifier.
        svm = GridSearchCV(make_pipeline(StandardScaler(), PCA(),
                                         SVC(kernel='rbf', class_weight='balanced')),
                           {'pca':['passthrough', PCA(.95, svd_solver='full')],
                            'svc__C':[.1, 1., 10.], 'svc__gamma':['scale', .01]},
                           cv=folds, scoring=lambda estimator, features, truth:
                               classification(truth, estimator.predict(features), labels)['macro_f1'],
                           n_jobs=1, error_score='raise')
        svm.fit(x[train], y[train])
        best_idx = svm.best_index_
        svm_cv = [float(svm.cv_results_[f'split{i}_test_score'][best_idx]) for i in range(len(folds))]
        svm_pred = svm.predict(x[test]).astype(object)
        for i, r in enumerate(test_rows):
            if r['id'] in failed_ids:
                svm_pred[i] = '__segmentation_failed__'
        svm_test = classification(y[test], svm_pred, labels)
        domains = {}
        for domain in ['plain', 'field']:
            idx = np.array([r['domain'] == domain for r in test_rows])
            domains[domain] = classification(y[test][idx], pred[idx], labels) if idx.any() else dict(n=0, status='pending')
        stress = []
        transforms = [('brightness_0.6', lambda a: np.clip(a.astype(float)*.6,0,255).astype('uint8')),
                      ('brightness_1.4', lambda a: np.clip(a.astype(float)*1.4,0,255).astype('uint8')),
                      ('blur_sigma_1', lambda a: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1)))),
                      ('blur_sigma_2', lambda a: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(2)))),
                      ('rotate_90', lambda a: np.rot90(a).copy()),
                      ('rotate_180', lambda a: np.rot90(a, 2).copy()),
                      ('rotate_15', lambda a: np.array(Image.fromarray(a).rotate(15, resample=Image.Resampling.BILINEAR,
                                                                              expand=True, fillcolor=(240,240,240))))]
        for transform_name, transform in transforms:
            stress_x, stress_y, failures = [], [], 0
            for r in test_rows:
                rgb = load_rgb(r['path'], config.max_side)
                rgb = transform(rgb)
                try:
                    stress_x.append(process(rgb, config)['features'])
                    stress_y.append(r['label'])
                except ValueError:
                    failures += 1
            measured = classification(stress_y, model.predict(np.array(stress_x)), labels) if stress_x else None
            # Keep all samples in an end-to-end accuracy denominator.
            end_to_end = (measured['accuracy']*len(stress_y)/len(test_rows)) if measured else 0.
            stress.append(dict(transform=transform_name,
                               factor=.6 if transform_name == 'brightness_0.6' else 1.4 if transform_name == 'brightness_1.4' else None,
                               end_to_end_accuracy=end_to_end, segmentation_failures=failures,
                               total=len(test_rows), successful_only_metrics=measured))
        timing = dict(mean_ms=float(np.mean(timings)*1000), std_ms=float(np.std(timings, ddof=1)*1000))
        report['crops'][crop] = dict(n_train=int(train.sum()), n_test=int(test.sum()),
            labels=labels, selected=best, knn_test=metrics, svm_test=svm_test,
            test_processing_failures=failed,
            svm_selected={k:str(v) for k,v in svm.best_params_.items()},
            svm_cv_mean=float(np.mean(svm_cv)), svm_cv_std=float(np.std(svm_cv, ddof=1)),
            domains=domains, brightness_stress=[r for r in stress if r['transform'].startswith('brightness')],
            robustness_stress=stress, processing_time=timing)
        # Engineering ablations on training folds only; no test-driven tuning.
        preprocessing_cv = []
        for variant, settings in [('without_preprocessing', replace(config, preprocessing=False)),
                                  ('with_preprocessing', config),
                                  ('lab_rule_baseline', replace(config, preprocessing=False, lesion_method='lab_rule'))]:
            if variant == 'with_preprocessing':
                variant_x = x[train]
            else:
                variant_x = np.array([process(load_rgb(r['path'], settings.max_side), settings)['features']
                                      for r in subset if r['split'] == 'train'])
            scores = []
            for a, b in folds:
                candidate = KNN(3, FEATURE_GROUPS['lesion']).fit(variant_x[a], y[train][a])
                scores.append(classification(y[train][b], candidate.predict(variant_x[b]), labels)['macro_f1'])
            preprocessing_cv.append(dict(variant=variant, macro_f1_mean=float(np.mean(scores)),
                                         macro_f1_std=float(np.std(scores, ddof=1)), fold_scores=scores))
        report['crops'][crop]['preprocessing_cv'] = preprocessing_cv
        print(f'Finished {crop}: accuracy={metrics["accuracy"]:.3f}', flush=True)
        all_cv.extend(cvrows)
        for r, a, d in zip(test_rows, pred, details):
            results.append(dict(id=r['id'], crop=crop, domain=r['domain'],
                                true=r['label'], predicted=str(a), **{k:d[k] for k in ['vote', 'distance', 'rejected']}))
        fig, ax = plt.subplots(figsize=(7, 5))
        cm = np.array(metrics['confusion'])
        ax.imshow(cm, cmap='Greens')
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha='center', va='center')
        ax.set(xticks=range(cm.shape[1]), yticks=range(len(labels)),
               xticklabels=metrics['prediction_labels'], yticklabels=labels,
               xlabel='Predicted class', ylabel='True class')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        fig.tight_layout()
        fig.savefig(outdir / f'{crop}_confusion.png', dpi=180)
        plt.close(fig)
    report['segmentation'] = evaluate_masks(rows, config)
    report['segmentation_comparison'] = {}
    for method in ['hybrid', 'hsv', 'exg_otsu', 'exg_exr', 'hybrid_se1x7', 'hybrid_se7x1', 'grabcut_reference']:
        method_config = replace(config, leaf_method=method if method in ['hybrid','hsv','exg_otsu','exg_exr'] else 'hybrid',
                                morph_height=7 if method == 'hybrid_se7x1' else 1 if method == 'hybrid_se1x7' else 3,
                                morph_width=7 if method == 'hybrid_se1x7' else 1 if method == 'hybrid_se7x1' else 3)
        report['segmentation_comparison'][method] = evaluate_masks(
            rows, method_config,
            leaf_reference=method == 'grabcut_reference')
    report['lesion_comparison'] = {}
    for method in ['ab_distance', 'lab_rule', 'otsu_a']:
        for clusters in [2, 3, 4] if method != 'otsu_a' else [2]:
            report['lesion_comparison'][f'{method}_k{clusters}'] = evaluate_masks(
                rows, replace(config, lesion_method=method, clusters=clusters))
    report['field_requirement'] = 'complete' if report['field_photos'] >= 100 else 'pending'
    save_json(outdir / 'metrics.json', report)
    save_json(outdir / 'cross_validation.json', all_cv)
    with open(outdir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]) if results else
                                ['id','crop','domain','true','predicted','vote','distance','rejected'])
        writer.writeheader()
        writer.writerows(results)
    if preview:
        fig, axes = plt.subplots(len(preview), 3, figsize=(9, 2.5*len(preview)), squeeze=False)
        for i, (rgb, leaf, overlay, label) in enumerate(preview):
            for j, image in enumerate([rgb, leaf, overlay]):
                axes[i, j].imshow(image, cmap='gray' if j == 1 else None)
                axes[i, j].axis('off')
            axes[i, 0].set_title(label, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / 'pipeline_examples.png', dpi=160)
        plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
