"""Standardized NumPy k-NN (optional LDA projection) with safe NPZ persistence and rejection gate."""
import json

import numpy as np

from .algorithms import kernel_ridge_fit, lda_projection, rbf_kernel
from .pipeline import Config
from .features import FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA


class KNN:
    """Uniform-vote Euclidean k-NN. Vote share is not calibrated probability.

    With `lda` set (shrinkage in (0, 1]) the standardized features are projected
    onto C-1 Fisher discriminant axes, fitted on training data only, before the
    neighbour search.
    """

    def __init__(self, k=3, indices=None, lda=None):
        self.k = k
        self.lda = None if lda is None else float(lda)
        self.projection = None
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        self.indices = np.arange(FEATURE_DIM) if indices is None else np.asarray(indices, int)

    def fit(self, x, y):
        """Fit scaling on training data only and estimate distance rejection."""
        x = np.asarray(x)[:, self.indices]
        if len(x) < 2 or len(y) != len(x) or not np.isfinite(x).all():
            raise ValueError('At least two finite training rows with matching labels required')
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[self.std < 1e-8] = 1
        self.x = (x - self.mean) / self.std
        self.y = np.asarray(y, str)
        self.classes = np.unique(self.y)
        nearest = None
        if self.lda is not None:
            nearest = self._held_out_nearest(x, self.y)
            self.projection = lda_projection(self.x, self.y, self.lda)
            self.x = self.x @ self.projection
        if nearest is None:
            norms = (self.x**2).sum(1)
            d = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2*self.x@self.x.T, 0))
            np.fill_diagonal(d, np.inf)
            nearest = d.min(1)
        self.radius = float(np.quantile(nearest, .95) * 1.5) if len(x) > 1 else 0.
        return self

    def _held_out_nearest(self, x, y, folds=5):
        """Nearest-neighbour distances of rows left out of the LDA fit.

        LDA is fitted to the training labels, so training rows sit unrealistically
        close together in its space; a rejection radius taken from them rejects
        almost every new image. Each row is therefore measured against a
        projection fitted without its fold (class-stratified, deterministic).
        Returns None when the data are too small to hold out.
        """
        fold = np.empty(len(y), int)
        fold[np.argsort(y, kind='stable')] = np.arange(len(y)) % folds
        nearest = np.empty(len(y))
        for f in range(folds):
            held, rest = fold == f, fold != f
            if not held.any() or len(np.unique(y[rest])) < 2:
                return None
            mean, std = x[rest].mean(0), x[rest].std(0)
            std[std < 1e-8] = 1
            z = (x[rest]-mean)/std
            projection = lda_projection(z, y[rest], self.lda)
            a, b = (x[held]-mean)/std@projection, z@projection
            d = (a**2).sum(1)[:, None]+(b**2).sum(1)[None]-2*a@b.T
            nearest[held] = np.sqrt(np.maximum(d.min(1), 0))
        return nearest

    def _transform(self, x):
        """Standardize with training statistics, then apply the LDA axes if any."""
        z = (x - self.mean) / self.std
        return z if self.projection is None else z @ self.projection

    def predict_details(self, x):
        """Return label, vote share, nearest distance, rejection flag for each row."""
        z = self._transform(np.asarray(x)[:, self.indices])
        details = []
        for row in z:
            distances = np.sqrt(((self.x - row)**2).sum(1))
            order = np.argsort(distances, kind='stable')[:min(self.k, len(self.x))]
            counts = np.array([(self.y[order] == c).sum() for c in self.classes])
            label = str(self.classes[counts.argmax()])
            vote, distance = float(counts.max()/len(order)), float(distances.min())
            details.append(dict(label=label, vote=vote, distance=distance,
                                rejected=bool(vote < .6 or distance > self.radius)))
        return details

    def predict(self, x):
        """Closed-set labels for comparable accuracy and macro-F1 reporting."""
        return np.array([r['label'] for r in self.predict_details(x)])

    method = 'knn'

    def _extra(self):
        """Additional arrays a subclass needs to persist."""
        return {}

    def save(self, path, config=Config(), crop='rice', sample_ids=None):
        """Save numeric/string arrays only; no pickle executable objects."""
        np.savez_compressed(path, method=self.method, **self._extra(),
                            x=self.x, y=self.y, mean=self.mean, std=self.std,
                            indices=self.indices, k=self.k, radius=self.radius,
                            lda=np.nan if self.lda is None else self.lda,
                            projection=np.zeros((0, 0)) if self.projection is None else self.projection,
                            config=json.dumps(config.to_dict()), crop=crop,
                            schema=FEATURE_SCHEMA, feature_dim=FEATURE_DIM,
                            feature_names=np.array(FEATURE_NAMES),
                            sample_ids=np.array(sample_ids if sample_ids is not None else
                                                [str(i) for i in range(len(self.x))]))

    @classmethod
    def load(cls, path):
        """Return validated model, frozen pipeline config, crop identifier."""
        with np.load(path, allow_pickle=False) as d:
            if str(d['schema']) != FEATURE_SCHEMA:
                raise ValueError('Mô hình cũ không tương thích; chạy lại run_experiments.py.')
            if int(d['feature_dim']) != FEATURE_DIM or d['feature_names'].tolist() != list(FEATURE_NAMES):
                raise ValueError('Feature schema mismatch; retrain model')
            if 'method' in d.files and str(d['method']) == 'kernel_ridge':
                obj = KernelRidge(float(d['alpha']), float(d['gamma']), d['indices'], int(d['k']))
                obj.weights, obj.offset = d['weights'], d['offset']
            else:
                obj = KNN(int(d['k']), d['indices'], None if np.isnan(d['lda']) else float(d['lda']))
            for name in ['x', 'y', 'mean', 'std']:
                setattr(obj, name, d[name])
            if obj.lda is not None:
                obj.projection = d['projection']
            obj.radius, obj.classes = float(d['radius']), np.unique(obj.y)
            width = len(obj.indices) if obj.projection is None else obj.projection.shape[1]
            if (obj.x.ndim != 2 or obj.x.shape[1] != width
                    or (obj.projection is not None and (obj.projection.shape[0] != len(obj.indices)
                                                        or not np.isfinite(obj.projection).all()))
                    or obj.mean.shape != (len(obj.indices),) or obj.std.shape != obj.mean.shape
                    or len(obj.y) != len(obj.x) or (obj.std <= 0).any()
                    or (obj.indices < 0).any() or (obj.indices >= FEATURE_DIM).any()
                    or not all(np.isfinite(a).all() for a in [obj.x, obj.mean, obj.std])):
                raise ValueError('Invalid classifier arrays')
            if isinstance(obj, KernelRidge) and (obj.weights.shape != (len(obj.x), len(obj.classes))
                                                 or obj.offset.shape != (len(obj.classes),)
                                                 or not np.isfinite(obj.weights).all()):
                raise ValueError('Invalid kernel ridge arrays')
            obj.sample_ids = d['sample_ids']
            return obj, Config(**json.loads(str(d['config']))), str(d['crop'])

    def explain(self, x, n_neighbors=3, n_features=5):
        """Nearest reference samples and largest squared-distance terms.

        These are distance contributions to one neighbor, not causal importance.
        With LDA the terms are discriminant axes, not original features.
        """
        row = self._transform(np.asarray(x)[self.indices][None])[0]
        squared = (self.x-row)**2
        nearest = np.argsort(squared.sum(1), kind='stable')[:n_neighbors]
        ids = getattr(self, 'sample_ids', np.arange(len(self.x)).astype(str))
        neighbors = [dict(id=str(ids[i]), label=str(self.y[i]),
                          distance=float(np.sqrt(squared[i].sum()))) for i in nearest]
        terms = np.argsort(squared[nearest[0]])[-n_features:][::-1]
        return dict(neighbors=neighbors, distance_terms=[dict(
            feature=FEATURE_NAMES[self.indices[j]] if self.projection is None else f'lda_axis_{j+1}',
            squared_difference=float(squared[nearest[0], j]))
            for j in terms])


class KernelRidge(KNN):
    """One-vs-rest RBF kernel ridge classifier on standardized features.

    Targets are +1/-1 per class, centred by their mean; the label is the class
    with the largest decision value. gamma is divided by the number of features.
    The rejection gate reuses the k-NN evidence: agreement of the k nearest
    training samples with the predicted label, and distance to the nearest one.
    """

    method = 'kernel_ridge'

    def __init__(self, alpha=.01, gamma=.1, indices=None, k=5):
        super().__init__(k, indices)
        if alpha <= 0 or gamma <= 0:
            raise ValueError('alpha and gamma must be positive')
        self.alpha, self.gamma = float(alpha), float(gamma)

    def fit(self, x, y):
        """Fit scaling, rejection radius and dual coefficients on training data only."""
        super().fit(x, y)
        targets = (self.y[:, None] == self.classes)*2.-1
        self.offset = targets.mean(0)
        kernel = rbf_kernel(self.x, self.x, self.gamma/self.x.shape[1])
        self.weights = kernel_ridge_fit(kernel, targets-self.offset, self.alpha)
        return self

    def decision(self, x):
        """Decision values, one column per class, for raw feature rows."""
        z = self._transform(np.asarray(x)[:, self.indices])
        return rbf_kernel(z, self.x, self.gamma/self.x.shape[1])@self.weights+self.offset

    def predict_details(self, x):
        """Label from decision values; vote = neighbour agreement with that label."""
        z = self._transform(np.asarray(x)[:, self.indices])
        scores = rbf_kernel(z, self.x, self.gamma/self.x.shape[1])@self.weights+self.offset
        details = []
        for row, score in zip(z, scores):
            label = str(self.classes[score.argmax()])
            distances = np.sqrt(((self.x-row)**2).sum(1))
            order = np.argsort(distances, kind='stable')[:min(self.k, len(self.x))]
            vote, distance = float((self.y[order] == label).mean()), float(distances.min())
            details.append(dict(label=label, vote=vote, distance=distance,
                                rejected=bool(vote < .6 or distance > self.radius)))
        return details

    def _extra(self):
        return dict(alpha=self.alpha, gamma=self.gamma, weights=self.weights, offset=self.offset)
