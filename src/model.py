"""Standardized NumPy k-NN with safe NPZ persistence and rejection gate."""
import json

import numpy as np

from .pipeline import Config
from .features import FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA


class KNN:
    """Uniform-vote Euclidean k-NN. Vote share is not calibrated probability."""

    def __init__(self, k=3, indices=None):
        self.k = k
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
        norms = (self.x**2).sum(1)
        d = np.sqrt(np.maximum(norms[:, None] + norms[None, :] - 2*self.x@self.x.T, 0))
        np.fill_diagonal(d, np.inf)
        self.radius = float(np.quantile(d.min(1), .95) * 1.5) if len(x) > 1 else 0.
        return self

    def predict_details(self, x):
        """Return label, vote share, nearest distance, rejection flag for each row."""
        z = (np.asarray(x)[:, self.indices] - self.mean) / self.std
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

    def save(self, path, config=Config(), crop='rice', sample_ids=None):
        """Save numeric/string arrays only; no pickle executable objects."""
        np.savez_compressed(path, x=self.x, y=self.y, mean=self.mean, std=self.std,
                            indices=self.indices, k=self.k, radius=self.radius,
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
            obj = cls(int(d['k']), d['indices'])
            for name in ['x', 'y', 'mean', 'std']:
                setattr(obj, name, d[name])
            obj.radius, obj.classes = float(d['radius']), np.unique(obj.y)
            if (obj.x.ndim != 2 or obj.x.shape[1] != len(obj.indices)
                    or obj.mean.shape != (len(obj.indices),) or obj.std.shape != obj.mean.shape
                    or len(obj.y) != len(obj.x) or (obj.std <= 0).any()
                    or (obj.indices < 0).any() or (obj.indices >= FEATURE_DIM).any()
                    or not all(np.isfinite(a).all() for a in [obj.x, obj.mean, obj.std])):
                raise ValueError('Invalid classifier arrays')
            obj.sample_ids = d['sample_ids']
            return obj, Config(**json.loads(str(d['config']))), str(d['crop'])

    def explain(self, x, n_neighbors=3, n_features=5):
        """Nearest reference samples and largest squared-distance terms.

        These are distance contributions to one neighbor, not causal importance.
        """
        row = (np.asarray(x)[self.indices]-self.mean)/self.std
        squared = (self.x-row)**2
        nearest = np.argsort(squared.sum(1), kind='stable')[:n_neighbors]
        ids = getattr(self, 'sample_ids', np.arange(len(self.x)).astype(str))
        neighbors = [dict(id=str(ids[i]), label=str(self.y[i]),
                          distance=float(np.sqrt(squared[i].sum()))) for i in nearest]
        terms = np.argsort(squared[nearest[0]])[-n_features:][::-1]
        return dict(neighbors=neighbors, distance_terms=[dict(
            feature=FEATURE_NAMES[self.indices[j]], squared_difference=float(squared[nearest[0], j]))
            for j in terms])
