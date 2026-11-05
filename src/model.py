"""Standardized NumPy k-NN with safe NPZ persistence and rejection gate."""
import json

import numpy as np

from .pipeline import Config


class KNN:
    """Uniform-vote Euclidean k-NN. Vote share is not calibrated probability."""

    def __init__(self, k=3, indices=None):
        self.k = k
        self.indices = np.arange(74) if indices is None else np.array(indices)

    def fit(self, x, y):
        """Fit scaling on training data only and estimate distance rejection."""
        x = np.asarray(x)[:, self.indices]
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

    def save(self, path, config=Config(), crop='rice'):
        """Save numeric/string arrays only; no pickle executable objects."""
        np.savez_compressed(path, x=self.x, y=self.y, mean=self.mean, std=self.std,
                            indices=self.indices, k=self.k, radius=self.radius,
                            config=json.dumps(config.to_dict()), crop=crop,
                            schema='leaflab-1')

    @classmethod
    def load(cls, path):
        """Return validated model, frozen pipeline config, crop identifier."""
        with np.load(path, allow_pickle=False) as d:
            if str(d['schema']) != 'leaflab-1':
                raise ValueError('Unsupported model schema')
            obj = cls(int(d['k']), d['indices'])
            for name in ['x', 'y', 'mean', 'std']:
                setattr(obj, name, d[name])
            obj.radius, obj.classes = float(d['radius']), np.unique(obj.y)
            return obj, Config(**json.loads(str(d['config']))), str(d['crop'])
