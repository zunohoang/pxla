"""Classification and mask metrics, including explicit empty-set conventions."""
import numpy as np


def classification(y, pred, labels):
    """Fixed-class metrics, counting failed/unknown predictions as errors.

    If present, failures have an extra confusion column, never a new true class.
    """
    y, pred = list(y), list(pred)
    if len(y) != len(pred):
        raise ValueError('Truth and prediction lengths differ')
    mapping = {str(c): i for i, c in enumerate(labels)}
    has_failure = any(str(c) not in mapping for c in pred)
    cm = np.zeros((len(labels), len(labels)+int(has_failure)), int)
    for a, b in zip(y, pred):
        cm[mapping[str(a)], mapping.get(str(b), len(labels))] += 1
    tp = cm[:, :len(labels)].diagonal()
    den = cm.sum(0)[:len(labels)] + cm.sum(1)
    f1 = np.divide(2*tp, den, out=np.zeros(len(tp), float), where=den > 0)
    return dict(n=int(cm.sum()), accuracy=float(tp.sum()/max(cm.sum(), 1)),
                macro_f1=float(f1.mean()), confusion=cm.tolist(),
                labels=[str(c) for c in labels],
                prediction_labels=[str(c) for c in labels] + (['processing_failed'] if has_failure else []))


def overlap(a, b):
    """IoU/Dice: both empty=1; exactly one empty=0. Inputs are bool masks."""
    intersection = int((a & b).sum())
    union, total = int((a | b).sum()), int(a.sum()+b.sum())
    return dict(iou=intersection/union if union else 1.,
                dice=2*intersection/total if total else 1.)
