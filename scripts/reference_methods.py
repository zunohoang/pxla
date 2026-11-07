"""Explicit library comparators; never used silently by the NumPy core."""
import cv2
import numpy as np
from src.pipeline import Config, leaf_mask


def grabcut_reference(rgb):
    """OpenCV GrabCut initialized from NumPy mask, with fixed RNG seed."""
    initial = leaf_mask(rgb, Config())
    mask = np.where(initial, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    mask[0] = mask[-1] = cv2.GC_BGD
    mask[:, 0] = mask[:, -1] = cv2.GC_BGD
    if not np.any(mask == cv2.GC_PR_FGD):
        raise ValueError('No foreground seed for GrabCut')
    cv2.setRNGSeed(42)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(rgb, mask, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise ValueError('GrabCut failed to construct foreground/background models') from exc
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
