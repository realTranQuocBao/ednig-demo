"""Bright Channel Prior (BCP) for illumination map estimation.

Port of the original NumPy/OpenCV implementation in core/bcp.py. The output
is an HxW illumination guidance map in [0, 1], where higher values mean
darker (i.e. needs more brightening).
"""
import cv2
import numpy as np


def estimate_bright_channel(image: np.ndarray, sz: int = 3) -> np.ndarray:
    """Per-pixel bright channel (max over R,G,B) followed by morphological dilation."""
    b, g, r = cv2.split(image)
    bc = cv2.max(cv2.max(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.dilate(bc, kernel)


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving guided filter."""
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * I + mean_b


def transmission_refine(image_bgr: np.ndarray, raw_bright: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64) / 255.0
    return guided_filter(gray, raw_bright, r=60, eps=1e-4)


def normalize_img(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.amin(x)), float(np.amax(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def estimate_illumination(src_bgr: np.ndarray) -> np.ndarray:
    """Return the illumination guidance map (HxW, float in [0,1]) for a BGR image."""
    rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    bright = estimate_bright_channel(rgb, sz=3)
    bright_refined = transmission_refine(src_bgr, bright)
    inv = 1.0 - bright_refined
    return normalize_img(inv).astype(np.float32)
