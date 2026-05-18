"""Inference utilities for the trained EDNIG generator.

Two paths:

1. EDNIGEnhancer.enhance(...) -- full neural network path (requires weights).
2. classical_illumination_enhance(...) -- training-free fallback based on BCP.
"""
from __future__ import annotations

import json
import os
import cv2
import numpy as np

from .bcp import estimate_illumination


def _preprocess_low_image_bgr(bgr, img_size):
    import torch
    h, w = bgr.shape[:2]
    bgr_small = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    illum = estimate_illumination(bgr_small)
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    illum_t = torch.from_numpy((illum - 0.5) * 2.0).unsqueeze(0).unsqueeze(0)
    inp = torch.cat([rgb_t, illum_t], dim=1)
    return inp, (h, w)


def _postprocess_output_bgr(out_tensor, original_hw):
    out = out_tensor.detach().cpu().clamp(-1, 1)
    out = (out.squeeze(0).permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    h, w = original_hw
    if (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return bgr


class EDNIGEnhancer:
    def __init__(self, weights_path=None, img_size=512, device=None):
        self.img_size = img_size
        self.model = None
        self.weights_path = weights_path
        self.device = None
        self.info = None       # populated from sidecar JSON if available
        self._requested_device = device
        if weights_path and os.path.isfile(weights_path):
            self._load(weights_path)
            self._load_info(weights_path)

    def has_weights(self):
        return self.model is not None

    def _load(self, path):
        import torch
        from .model import Generator
        device = self._requested_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        net = Generator()
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        net.eval().to(self.device)
        self.model = net

    def _load_info(self, weights_path):
        """Look for a sidecar JSON next to the checkpoint."""
        wp = os.path.dirname(os.path.abspath(weights_path)) or "."
        candidates = [
            os.path.join(wp, "ednig_model_info.json"),
            weights_path.replace(".pt", ".json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        self.info = json.load(f)
                    break
                except (OSError, json.JSONDecodeError):
                    pass

    def enhance(self, bgr):
        if self.model is None:
            raise RuntimeError("No model weights loaded.")
        import torch
        with torch.inference_mode():
            inp, hw = _preprocess_low_image_bgr(bgr, self.img_size)
            inp = inp.to(self.device)
            out = self.model(inp)
            return _postprocess_output_bgr(out, hw)


def classical_illumination_enhance(bgr, strength=1.0):
    """Training-free fallback: BCP-driven adaptive gamma + CLAHE."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    illum = estimate_illumination(bgr)
    illum = cv2.GaussianBlur(illum, (0, 0), sigmaX=3.0)

    gamma = 1.0 - 0.6 * (illum * strength)
    gamma = np.clip(gamma, 0.4, 1.0)

    v_norm = np.clip(v_ch / 255.0, 1e-6, 1.0)
    v_new = np.power(v_norm, gamma) * 255.0
    v_new = np.clip(v_new, 0, 255)

    sat_gain = 1.0 + 0.3 * (illum * strength)
    s_new = np.clip(s_ch * sat_gain, 0, 255)

    hsv_new = cv2.merge([h_ch, s_new, v_new]).astype(np.uint8)
    out = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_ch = clahe.apply(L_ch)
    lab = cv2.merge([L_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
