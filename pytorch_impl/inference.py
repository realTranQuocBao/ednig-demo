"""Inference utilities for the trained EDNIG generator.

Three modes (via ``EDNIGEnhancer.enhance(..., mode=...)``):

  - ``"single"`` : resize the whole image to 512x512, run the model once,
                   then resize the result back. Fastest but loses fine
                   detail on high-resolution input.
  - ``"tiled"``  : slide a 512x512 window with overlap across the *native*
                   resolution, blend overlapping tiles with a Hann window.
                   Preserves full detail at any resolution.
  - ``"auto"``   : pick single for images with max side <= 768, tiled
                   otherwise. Sensible default.

Also exposes ``classical_illumination_enhance`` as a training-free fallback.
"""
from __future__ import annotations

import json
import os
import cv2
import numpy as np

from .bcp import estimate_illumination


# ---------- Tile helpers ----------

_HANN_CACHE: dict[int, np.ndarray] = {}


def _hann2d(size: int) -> np.ndarray:
    """2D Hann window for smooth tile blending."""
    if size in _HANN_CACHE:
        return _HANN_CACHE[size]
    w = np.hanning(size).astype(np.float32)
    w[0] = w[-1] = 1e-3  # avoid exact zeros at edges
    w2 = np.outer(w, w)[..., None]  # H x W x 1
    _HANN_CACHE[size] = w2
    return w2


# ---------- Single-pass tensor I/O (model always sees img_size x img_size) ----------

def _bgr_to_model_input(bgr_512, device):
    """Convert a BGR image (already at img_size x img_size) to a 4-channel tensor."""
    import torch
    illum = estimate_illumination(bgr_512)
    rgb = cv2.cvtColor(bgr_512, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    illum_t = torch.from_numpy((illum - 0.5) * 2.0).unsqueeze(0).unsqueeze(0)
    inp = torch.cat([rgb_t, illum_t], dim=1).to(device)
    return inp


def _model_output_to_bgr(out_tensor):
    """Generator output [-1,1] -> uint8 BGR at the model's native size."""
    out = out_tensor.detach().cpu().clamp(-1, 1)
    out = (out.squeeze(0).permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class EDNIGEnhancer:
    """Run the trained EDNIG generator on arbitrary-resolution images.

    Tile inference makes the model work well on inputs much larger than its
    training crop (e.g. 12 MP photos), without resizing the whole image
    down to 512x512.
    """

    def __init__(self, weights_path=None, img_size=512, device=None):
        self.img_size = img_size
        self.model = None
        self.weights_path = weights_path
        self.device = None
        self.info = None
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

    # ----- Strategy 1: single pass (resize whole image to 512) -----

    def _enhance_single(self, bgr):
        import torch
        h, w = bgr.shape[:2]
        sz = self.img_size
        small = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            inp = _bgr_to_model_input(small, self.device)
            out = self.model(inp)
            out_bgr = _model_output_to_bgr(out)
        if (out_bgr.shape[1], out_bgr.shape[0]) != (w, h):
            out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out_bgr

    # ----- Strategy 2: tiled inference at native resolution -----

    def _enhance_tiled(self, bgr, overlap=64):
        """Slide a tile of size ``img_size`` over the native-res image."""
        import torch
        h, w = bgr.shape[:2]
        tile = self.img_size
        stride = max(1, tile - overlap)

        # If image is smaller than a tile, upscale just enough to fit
        scale_up = 1.0
        if h < tile or w < tile:
            scale_up = max(tile / h, tile / w)
            new_h = int(np.ceil(h * scale_up))
            new_w = int(np.ceil(w * scale_up))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = bgr.shape[:2]

        # Pad on bottom/right so (h - tile) % stride == 0
        pad_h = (stride - max(0, (h - tile)) % stride) % stride
        pad_w = (stride - max(0, (w - tile)) % stride) % stride
        H = h + pad_h
        W = w + pad_w
        bgr_p = cv2.copyMakeBorder(bgr, 0, H - h, 0, W - w, cv2.BORDER_REFLECT_101)

        out_acc = np.zeros((H, W, 3), dtype=np.float32)
        wgt_acc = np.zeros((H, W, 1), dtype=np.float32)
        hann = _hann2d(tile)

        ys = list(range(0, H - tile + 1, stride))
        xs = list(range(0, W - tile + 1, stride))
        if ys[-1] + tile < H:
            ys.append(H - tile)
        if xs[-1] + tile < W:
            xs.append(W - tile)

        with torch.inference_mode():
            for y in ys:
                for x in xs:
                    tile_bgr = bgr_p[y:y + tile, x:x + tile]
                    inp = _bgr_to_model_input(tile_bgr, self.device)
                    out = self.model(inp)
                    tile_out = _model_output_to_bgr(out)
                    out_acc[y:y + tile, x:x + tile] += tile_out.astype(np.float32) * hann
                    wgt_acc[y:y + tile, x:x + tile] += hann

        out = out_acc / np.maximum(wgt_acc, 1e-6)
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = out[:h, :w]
        if scale_up != 1.0:
            # downscale back to the ORIGINAL pre-upscale size
            orig_h = int(round(h / scale_up))
            orig_w = int(round(w / scale_up))
            out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        return out

    # ----- Public API -----

    def enhance(self, bgr, mode: str = "auto", overlap: int = 64):
        """Enhance ``bgr`` and return an array of the same H,W.

        ``mode`` is one of ``"auto"``, ``"single"``, ``"tiled"``.
        """
        if self.model is None:
            raise RuntimeError("No model weights loaded.")
        if mode not in ("auto", "single", "tiled"):
            raise ValueError(f"Bad mode: {mode}")

        if mode == "auto":
            h, w = bgr.shape[:2]
            mode = "tiled" if max(h, w) > 768 else "single"

        if mode == "single":
            return self._enhance_single(bgr)
        else:
            return self._enhance_tiled(bgr, overlap=overlap)


# ---------- Classical fallback ----------

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
