"""LOL dataset loader for EDNIG training, with data augmentation.

Directory layout:
    <root>/our485/low/<name>.png
    <root>/our485/high/<name>.png
    <root>/eval15/low/...
    <root>/eval15/high/...

Augmentations (training split only, disable with augment=False):
  1. Random multi-scale crop      : crop a square region of size
                                    s = uniform([min_scale, max_scale]) * min(H, W),
                                    then resize to img_size.
                                    (Mirrors the original Keras "crop_sizes"
                                    intention of [0.4, 0.5, 0.6, 0.7, 0.8] of min(H,W).)
  2. Horizontal flip 50%
  3. Vertical flip 20% (mild)
  4. 90/180/270 rotation 30%
  5. Photometric jitter on the LOW image only (brightness +/- 10%, contrast +/- 10%)
     — applied at low probability (15%) and NEVER on the high (target) image, so the
     enhancement target stays the ground-truth distribution.

The illumination guidance map (BCP) is recomputed from the *augmented* low image
so it stays consistent with what the model receives.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .bcp import estimate_illumination


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _list_images(folder: str) -> List[str]:
    p = Path(folder)
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)


def _to_tensor_rgb(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _to_tensor_illum(illum: np.ndarray) -> torch.Tensor:
    t = (illum.astype(np.float32) - 0.5) * 2.0
    return torch.from_numpy(t).unsqueeze(0).contiguous()


def make_input_4ch(rgb_tensor: torch.Tensor, illum_tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat([rgb_tensor, illum_tensor], dim=0)


def _random_square_crop(low: np.ndarray, high: np.ndarray,
                       min_scale: float, max_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a random square from both images at the SAME location."""
    h, w = low.shape[:2]
    min_side = min(h, w)
    scale = random.uniform(min_scale, max_scale)
    size = max(8, int(min_side * scale))
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return low[y:y + size, x:x + size], high[y:y + size, x:x + size]


def _random_rotate_90(low: np.ndarray, high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Random 0/90/180/270 rotation."""
    k = random.randint(0, 3)
    if k == 0:
        return low, high
    return np.rot90(low, k=k).copy(), np.rot90(high, k=k).copy()


def _photometric_jitter_low(low: np.ndarray,
                            brightness_range: float = 0.10,
                            contrast_range: float = 0.10) -> np.ndarray:
    """Mild brightness/contrast jitter on the LOW image only.

    Done in float32 then clipped back to uint8 to keep the pipeline simple.
    Applied only sometimes (caller controls probability).
    """
    alpha = 1.0 + random.uniform(-contrast_range, contrast_range)  # contrast
    beta = 255.0 * random.uniform(-brightness_range, brightness_range)  # brightness
    out = low.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


class LOLDataset(Dataset):
    """Paired low/high light dataset with comprehensive data augmentation."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 512,
        augment: bool = True,
        crop_min: float = 0.5,
        crop_max: float = 1.0,
        flip_prob: float = 0.5,
        vflip_prob: float = 0.0,    # off by default — was 0.2, caused weird outputs
        rot90_prob: float = 0.0,    # off by default — was 0.3
        jitter_prob: float = 0.0,   # off by default — was 0.15, "âm bản" artifact
    ):
        if split == "train":
            sub = "our485"
        elif split in ("val", "eval", "test"):
            sub = "eval15"
        else:
            raise ValueError(f"Unknown split: {split}")
        self.low_dir = os.path.join(root, sub, "low")
        self.high_dir = os.path.join(root, sub, "high")
        if not (os.path.isdir(self.low_dir) and os.path.isdir(self.high_dir)):
            raise FileNotFoundError(f"Expected {sub}/low and {sub}/high under {root}")

        self.low_paths = _list_images(self.low_dir)
        self.high_paths = _list_images(self.high_dir)
        if len(self.low_paths) != len(self.high_paths):
            raise RuntimeError(
                f"low/high count mismatch: {len(self.low_paths)} vs {len(self.high_paths)}"
            )

        self.img_size = img_size
        self.augment = augment
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.flip_prob = flip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.jitter_prob = jitter_prob

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_bgr = cv2.imread(self.low_paths[idx], cv2.IMREAD_COLOR)
        high_bgr = cv2.imread(self.high_paths[idx], cv2.IMREAD_COLOR)
        if low_bgr is None or high_bgr is None:
            raise RuntimeError(f"Failed to read pair at index {idx}")

        if self.augment:
            # 1. Random square crop at the same location (multi-scale)
            low_bgr, high_bgr = _random_square_crop(low_bgr, high_bgr,
                                                    self.crop_min, self.crop_max)
            # 2. Horizontal flip
            if random.random() < self.flip_prob:
                low_bgr = cv2.flip(low_bgr, 1)
                high_bgr = cv2.flip(high_bgr, 1)
            # 3. Vertical flip (mild)
            if random.random() < self.vflip_prob:
                low_bgr = cv2.flip(low_bgr, 0)
                high_bgr = cv2.flip(high_bgr, 0)
            # 4. 90-degree rotation (random k in {0..3} with rot90_prob)
            if random.random() < self.rot90_prob:
                low_bgr, high_bgr = _random_rotate_90(low_bgr, high_bgr)
            # 5. Photometric jitter on LOW only (so target distribution stays fixed)
            if random.random() < self.jitter_prob:
                low_bgr = _photometric_jitter_low(low_bgr)

        # Final resize to img_size for both
        sz = (self.img_size, self.img_size)
        low_bgr = cv2.resize(low_bgr, sz, interpolation=cv2.INTER_AREA)
        high_bgr = cv2.resize(high_bgr, sz, interpolation=cv2.INTER_AREA)

        # Compute illumination map AFTER augmentation so it matches the input
        illum = estimate_illumination(low_bgr)

        inp_rgb = _to_tensor_rgb(low_bgr)
        inp_illum = _to_tensor_illum(illum)
        inp = make_input_4ch(inp_rgb, inp_illum)
        tgt = _to_tensor_rgb(high_bgr)
        return inp, tgt
