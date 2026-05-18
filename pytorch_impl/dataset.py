"""LOL dataset (paired low/high images) for EDNIG training.

The LOL dataset directory layout used here:

    <root>/our485/low/<name>.png   <- low-light input
    <root>/our485/high/<name>.png  <- ground-truth normal-light
    <root>/eval15/low/...
    <root>/eval15/high/...

Each item returned is:
    inp  : Tensor [4, H, W] in [-1, 1] (RGB + BCP illumination)
    tgt  : Tensor [3, H, W] in [-1, 1]

with optional horizontal flip data augmentation.
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
    """BGR uint8 HxWx3 -> tensor [3,H,W] in [-1, 1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _to_tensor_illum(illum: np.ndarray) -> torch.Tensor:
    """Float HxW in [0,1] -> tensor [1,H,W] in [-1, 1]."""
    t = (illum.astype(np.float32) - 0.5) * 2.0
    return torch.from_numpy(t).unsqueeze(0).contiguous()


def make_input_4ch(rgb_tensor: torch.Tensor, illum_tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat([rgb_tensor, illum_tensor], dim=0)


class LOLDataset(Dataset):
    """Paired low/high light dataset with random horizontal flip + resize."""

    def __init__(self, root: str, split: str = "train", img_size: int = 512, augment: bool = True):
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

    def __len__(self) -> int:
        return len(self.low_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_bgr = cv2.imread(self.low_paths[idx], cv2.IMREAD_COLOR)
        high_bgr = cv2.imread(self.high_paths[idx], cv2.IMREAD_COLOR)
        if low_bgr is None or high_bgr is None:
            raise RuntimeError(f"Failed to read pair at index {idx}")

        sz = (self.img_size, self.img_size)
        low_bgr = cv2.resize(low_bgr, sz, interpolation=cv2.INTER_AREA)
        high_bgr = cv2.resize(high_bgr, sz, interpolation=cv2.INTER_AREA)

        if self.augment and random.random() < 0.5:
            low_bgr = cv2.flip(low_bgr, 1)
            high_bgr = cv2.flip(high_bgr, 1)

        illum = estimate_illumination(low_bgr)

        inp_rgb = _to_tensor_rgb(low_bgr)
        inp_illum = _to_tensor_illum(illum)
        inp = make_input_4ch(inp_rgb, inp_illum)
        tgt = _to_tensor_rgb(high_bgr)
        return inp, tgt
