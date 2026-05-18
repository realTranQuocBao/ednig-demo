"""Losses used during training.

Matches the original Keras setup (core/losses.py):
- L2 loss on RGB
- Perceptual loss using VGG16 block3_conv3 features (TF) ~ VGG16 features[:16] (PyTorch)
- Generator loss = 1 * perceptual + 10 * L2 (combined function), weighted by 100
  against the Wasserstein critic loss (weight 1) in the original GAN model.
- Wasserstein critic loss = mean(y_true * y_pred); critic outputs are real -> +1, fake -> -1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


# Normalisation used by torchvision pretrained models
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptual(nn.Module):
    """VGG16 feature extractor up to block3_conv3 (Keras name) == features index 15.

    Frozen feature network operating on tensors expected to be in [-1, 1].
    """

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        # block3_conv3 in Keras corresponds to features[15] in torchvision (Conv2d output before relu)
        # We use the same cut-off as the paper.
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)])
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [-1, 1] -> rescale to [0, 1], then ImageNet-normalize
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slice(self.normalize(x))


class PerceptualAndL2Loss(nn.Module):
    """A * perceptual + B * L2 (paper uses A=1, B=10)."""

    def __init__(self, a: float = 1.0, b: float = 10.0, vgg: VGGPerceptual | None = None):
        super().__init__()
        self.vgg = vgg if vgg is not None else VGGPerceptual()
        self.a = a
        self.b = b

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Both pred and target are expected to be in [-1, 1]
        l2 = F.mse_loss(pred, target)
        feat_pred = self.vgg(pred)
        feat_tgt = self.vgg(target)
        perc = F.mse_loss(feat_pred, feat_tgt)
        return self.a * perc + self.b * l2


def wasserstein_loss(pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Wasserstein loss: mean(y_true * pred).

    y_true is +1 for real, -1 for fake (matches original Keras setup).
    """
    return (y_true * pred).mean()
