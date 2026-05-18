"""PyTorch implementation of EDNIG (Encoder-Decoder Network with Illumination Guidance).

Generator: U-Net with Spatial Pyramid Pooling (SPP) at the bottleneck and Swish
activations. The input has 4 channels (RGB + illumination map from BCP).
Output is the enhanced RGB image (range [-1, 1] via tanh).

Discriminator: U-Net encoder + GAP + Dense head, used as a Wasserstein critic.

This mirrors the original Keras implementation in core/networks.py but is
written in modern PyTorch (>= 1.10) and works on both CPU and CUDA.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ConvAct(nn.Module):
    """Conv2D + Swish (or LeakyReLU)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: str = "swish"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2)
        # He-normal-like init to match original Keras kernel_initializer
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act == "swish":
            return swish(x)
        if self.act == "leaky":
            return F.leaky_relu(x, 0.1)
        if self.act == "tanh":
            return torch.tanh(x)
        return x


class ConvBlock(nn.Module):
    """Three stacked Conv+Swish blocks."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.c1 = ConvAct(in_ch, out_ch, k)
        self.c2 = ConvAct(out_ch, out_ch, k)
        self.c3 = ConvAct(out_ch, out_ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c3(self.c2(self.c1(x)))


class SPP(nn.Module):
    """Spatial Pyramid Pooling using 5/9/13 max-pools with stride 1."""

    def __init__(self, ch: int):
        super().__init__()
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(13, stride=1, padding=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.pool13(x), self.pool9(x), self.pool5(x)], dim=1)


class UpBlock(nn.Module):
    """UpSample (2x nearest) + 2x2 Conv with Keras-'same' padding.

    Keras `padding='same'` with kernel=2 pads (left=0, right=1, top=0, bottom=1)
    to keep spatial size constant. PyTorch's `padding=k//2` would pad
    symmetrically and grow the size by 1, so we apply the asymmetric pad
    manually before a `padding=0` conv.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=0)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Asymmetric pad (left=0, right=1, top=0, bottom=1) -> same as Keras 'same'
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return swish(x)


class Generator(nn.Module):
    """EDNIG generator (4-channel input, 3-channel tanh output)."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_filters: int = 96,
        ch_mul: float = 0.125,
    ):
        super().__init__()
        f1 = int(base_filters * ch_mul)
        f2 = int(2 * base_filters * ch_mul)
        f3 = int(4 * base_filters * ch_mul)
        f4 = int(8 * base_filters * ch_mul)
        f5 = int(16 * base_filters * ch_mul)

        # Encoder
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(f3, f4)
        self.drop4 = nn.Dropout2d(0.5)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck + SPP
        self.bottleneck = ConvBlock(f4, f5)
        self.bottleneck_reduce = ConvAct(f5, f4, k=1)
        self.spp = SPP(f4)
        self.bottleneck_merge = ConvAct(f4 * 4, f5, k=1)
        self.drop5 = nn.Dropout2d(0.5)

        # Decoder
        self.up6 = UpBlock(f5, f4)
        self.dec6 = ConvBlock(f4 + f4, f4)
        self.up7 = UpBlock(f4, f3)
        self.dec7 = ConvBlock(f3 + f3, f3)
        self.up8 = UpBlock(f3, f2)
        self.dec8 = ConvBlock(f2 + f2, f2)
        self.up9 = UpBlock(f2, f1)
        self.dec9 = ConvBlock(f1 + f1, f1)

        self.final = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        d4 = self.drop4(c4)
        p4 = self.pool4(d4)

        # Bottleneck + SPP
        b = self.bottleneck(p4)
        b = self.bottleneck_reduce(b)
        b = self.spp(b)
        b = self.bottleneck_merge(b)
        b = self.drop5(b)

        # Decoder with skip connections
        u6 = self.up6(b)
        u6 = torch.cat([d4, u6], dim=1)
        u6 = self.dec6(u6)

        u7 = self.up7(u6)
        u7 = torch.cat([c3, u7], dim=1)
        u7 = self.dec7(u7)

        u8 = self.up8(u7)
        u8 = torch.cat([c2, u8], dim=1)
        u8 = self.dec8(u8)

        u9 = self.up9(u8)
        u9 = torch.cat([c1, u9], dim=1)
        u9 = self.dec9(u9)

        return torch.tanh(self.final(u9))


class Discriminator(nn.Module):
    """Critic with U-Net-like encoder + GAP + Dense head."""

    def __init__(self, in_channels: int = 3, base_filters: int = 96, ch_mul: float = 0.125):
        super().__init__()
        f1 = int(base_filters * ch_mul)
        f2 = int(2 * base_filters * ch_mul)
        f3 = int(4 * base_filters * ch_mul)
        f4 = int(8 * base_filters * ch_mul)
        f5 = int(16 * base_filters * ch_mul)

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.b1 = block(in_channels, f1)
        self.p1 = nn.MaxPool2d(2)
        self.b2 = block(f1, f2)
        self.p2 = nn.MaxPool2d(2)
        self.b3 = block(f2, f3)
        self.p3 = nn.MaxPool2d(2)
        self.b4 = block(f3, f4)
        self.drop4 = nn.Dropout2d(0.5)
        self.p4 = nn.MaxPool2d(2)
        self.b5 = block(f4, f5)
        self.drop5 = nn.Dropout2d(0.5)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f5, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p1(self.b1(x))
        x = self.p2(self.b2(x))
        x = self.p3(self.b3(x))
        x = self.drop4(self.b4(x))
        x = self.p4(x)
        x = self.drop5(self.b5(x))
        return self.head(x)
