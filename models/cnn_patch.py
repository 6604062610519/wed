"""
models/cnn_patch.py — CNN Patch-based Wildfire Predictor

Architecture: lightweight CNN สำหรับ spatial pattern learning
- Input:  (B, 27, 32, 32) — feature patches
- Output: (B, 1, 32, 32)  — fire probability map

Encoder: Conv blocks ที่เพิ่ม channels ขึ้นเรื่อยๆ
Decoder: ConvTranspose layers กลับมา full resolution
Skip connections เชื่อม encoder ↔ decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ─────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConv(nn.Module):
    """Conv-BN-ReLU × 2"""
    def __init__(self, in_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        layers = [
            ConvBNReLU(in_ch, out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ConvBNReLU(out_ch, out_ch),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


# ─────────────────────────────────────────────────────
# CNN Patch Model
# ─────────────────────────────────────────────────────

class CNNPatch(nn.Module):
    """
    Lightweight CNN-Patch สำหรับ 32×32 spatial prediction

    Parameters
    ----------
    in_channels  : จำนวน input channels (27)
    base_filters : จำนวน base filters (ขยาย x2 ต่อ level)
    n_levels     : จำนวน encoder levels (เพิ่มลด depth)
    dropout      : dropout rate
    """

    def __init__(self,
                 in_channels: int = 27,
                 base_filters: int = 32,
                 n_levels: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for i in range(n_levels):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(DoubleConv(ch, out_ch, dropout=dropout))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** n_levels)
        self.bottleneck = DoubleConv(ch, bottleneck_ch, dropout=dropout)
        ch = bottleneck_ch

        # Decoder (with skip connections)
        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            skip_ch = base_filters * (2 ** i)
            self.upsamples.append(
                nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(
                DoubleConv(skip_ch * 2, skip_ch, dropout=dropout)
            )
            ch = skip_ch

        # Output
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → logits: (B, 1, H, W)"""
        # Encode
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode + skip connections
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Pad ถ้า size ไม่ตรงกัน (edge case)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)   # (B, 1, H, W) raw logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────
# Factory + default configs
# ─────────────────────────────────────────────────────

CNN_CONFIGS = {
    "small":  dict(base_filters=16, n_levels=2, dropout=0.1),
    "medium": dict(base_filters=32, n_levels=3, dropout=0.1),
    "large":  dict(base_filters=64, n_levels=4, dropout=0.2),
}

def build_cnn(size: str = "medium", in_channels: int = 27) -> CNNPatch:
    """สร้าง CNN จาก preset size"""
    cfg = CNN_CONFIGS.get(size, CNN_CONFIGS["medium"])
    return CNNPatch(in_channels=in_channels, **cfg)


if __name__ == "__main__":
    print("=== CNN-Patch Self-test ===")
    for size in ["small", "medium", "large"]:
        model = build_cnn(size)
        x     = torch.randn(4, 27, 32, 32)
        out   = model(x)
        n_par = model.count_params()
        print(f"  [{size:6s}] Input: {x.shape} → Output: {out.shape} | Params: {n_par:,}")
        assert out.shape == (4, 1, 32, 32), f"Shape mismatch: {out.shape}"
    print("\n✅ cnn_patch.py OK")
