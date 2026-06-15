"""
models/unet.py — Standard U-Net สำหรับ Wildfire Segmentation

Architecture ตาม Ronneberger et al. (2015) ดัดแปลงสำหรับ:
- Input channels: 27 (multi-modal features)
- Task: binary segmentation (fire / no-fire)

U-Net:
  Encoder: 4 levels × (Conv-BN-ReLU × 2) + MaxPool
  Bottleneck: DoubleConv ที่ resolution ต่ำสุด
  Decoder: 4 levels × (UpSample + Skip concat + DoubleConv)
  Output: Conv(1) → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConvBlock(nn.Module):
    """Standard U-Net double conv: (Conv-BN-ReLU) × 2"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.drop(self.conv1(x)))


class EncoderBlock(nn.Module):
    """Encoder block: DoubleConv + MaxPool"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = DoubleConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + Skip concat + DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net สำหรับ wildfire binary segmentation

    Parameters
    ----------
    in_channels  : 27 (จำนวน feature channels)
    base_filters : 32 (channels ที่ encoder level 1)
    n_levels     : 4 (จำนวน encoder/decoder levels)
    dropout      : dropout rate ใน double conv blocks
    """

    def __init__(self,
                 in_channels: int = 27,
                 base_filters: int = 32,
                 n_levels: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels

        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels = []
        for i in range(n_levels):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(EncoderBlock(ch, out_ch, dropout=dropout))
            enc_channels.append(out_ch)
            ch = out_ch

        bottleneck_ch = base_filters * (2 ** n_levels)
        self.bottleneck = DoubleConvBlock(ch, bottleneck_ch, dropout=dropout)
        ch = bottleneck_ch

        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch  = base_filters * (2 ** i)
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch, dropout=dropout))
            ch = out_ch

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


UNET_CONFIGS = {
    "tiny":   dict(base_filters=16, n_levels=3, dropout=0.1),
    "small":  dict(base_filters=32, n_levels=4, dropout=0.1),
    "medium": dict(base_filters=64, n_levels=4, dropout=0.2),
}

def build_unet(size: str = "small", in_channels: int = 27) -> UNet:
    cfg = UNET_CONFIGS.get(size, UNET_CONFIGS["small"])
    return UNet(in_channels=in_channels, **cfg)


if __name__ == "__main__":
    print("=== U-Net Self-test ===")
    for size in ["tiny", "small", "medium"]:
        model = build_unet(size)
        x     = torch.randn(2, 27, 32, 32)
        out   = model(x)
        n_par = model.count_params()
        print(f"  [{size:6s}] {x.shape} → {out.shape} | Params: {n_par:,}")
        assert out.shape == (2, 1, 32, 32)
    print("\n✅ unet.py OK")
