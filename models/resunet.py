"""
models/resunet.py — Residual U-Net (ResU-Net)

เหมือน U-Net แต่แทน DoubleConv ด้วย Residual Block
ซึ่งช่วยให้:
- Train ได้ลึกกว่าโดยไม่มี vanishing gradient
- เรียนรู้ residual features (ส่วนต่างจาก identity)
- Generalize ดีกว่า standard U-Net

อ้างอิง: Zhang et al. (2018) "Road Extraction by Deep Residual U-Net"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual Block: x → [Conv-BN-ReLU → Conv-BN] + shortcut → ReLU
    shortcut ใช้ 1×1 conv ถ้า in_ch ≠ out_ch
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return self.act(out + residual)


class ResEncoderBlock(nn.Module):
    """ResBlock + MaxPool"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res(x)
        return self.pool(skip), skip


class ResDecoderBlock(nn.Module):
    """Upsample + Skip concat + ResBlock"""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class ResUNet(nn.Module):
    """
    Residual U-Net — encoder + decoder ทั้งหมดใช้ ResBlock

    Parameters
    ----------
    in_channels  : 27
    base_filters : base channel count (ขยาย ×2 ต่อ level)
    n_levels     : จำนวน encoder levels
    dropout      : dropout ใน residual blocks
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
            self.encoders.append(ResEncoderBlock(ch, out_ch, dropout=dropout))
            enc_channels.append(out_ch)
            ch = out_ch

        bottleneck_ch = base_filters * (2 ** n_levels)
        self.bottleneck = ResBlock(ch, bottleneck_ch, dropout=dropout)
        ch = bottleneck_ch

        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch  = base_filters * (2 ** i)
            self.decoders.append(ResDecoderBlock(ch, skip_ch, out_ch, dropout=dropout))
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


RESUNET_CONFIGS = {
    "small":  dict(base_filters=32, n_levels=4, dropout=0.1),
    "medium": dict(base_filters=64, n_levels=4, dropout=0.2),
    "large":  dict(base_filters=64, n_levels=5, dropout=0.2),
}

def build_resunet(size: str = "small", in_channels: int = 27) -> ResUNet:
    cfg = RESUNET_CONFIGS.get(size, RESUNET_CONFIGS["small"])
    return ResUNet(in_channels=in_channels, **cfg)


if __name__ == "__main__":
    print("=== ResU-Net Self-test ===")
    for size in ["small", "medium"]:
        model = build_resunet(size)
        x     = torch.randn(2, 27, 32, 32)
        out   = model(x)
        n_par = model.count_params()
        print(f"  [{size:6s}] {x.shape} → {out.shape} | Params: {n_par:,}")
        assert out.shape == (2, 1, 32, 32)
    print("\n✅ resunet.py OK")
