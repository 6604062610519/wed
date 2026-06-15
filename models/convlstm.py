"""
models/convlstm.py — ConvLSTM สำหรับ Temporal Wildfire Prediction

Architecture ใช้ ConvLSTM cell เพื่อ capture temporal dynamics ของสภาพอากาศ/พืชพรรณ
Input:  (B, T, C, H, W) — T เดือนย้อนหลัง
Output: (B, 1, H, W)    — fire probability map ของเดือนถัดไป

ConvLSTM cell อ้างอิง: Shi et al. (2015) "Convolutional LSTM Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell — แทนที่ matrix multiplication ด้วย convolution
    ทำให้ LSTM capture spatial-temporal patterns ได้

    State: (h, c) — hidden state และ cell state แต่ละอันเป็น (B, hidden_ch, H, W)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W,
                            device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W,
                            device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, return_all: bool = False):
        super().__init__()
        self.cell       = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.return_all = return_all

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        state = None
        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t], state)
            state = (h, c)
            if self.return_all:
                outputs.append(h)

        if self.return_all:
            return torch.stack(outputs, dim=1)
        return h


class ConvLSTMWildfire(nn.Module):
    """
    Stacked ConvLSTM → Convolutional decoder → binary prediction

    Parameters
    ----------
    in_channels     : feature channels (27)
    hidden_channels : ConvLSTM hidden channels per layer
    n_lstm_layers   : จำนวน ConvLSTM layers
    kernel_size     : convolution kernel ใน LSTM cells
    dropout         : spatial dropout ระหว่าง LSTM layers
    """

    def __init__(self,
                 in_channels: int = 27,
                 hidden_channels: int = 64,
                 n_lstm_layers: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_lstm_layers = n_lstm_layers

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.lstm_layers = nn.ModuleList()
        ch = hidden_channels
        for i in range(n_lstm_layers):
            return_all = (i < n_lstm_layers - 1)
            self.lstm_layers.append(
                ConvLSTMLayer(ch, hidden_channels, kernel_size, return_all=return_all)
            )
            ch = hidden_channels

        self.drop = nn.Dropout2d(dropout)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x_proj = []
        for t in range(T):
            x_proj.append(self.input_proj(x[:, t]))
        x = torch.stack(x_proj, dim=1)

        for i, lstm_layer in enumerate(self.lstm_layers):
            x = lstm_layer(x)
            if x.dim() == 4:
                break
            B2, T2, ch2, H2, W2 = x.shape
            x = self.drop(x.reshape(B2 * T2, ch2, H2, W2)).reshape(B2, T2, ch2, H2, W2)

        return self.decoder(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CONVLSTM_CONFIGS = {
    "small":  dict(hidden_channels=32,  n_lstm_layers=1, dropout=0.1),
    "medium": dict(hidden_channels=64,  n_lstm_layers=2, dropout=0.1),
    "large":  dict(hidden_channels=128, n_lstm_layers=2, dropout=0.2),
}

def build_convlstm(size: str = "small", in_channels: int = 27) -> ConvLSTMWildfire:
    cfg = CONVLSTM_CONFIGS.get(size, CONVLSTM_CONFIGS["small"])
    return ConvLSTMWildfire(in_channels=in_channels, **cfg)


if __name__ == "__main__":
    print("=== ConvLSTM Self-test ===")
    for size in ["small", "medium"]:
        model = build_convlstm(size)
        T     = 3
        x     = torch.randn(2, T, 27, 32, 32)
        out   = model(x)
        n_par = model.count_params()
        print(f"  [{size:6s}] {x.shape} → {out.shape} | Params: {n_par:,}")
        assert out.shape == (2, 1, 32, 32), f"Shape mismatch: {out.shape}"
    print("\n✅ convlstm.py OK")
