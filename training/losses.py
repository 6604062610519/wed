"""
training/losses.py — Loss functions สำหรับ wildfire binary segmentation

FocalLoss    : จัดการ class imbalance (fire rate ~2.3%) ด้วย hard example mining
DiceLoss     : จาก semantic segmentation, ดีกับ spatial overlap
CombinedLoss : Focal + Dice (แนะนำสำหรับ U-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss = -(1-p_t)^gamma * log(p_t)
    γ=2 ทำให้ model focus บน hard examples (missed fire pixels)
    α จัดการ class imbalance (α_fire > 0.5)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        p       = torch.sigmoid(logits)
        p_t     = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce_loss     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss         = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - 2|X∩Y| / (|X| + |Y|)
    ดีสำหรับ spatial overlap — ไม่ขึ้นกับ class imbalance โดยธรรมชาติ
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        prob      = torch.sigmoid(logits)
        prob_flat = prob.reshape(-1)
        tgt_flat  = targets.reshape(-1)

        intersection = (prob_flat * tgt_flat).sum()
        dice = (2 * intersection + self.smooth) / \
               (prob_flat.sum() + tgt_flat.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Loss = w_focal * FocalLoss + w_dice * DiceLoss
    ดีที่สุดสำหรับ spatial binary segmentation ที่มี class imbalance
    """

    def __init__(self, focal_weight: float = 0.6, dice_weight: float = 0.4,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal        = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice         = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self.focal(logits, targets) +
                self.dice_weight  * self.dice(logits, targets))


class BCEWithPosWeight(nn.Module):
    """Standard BCE แต่ weighted — เป็น baseline loss"""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


LOSS_REGISTRY = {
    "focal":    lambda cfg: FocalLoss(
                    alpha=cfg.get("focal_alpha", 0.75),
                    gamma=cfg.get("focal_gamma", 2.0)),
    "dice":     lambda cfg: DiceLoss(),
    "combined": lambda cfg: CombinedLoss(
                    focal_weight=cfg.get("focal_weight", 0.6),
                    dice_weight=cfg.get("dice_weight", 0.4),
                    focal_alpha=cfg.get("focal_alpha", 0.75),
                    focal_gamma=cfg.get("focal_gamma", 2.0)),
    "bce":      lambda cfg: BCEWithPosWeight(
                    pos_weight=cfg.get("pos_weight", 10.0)),
}


def build_loss(name: str, cfg: dict = None) -> nn.Module:
    """Factory function สำหรับสร้าง loss จากชื่อ"""
    cfg = cfg or {}
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](cfg)


if __name__ == "__main__":
    print("=== Testing Losses ===")
    B, H, W = 4, 32, 32
    logits  = torch.randn(B, H, W)
    targets = (torch.rand(B, H, W) > 0.95).float()

    for name, loss_fn in [
        ("FocalLoss",    FocalLoss()),
        ("DiceLoss",     DiceLoss()),
        ("CombinedLoss", CombinedLoss()),
        ("BCE+Weight",   BCEWithPosWeight()),
    ]:
        val = loss_fn(logits, targets)
        print(f"  {name:20s}: {val.item():.4f}")

    print("\n✅ losses.py OK")
