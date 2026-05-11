import torch
import torch.nn as nn
import torch.nn.functional as F


class CAFIM_MSIFusion(nn.Module):
    """
    Unified CAFIM (intra-scale) + Multi-Scale Interaction (inter-scale) fusion
    current  : current stage change-aware feature
    previous : higher-level semantic feature
    """

    def __init__(self, dim):
        super().__init__()

        # 用高层语义调制低层（scale-aware）
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)

        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, current, previous):
        """
        current  : [B, C, H, W]
        previous : [B, C, h, w]
        """

        if previous is None:
            return self.proj(current)

        prev_up = F.interpolate(
            previous,
            size=current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        q = self.q_conv(current)
        k = self.k_conv(prev_up)
        v = self.v_conv(prev_up)

        attn = torch.sigmoid(q * k)
        out = current + attn * v

        return self.proj(out)