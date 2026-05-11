import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleInteraction(nn.Module):
    """
    High-level change semantics guide low-level details
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, feat_low, feat_high):
        """
        feat_low  : high-resolution (detail)
        feat_high : low-resolution (semantic)
        """

        # 上采样高层特征
        feat_high = F.interpolate(
            feat_high,
            size=feat_low.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ========== Channel interaction ==========
        ch_weight = self.channel_attn(feat_high)
        feat_low = feat_low * ch_weight

        # ========== Spatial interaction ==========
        avg_map = torch.mean(feat_high, dim=1, keepdim=True)
        max_map, _ = torch.max(feat_high, dim=1, keepdim=True)
        sp_weight = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))

        feat_low = feat_low * sp_weight

        # ========== Fusion ==========
        out = self.fuse(feat_low + feat_high)

        return out