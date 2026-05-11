import torch
import torch.nn as nn
import torch.nn.functional as F


class CAFIM_MSI_Block(nn.Module):
    """
    Unified CAFIM + Multi-Scale Interaction block

    Inputs:
        pre_feat:      [B, C, H, W]
        post_feat:     [B, C, H, W]
        previous_feat: [B, C, h, w] or None

    Output:
        out:           [B, C, H, W]
    """

    def __init__(self, dim):
        super().__init__()

        # ========== CAFIM (intra-scale change modeling) ==========
        self.diff_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.attn_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.Sigmoid(),
        )

        self.fuse_change = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # ========== Multi-Scale Interaction (inter-scale) ==========
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_feat, post_feat, previous_feat=None):
        # ---------------------------------------------------------
        # 1. CAFIM: change-aware feature interaction
        # ---------------------------------------------------------
        diff = torch.abs(pre_feat - post_feat)
        diff = self.diff_conv(diff)

        attn = self.attn_conv(diff)
        change_feat = attn * post_feat + (1 - attn) * pre_feat

        fused = self.fuse_change(
            torch.cat([change_feat, diff], dim=1)
        )

        # ---------------------------------------------------------
        # 2. Multi-scale interaction (if previous exists)
        # ---------------------------------------------------------
        if previous_feat is not None:
            prev_up = F.interpolate(
                previous_feat,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            q = self.q_conv(fused)
            k = self.k_conv(prev_up)
            v = self.v_conv(prev_up)

            scale_attn = torch.sigmoid(q * k)
            fused = fused + scale_attn * v

        return self.out_proj(fused)