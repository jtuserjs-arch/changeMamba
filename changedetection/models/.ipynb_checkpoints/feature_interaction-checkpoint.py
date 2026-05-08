import torch
import torch.nn as nn
import torch.nn.functional as F


class CAFIM(nn.Module):
    """
    Change-Aware Feature Interaction Module

    输入:
        f1, f2: [B, C, H, W]
    输出:
        f1_out, f2_out: [B, C, H, W]
    """
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 16)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.context_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.pre_update = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.post_update = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 初始为 0，保证刚开始训练时不破坏原始 backbone 特征
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        summ = f1 + f2

        gate_input = torch.cat([f1, f2, diff, prod], dim=1)
        c_gate = self.channel_gate(gate_input)

        avg_map = torch.mean(diff, dim=1, keepdim=True)
        max_map, _ = torch.max(diff, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        diff_enhanced = diff * c_gate * s_gate
        prod_enhanced = prod * c_gate

        context = self.context_fuse(
            torch.cat([summ, diff_enhanced, prod_enhanced], dim=1)
        )

        f1_out = f1 + self.gamma * self.pre_update(context)
        f2_out = f2 + self.gamma * self.post_update(context)

        return f1_out, f2_out


class MultiScaleCAFIM(nn.Module):
    """
    对 encoder 多尺度特征进行交互。
    stages 使用 0-based index:
        0: 最浅层
        1: 中低层
        2: 中高层
        3: 最高层
    """
    def __init__(self, encoder_dims, stages=(2, 3), reduction=4):
        super().__init__()
        self.stages = set(stages)

        self.blocks = nn.ModuleDict()
        for idx, channels in enumerate(encoder_dims):
            if idx in self.stages:
                self.blocks[str(idx)] = CAFIM(channels, reduction=reduction)

    def forward(self, pre_features, post_features):
        pre_features = list(pre_features)
        post_features = list(post_features)

        for idx in self.stages:
            key = str(idx)
            pre_features[idx], post_features[idx] = self.blocks[key](
                pre_features[idx],
                post_features[idx],
            )

        return pre_features, post_features