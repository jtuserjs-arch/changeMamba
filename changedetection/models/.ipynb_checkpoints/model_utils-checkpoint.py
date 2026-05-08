import torch.nn as nn
from .vmamba import LayerNorm2d

# Shared norm/act layer registries used across all ChangeMamba model files.
_NORMLAYERS = dict(
    ln=nn.LayerNorm,
    ln2d=LayerNorm2d,
    bn=nn.BatchNorm2d,
)

_ACTLAYERS = dict(
    silu=nn.SiLU,
    gelu=nn.GELU,
    relu=nn.ReLU,
    sigmoid=nn.Sigmoid,
)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Cross-attention for two feature maps
    f1, f2: [B, C, H, W]
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        # QKV projection
        self.qkv1 = nn.Conv2d(dim, dim*3, 1)
        self.qkv2 = nn.Conv2d(dim, dim*3, 1)

        # output projection
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape

        # reshape for multi-head attention
        def split_qkv(x, qkv_layer):
            qkv = qkv_layer(x).reshape(B, 3, self.num_heads, C//self.num_heads, H*W)
            return qkv[:,0], qkv[:,1], qkv[:,2]

        q1, k1, v1 = split_qkv(f1, self.qkv1)
        q2, k2, v2 = split_qkv(f2, self.qkv2)

        # f1 attends f2
        attn1 = torch.einsum('bhcd,bhce->bhde', q1, k2) * self.scale
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.einsum('bhde,bhce->bhcd', attn1, v2).reshape(B, C, H, W)
        out1 = self.proj1(out1) + f1  # residual

        # f2 attends f1
        attn2 = torch.einsum('bhcd,bhce->bhde', q2, k1) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.einsum('bhde,bhce->bhcd', attn2, v1).reshape(B, C, H, W)
        out2 = self.proj2(out2) + f2

        return out1, out2