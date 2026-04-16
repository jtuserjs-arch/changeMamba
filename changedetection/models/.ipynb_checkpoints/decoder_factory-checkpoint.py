import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import ResBlock
from .vmamba import Permute, VSSBlock

def make_processing_block(
    *,
    channel_first,
    norm_layer,
    ssm_act_layer,
    mlp_act_layer,
    hidden_dim=128,
    in_channels=None,
    use_vss=True,
    **kwargs,
):
    layers = []
    if in_channels is not None:
        layers.append(nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=hidden_dim))

    if use_vss:
        layers.extend(
            [
                Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
                VSSBlock(
                    hidden_dim=hidden_dim,
                    drop_path=0.1,
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=kwargs["ssm_d_state"],
                    ssm_ratio=kwargs["ssm_ratio"],
                    ssm_dt_rank=kwargs["ssm_dt_rank"],
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=kwargs["ssm_conv"],
                    ssm_conv_bias=kwargs["ssm_conv_bias"],
                    ssm_drop_rate=kwargs["ssm_drop_rate"],
                    ssm_init=kwargs["ssm_init"],
                    forward_type=kwargs["forward_type"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=kwargs["mlp_drop_rate"],
                    gmlp=kwargs["gmlp"],
                    use_checkpoint=kwargs["use_checkpoint"],
                ),
                Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
            ]
        )
    else:
        layers.extend(
            [
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )

    return nn.Sequential(*layers)

# 在 import 部分之后，类定义之前添加
class DynamicGating(nn.Module):
    """动态门控模块，自适应融合三种时空建模输出"""
    def __init__(self, hidden_dim, mode='pixel'):
        super().__init__()
        self.mode = mode
        if mode == 'image':
            self.gate_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim * 3, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3)
            )
        else:  # pixel
            self.gate_net = nn.Conv2d(hidden_dim * 3, 3, kernel_size=1)
    
    def forward(self, out_seq, out_cross, out_par):
        cat_feat = torch.cat([out_seq, out_cross, out_par], dim=1)
        gate = self.gate_net(cat_feat)
        gate = F.softmax(gate, dim=1)
        if self.mode == 'image':
            gate = gate.view(-1, 3, 1, 1)
        fused = gate[:,0:1]*out_seq + gate[:,1:2]*out_cross + gate[:,2:3]*out_par
        return fused, gate
class DynamicChangeDecoder(nn.Module):
    """动态门控 + 不确定性估计的变化解码器"""
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer,
                 hidden_dim=128, gate_mode='pixel', use_uncertainty=True, **kwargs):
        super().__init__()
        stage_dims = list(reversed(encoder_dims))
        self.gate_mode = gate_mode
        self.use_uncertainty = use_uncertainty
        
        # 为每个 stage 创建三个分支：顺序(seq)、交叉(cross)、并行(par)
        self.seq_blocks = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()
        self.par_blocks = nn.ModuleList()
        self.gate_modules = nn.ModuleList()
        
        for stage_idx, stage_dim in enumerate(stage_dims):
            # 顺序分支：使用原始的 cat 模式（双时相拼接）
            seq_block = make_processing_block(
                channel_first=channel_first, norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer,
                hidden_dim=hidden_dim, in_channels=stage_dim*2, use_vss=True, **kwargs)
            # 交叉分支：使用 interleave 模式
            cross_block = make_processing_block(
                channel_first=channel_first, norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer,
                hidden_dim=hidden_dim, in_channels=stage_dim, use_vss=True, **kwargs)
            # 并行分支：使用 split 模式
            par_block = make_processing_block(
                channel_first=channel_first, norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer,
                hidden_dim=hidden_dim, in_channels=stage_dim, use_vss=True, **kwargs)
            self.seq_blocks.append(seq_block)
            self.cross_blocks.append(cross_block)
            self.par_blocks.append(par_block)
            self.gate_modules.append(DynamicGating(hidden_dim, mode=gate_mode))
        
        # 融合后的平滑层和上采样
        self.fuse_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
            ) for _ in range(len(stage_dims))
        ])
        self.smooth_layers = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim, stride=1) for _ in range(len(stage_dims)-1)
        ])
        
        # 不确定性头（可选）
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 2, kernel_size=1),
                nn.Softplus()  # 输出证据
            )
    
    def _interleave(self, pre, post):
        B,C,H,W = pre.shape
        out = pre.new_empty(B, C, H, 2*W)
        out[:,:,:,::2] = pre
        out[:,:,:,1::2] = post
        return out
    
    def _split(self, pre, post):
        B,C,H,W = pre.shape
        out = pre.new_empty(B, C, H, 2*W)
        out[:,:,:,:W] = pre
        out[:,:,:,W:] = post
        return out
    
    def forward(self, pre_features, post_features):
        previous = None
        pre_rev = list(reversed(pre_features))
        post_rev = list(reversed(post_features))

        for i, (pre_feat, post_feat) in enumerate(zip(pre_rev, post_rev)):
            # 三个分支
            out_seq = self.seq_blocks[i](torch.cat([pre_feat, post_feat], dim=1))
            out_cross = self.cross_blocks[i](self._interleave(pre_feat, post_feat))
            out_cross = out_cross[:, :, :, ::2] + out_cross[:, :, :, 1::2]
            out_par = self.par_blocks[i](self._split(pre_feat, post_feat))
            out_par = out_par[:, :, :, :pre_feat.shape[-1]] + out_par[:, :, :, pre_feat.shape[-1]:]

            # 动态门控融合
            fused, gate_weights = self.gate_modules[i](out_seq, out_cross, out_par)
            fused = self.fuse_layers[i](fused)

            if previous is not None:
                fused = self._upsample_add(previous, fused)
                fused = self.smooth_layers[i-1](fused)
            previous = fused

        # 最终特征图
        change_feat = previous

        if self.use_uncertainty:
            evidence = self.uncertainty_head(change_feat)   # [B,2,H,W]
            alpha = evidence + 1
            uncertainty = 2 / (alpha.sum(dim=1, keepdim=True))  # [B,1,H,W]
            return change_feat, uncertainty, gate_weights
        else:
            return change_feat, None, gate_weights
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear') + y

class HierarchicalChangeDecoder(nn.Module):
    _MODE_CHUNKS = {
        "cat": 1,
        "interleave": 2,
        "split": 2,
    }

    def __init__(
        self,
        *,
        encoder_dims,
        channel_first,
        norm_layer,
        ssm_act_layer,
        mlp_act_layer,
        fusion_modes_by_stage,
        use_vss_by_stage,
        hidden_dim=128,
        **kwargs,
    ):
        super().__init__()
        self.fusion_modes_by_stage = [tuple(modes) for modes in fusion_modes_by_stage]
        stage_dims = list(reversed(encoder_dims))

        self.stage_blocks = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList(
            [ResBlock(in_channels=hidden_dim, out_channels=hidden_dim, stride=1) for _ in range(len(stage_dims) - 1)]
        )

        for stage_idx, (stage_dim, fusion_modes) in enumerate(zip(stage_dims, self.fusion_modes_by_stage)):
            stage_block = nn.ModuleDict()
            for mode in fusion_modes:
                in_channels = stage_dim * 2 if mode == "cat" else stage_dim
                stage_block[mode] = make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    in_channels=in_channels,
                    use_vss=use_vss_by_stage[stage_idx],
                    **kwargs,
                )
            self.stage_blocks.append(stage_block)

            input_chunks = sum(self._MODE_CHUNKS[mode] for mode in fusion_modes)
            self.fuse_layers.append(
                nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=hidden_dim * input_chunks, out_channels=hidden_dim),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear") + y

    def _interleave(self, pre_feat, post_feat):
        batch_size, channels, height, width = pre_feat.shape
        tensor = pre_feat.new_empty(batch_size, channels, height, 2 * width)
        tensor[:, :, :, ::2] = pre_feat
        tensor[:, :, :, 1::2] = post_feat
        return tensor

    def _split(self, pre_feat, post_feat):
        batch_size, channels, height, width = pre_feat.shape
        tensor = pre_feat.new_empty(batch_size, channels, height, 2 * width)
        tensor[:, :, :, :width] = pre_feat
        tensor[:, :, :, width:] = post_feat
        return tensor

    def _collect_fusion_features(self, mode, block, pre_feat, post_feat):
        if mode == "cat":
            return [block(torch.cat([pre_feat, post_feat], dim=1))]
        if mode == "interleave":
            mixed = block(self._interleave(pre_feat, post_feat))
            return [mixed[:, :, :, ::2], mixed[:, :, :, 1::2]]
        if mode == "split":
            mixed = block(self._split(pre_feat, post_feat))
            width = pre_feat.shape[-1]
            return [mixed[:, :, :, :width], mixed[:, :, :, width:]]
        raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, pre_features, post_features):
        previous = None
        for stage_idx, (pre_feat, post_feat) in enumerate(zip(reversed(pre_features), reversed(post_features))):
            collected = []
            for mode in self.fusion_modes_by_stage[stage_idx]:
                collected.extend(
                    self._collect_fusion_features(
                        mode,
                        self.stage_blocks[stage_idx][mode],
                        pre_feat,
                        post_feat,
                    )
                )

            current = self.fuse_layers[stage_idx](torch.cat(collected, dim=1))
            if previous is not None:
                current = self.smooth_layers[stage_idx - 1](self._upsample_add(previous, current))
            previous = current
        return previous


class HierarchicalSemanticDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_dims,
        channel_first,
        norm_layer,
        ssm_act_layer,
        mlp_act_layer,
        hidden_dim=128,
        **kwargs,
    ):
        super().__init__()
        stage_dims = list(reversed(encoder_dims))

        self.stage_blocks = nn.ModuleList(
            [
                make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    in_channels=stage_dims[0],
                    **kwargs,
                )
            ]
        )
        self.stage_blocks.extend(
            [
                make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    **kwargs,
                )
                for _ in stage_dims[1:]
            ]
        )

        self.transition_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(kernel_size=1, in_channels=stage_dim, out_channels=hidden_dim),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for stage_dim in stage_dims[1:]
            ]
        )
        self.smooth_layers = nn.ModuleList(
            [ResBlock(in_channels=hidden_dim, out_channels=hidden_dim, stride=1) for _ in range(len(stage_dims))]
        )

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear") + y

    def forward(self, features):
        stages = list(reversed(features))
        current = self.stage_blocks[0](stages[0])

        for stage_idx, stage_feat in enumerate(stages[1:], start=1):
            current = self._upsample_add(current, self.transition_layers[stage_idx - 1](stage_feat))
            current = self.smooth_layers[stage_idx - 1](current)
            current = self.stage_blocks[stage_idx](current)

        return self.smooth_layers[-1](current)
