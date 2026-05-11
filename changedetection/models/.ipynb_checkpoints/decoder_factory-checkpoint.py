import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import ResBlock
from .vmamba import Permute, VSSBlock

from .cafim_msi_fusion import CAFIM_MSIFusion
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
        layers.append(
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=hidden_dim,
            )
        )

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


class DynamicGating(nn.Module):
    """
    Dynamic gate for fusing three spatio-temporal modeling branches.

    mode:
        - "pixel": produce [B, 3, H, W]
        - "image": produce [B, 3, 1, 1]
        - "none": average three branches
    """

    def __init__(self, hidden_dim=128, mode="pixel", reduction=4):
        super().__init__()
        
        self.mode = mode
        gate_hidden = max(hidden_dim // reduction, 16)

        if mode == "pixel":
            self.gate_net = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim * 3,
                    out_channels=gate_hidden,
                    kernel_size=1,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=gate_hidden,
                    out_channels=3,
                    kernel_size=1,
                ),
            )

        elif mode == "image":
            self.gate_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    in_channels=hidden_dim * 3,
                    out_channels=gate_hidden,
                    kernel_size=1,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=gate_hidden,
                    out_channels=3,
                    kernel_size=1,
                ),
            )

        elif mode == "none":
            self.gate_net = None

        else:
            raise ValueError(
                f"Unsupported gate mode: {mode}. "
                f"Expected one of ['none', 'image', 'pixel']."
            )

    def forward(self, out_seq, out_cross, out_par):
        if self.mode == "none":
            fused = (out_seq + out_cross + out_par) / 3.0
            gate = torch.ones(
                out_seq.shape[0],
                3,
                out_seq.shape[2],
                out_seq.shape[3],
                device=out_seq.device,
                dtype=out_seq.dtype,
            ) / 3.0
            return fused, gate

        cat_feat = torch.cat([out_seq, out_cross, out_par], dim=1)
        gate_logits = self.gate_net(cat_feat)
        gate = F.softmax(gate_logits, dim=1)

        fused = (
            gate[:, 0:1] * out_seq
            + gate[:, 1:2] * out_cross
            + gate[:, 2:3] * out_par
        )

        return fused, gate

class DynamicChangeDecoder(nn.Module):
    """
    Dynamic STSS decoder with unified CAFIM + Multi-Scale Interaction
    """

    def __init__(
        self,
        *,
        encoder_dims,
        channel_first,
        norm_layer,
        ssm_act_layer,
        mlp_act_layer,
        hidden_dim=128,
        gate_mode="pixel",
        use_ms_interaction=True,
        **kwargs,
    ):
        super().__init__()

        stage_dims = list(reversed(encoder_dims))
        self.hidden_dim = hidden_dim
        self.gate_mode = gate_mode
        self.use_ms_interaction = use_ms_interaction

        self.seq_blocks = nn.ModuleList()
        self.cross_blocks = nn.ModuleList()
        self.par_blocks = nn.ModuleList()
        self.gate_modules = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()

        for stage_dim in stage_dims:
            self.seq_blocks.append(
                make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    in_channels=stage_dim * 2,
                    use_vss=True,
                    **kwargs,
                )
            )

            self.cross_blocks.append(
                make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    in_channels=stage_dim,
                    use_vss=True,
                    **kwargs,
                )
            )

            self.par_blocks.append(
                make_processing_block(
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    hidden_dim=hidden_dim,
                    in_channels=stage_dim,
                    use_vss=True,
                    **kwargs,
                )
            )

            self.gate_modules.append(
                DynamicGating(hidden_dim=hidden_dim, mode=gate_mode)
            )

            self.fuse_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

        # 🔥 CAFIM + 多尺度交互统一模块（stage 间）
        if self.use_ms_interaction:
            self.cafim_msi = nn.ModuleList(
                [
                    CAFIM_MSIFusion(hidden_dim)
                    for _ in range(len(stage_dims) - 1)
                ]
            )

        self.smooth_layers = nn.ModuleList(
            [
                ResBlock(hidden_dim, hidden_dim, stride=1)
                for _ in range(len(stage_dims) - 1)
            ]
        )

        self.latest_gate_weights = []

    def _interleave(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.shape
        out = pre_feat.new_empty(B, C, H, 2 * W)
        out[:, :, :, ::2] = pre_feat
        out[:, :, :, 1::2] = post_feat
        return out

    def _split(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.shape
        out = pre_feat.new_empty(B, C, H, 2 * W)
        out[:, :, :, :W] = pre_feat
        out[:, :, :, W:] = post_feat
        return out

    def forward(self, pre_features, post_features):
        previous = None
        last_gate_weights = None
        self.latest_gate_weights = []

        pre_stages = list(reversed(pre_features))
        post_stages = list(reversed(post_features))

        for i, (pre_feat, post_feat) in enumerate(
            zip(pre_stages, post_stages)
        ):
            width = pre_feat.shape[-1]

            # -------- CAFIM（尺度内）--------
            out_seq = self.seq_blocks[i](
                torch.cat([pre_feat, post_feat], dim=1)
            )

            mixed_cross = self.cross_blocks[i](
                self._interleave(pre_feat, post_feat)
            )
            out_cross = mixed_cross[:, :, :, ::2] + mixed_cross[:, :, :, 1::2]

            mixed_par = self.par_blocks[i](
                self._split(pre_feat, post_feat)
            )
            out_par = mixed_par[:, :, :, :width] + mixed_par[:, :, :, width:]

            current, gate_weights = self.gate_modules[i](
                out_seq, out_cross, out_par
            )
            current = self.fuse_layers[i](current)

            self.latest_gate_weights.append(gate_weights.detach())
            last_gate_weights = gate_weights

            # -------- CAFIM + 多尺度交互（统一）--------
            if previous is not None:
                if self.use_ms_interaction:
                    current = self.cafim_msi[i - 1](current, previous)
                else:
                    current = current + F.interpolate(
                        previous,
                        size=current.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                current = self.smooth_layers[i - 1](current)

            previous = current

        return previous, last_gate_weights


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

        self.fusion_modes_by_stage = [
            tuple(modes) for modes in fusion_modes_by_stage
        ]

        stage_dims = list(reversed(encoder_dims))

        self.stage_blocks = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()

        self.smooth_layers = nn.ModuleList(
            [
                ResBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=1,
                )
                for _ in range(len(stage_dims) - 1)
            ]
        )

        for stage_idx, (stage_dim, fusion_modes) in enumerate(
            zip(stage_dims, self.fusion_modes_by_stage)
        ):
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

            input_chunks = sum(
                self._MODE_CHUNKS[mode] for mode in fusion_modes
            )

            self.fuse_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        kernel_size=1,
                        in_channels=hidden_dim * input_chunks,
                        out_channels=hidden_dim,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )

    def _upsample_add(self, x, y):
        return F.interpolate(
            x,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + y

    def _interleave(self, pre_feat, post_feat):
        batch_size, channels, height, width = pre_feat.shape

        tensor = pre_feat.new_empty(
            batch_size,
            channels,
            height,
            2 * width,
        )

        tensor[:, :, :, ::2] = pre_feat
        tensor[:, :, :, 1::2] = post_feat

        return tensor

    def _split(self, pre_feat, post_feat):
        batch_size, channels, height, width = pre_feat.shape

        tensor = pre_feat.new_empty(
            batch_size,
            channels,
            height,
            2 * width,
        )

        tensor[:, :, :, :width] = pre_feat
        tensor[:, :, :, width:] = post_feat

        return tensor

    def _collect_fusion_features(self, mode, block, pre_feat, post_feat):
        if mode == "cat":
            return [
                block(
                    torch.cat(
                        [pre_feat, post_feat],
                        dim=1,
                    )
                )
            ]

        if mode == "interleave":
            mixed = block(self._interleave(pre_feat, post_feat))
            return [
                mixed[:, :, :, ::2],
                mixed[:, :, :, 1::2],
            ]

        if mode == "split":
            mixed = block(self._split(pre_feat, post_feat))
            width = pre_feat.shape[-1]
            return [
                mixed[:, :, :, :width],
                mixed[:, :, :, width:],
            ]

        raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, pre_features, post_features):
        previous = None

        for stage_idx, (pre_feat, post_feat) in enumerate(
            zip(reversed(pre_features), reversed(post_features))
        ):
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

            current = self.fuse_layers[stage_idx](
                torch.cat(collected, dim=1)
            )

            if previous is not None:
                current = self.smooth_layers[stage_idx - 1](
                    self._upsample_add(previous, current)
                )

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
                    nn.Conv2d(
                        kernel_size=1,
                        in_channels=stage_dim,
                        out_channels=hidden_dim,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for stage_dim in stage_dims[1:]
            ]
        )

        self.smooth_layers = nn.ModuleList(
            [
                ResBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=1,
                )
                for _ in range(len(stage_dims))
            ]
        )

    def _upsample_add(self, x, y):
        return F.interpolate(
            x,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + y

    def forward(self, features):
        stages = list(reversed(features))

        current = self.stage_blocks[0](stages[0])

        for stage_idx, stage_feat in enumerate(stages[1:], start=1):
            current = self._upsample_add(
                current,
                self.transition_layers[stage_idx - 1](stage_feat),
            )
            current = self.smooth_layers[stage_idx - 1](current)
            current = self.stage_blocks[stage_idx](current)

        return self.smooth_layers[-1](current)