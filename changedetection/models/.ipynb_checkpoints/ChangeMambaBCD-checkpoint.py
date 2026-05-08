import torch
import torch.nn as nn
import torch.nn.functional as F

from .builders import (
    build_backbone,
    resolve_decoder_components,
    build_head,
    resize_to_input,
)
from .ChangeDecoder import ChangeDecoder
from .decoder_factory import DynamicChangeDecoder


def _normalize_stages(stages, num_stages):
    """
    Convert user-provided stages to a valid sorted tuple.

    stages:
        int / list / tuple / comma-separated string.
        0-based stage index:
            0: shallowest
            3: deepest
    """
    if stages is None:
        stages = (2, 3)

    if isinstance(stages, int):
        stages = (stages,)
    elif isinstance(stages, str):
        stages = stages.replace(",", " ").split()
        stages = tuple(int(s) for s in stages)
    else:
        stages = tuple(int(s) for s in stages)

    valid = []
    for stage in stages:
        if 0 <= stage < num_stages and stage not in valid:
            valid.append(stage)

    return tuple(sorted(valid))


class CAFIM(nn.Module):
    """
    Change-Aware Feature Interaction Module.

    Input:
        f1, f2: [B, C, H, W]
    Output:
        f1_out, f2_out: [B, C, H, W]

    Design:
        diff = |f1 - f2|     : change discrepancy
        prod = f1 * f2       : temporal correlation
        summ = f1 + f2       : shared semantic context
        channel gate + spatial gate enhance change-aware interaction.
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

        # Zero-init residual scale. This makes CAFIM start as an identity mapping,
        # so it is safer when loading / fine-tuning from the original model.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        summ = f1 + f2

        gate_input = torch.cat([f1, f2, diff, prod], dim=1)
        channel_weight = self.channel_gate(gate_input)

        avg_map = torch.mean(diff, dim=1, keepdim=True)
        max_map, _ = torch.max(diff, dim=1, keepdim=True)
        spatial_weight = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))

        diff_enhanced = diff * channel_weight * spatial_weight
        prod_enhanced = prod * channel_weight

        context = self.context_fuse(
            torch.cat([summ, diff_enhanced, prod_enhanced], dim=1)
        )

        f1_out = f1 + self.gamma * self.pre_update(context)
        f2_out = f2 + self.gamma * self.post_update(context)

        return f1_out, f2_out


class MultiScaleCAFIM(nn.Module):
    """
    Multi-scale CAFIM wrapper.

    encoder_dims:
        channel dimensions of encoder outputs, e.g. [96, 192, 384, 768]

    stages:
        0-based stage indices.
        Recommended default is (2, 3), which enhances high-level semantic stages.
    """

    def __init__(self, encoder_dims, stages=(2, 3), reduction=4):
        super().__init__()

        encoder_dims = list(encoder_dims)
        self.stages = _normalize_stages(stages, len(encoder_dims))

        self.blocks = nn.ModuleDict()
        for idx in self.stages:
            self.blocks[str(idx)] = CAFIM(
                channels=encoder_dims[idx],
                reduction=reduction,
            )

    def forward(self, pre_features, post_features):
        pre_features = list(pre_features)
        post_features = list(post_features)

        for idx in self.stages:
            pre_features[idx], post_features[idx] = self.blocks[str(idx)](
                pre_features[idx],
                post_features[idx],
            )

        return pre_features, post_features


class ChangeMambaBCD(nn.Module):
    """
    BCD model for ChangeMamba.

    Main switches:
        gate_mode:
            "none"  : original ChangeDecoder
            "image" : image-level dynamic gate
            "pixel" : pixel-level dynamic gate

        interaction_mode:
            "none"  : no feature interaction before decoder
            "cafim" : multi-scale change-aware feature interaction

        interaction_stages:
            0-based encoder stages used by CAFIM.
            Recommended:
                (2, 3) for stable high-level interaction
                (1, 2, 3) for stronger but slightly noisier interaction

        use_uncertainty:
            False: normal binary change detection
            True : add evidential uncertainty head when return_aux=True
    """

    def __init__(
        self,
        pretrained,
        gate_mode="pixel",
        use_uncertainty=False,
        dropout_rate=0.0,
        interaction_mode="cafim",
        interaction_stages=(2, 3),
        interaction_reduction=4,
        **kwargs,
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        self.encoder_dims = list(self.encoder.dims)

        # -------------------------
        # Feature interaction
        # -------------------------
        self.interaction_mode = interaction_mode

        if interaction_mode in [None, "none", "None"]:
            self.feature_interaction = None
        elif interaction_mode == "cafim":
            self.feature_interaction = MultiScaleCAFIM(
                encoder_dims=self.encoder_dims,
                stages=interaction_stages,
                reduction=interaction_reduction,
            )
        else:
            raise ValueError(
                f"Unsupported interaction_mode: {interaction_mode}. "
                f"Expected one of ['none', 'cafim']."
            )

        # -------------------------
        # Decoder + dynamic gate
        # -------------------------
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = (
            resolve_decoder_components(kwargs)
        )

        self.gate_mode = gate_mode
        self.use_uncertainty = use_uncertainty

        if gate_mode == "none":
            self.decoder = ChangeDecoder(
                encoder_dims=self.encoder.dims,
                channel_first=self.encoder.channel_first,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **clean_kwargs,
            )
            self.is_dynamic_decoder = False

        elif gate_mode in ["image", "pixel"]:
            self.decoder = DynamicChangeDecoder(
                encoder_dims=self.encoder.dims,
                channel_first=self.encoder.channel_first,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                gate_mode=gate_mode,
                hidden_dim=128,
                **clean_kwargs,
            )
            self.is_dynamic_decoder = True

        else:
            raise ValueError(
                f"Unsupported gate_mode: {gate_mode}. "
                f"Expected one of ['none', 'image', 'pixel']."
            )

        # -------------------------
        # Main classification head
        # -------------------------
        self.main_clf = build_head(
            out_channels=2,
            in_channels=128,
            dropout_rate=dropout_rate,
        )

        # -------------------------
        # Evidential uncertainty head
        # -------------------------
        if self.use_uncertainty:
            self.evidence_head = nn.Conv2d(
                in_channels=128,
                out_channels=2,
                kernel_size=1,
            )
        else:
            self.evidence_head = None

        self.last_gate_weights = None
        self.last_uncertainty = None

    def forward(
        self,
        pre_data,
        post_data,
        return_aux=False,
        return_uncertainty=False,
    ):
        """
        return_aux=False:
            return logits: [B, 2, H, W]

        return_aux=True:
            return {
                "logits": logits,
                "gate_weights": gate_weights,
                optional "evidence", "alpha", "uncertainty"
            }

        return_uncertainty is kept only for compatibility with older mc_dropout.py.
        """
        if return_uncertainty:
            return_aux = True

        # -------------------------
        # Encoder
        # -------------------------
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # -------------------------
        # Multi-scale change-aware feature interaction
        # -------------------------
        if self.feature_interaction is not None:
            pre_features, post_features = self.feature_interaction(
                pre_features,
                post_features,
            )

        # -------------------------
        # Decoder
        # -------------------------
        gate_weights = None

        if self.is_dynamic_decoder:
            decoder_out = self.decoder(pre_features, post_features)

            if isinstance(decoder_out, tuple):
                change_feat = decoder_out[0]
                gate_weights = decoder_out[1] if len(decoder_out) > 1 else None
            else:
                change_feat = decoder_out
        else:
            change_feat = self.decoder(pre_features, post_features)

        # -------------------------
        # Main classifier
        # -------------------------
        logits = self.main_clf(change_feat)
        logits = resize_to_input(logits, pre_data)

        if gate_weights is not None:
            self.last_gate_weights = gate_weights.detach()
        else:
            self.last_gate_weights = None

        if not return_aux:
            return logits

        # -------------------------
        # Auxiliary outputs
        # -------------------------
        aux = {
            "logits": logits,
            "gate_weights": gate_weights,
        }

        if self.use_uncertainty:
            evidence = F.softplus(self.evidence_head(change_feat))
            evidence = resize_to_input(evidence, pre_data)

            alpha = evidence + 1.0
            uncertainty = 2.0 / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

            aux.update(
                {
                    "evidence": evidence,
                    "alpha": alpha,
                    "uncertainty": uncertainty,
                }
            )
            self.last_uncertainty = uncertainty.detach()
        else:
            self.last_uncertainty = None

        return aux
