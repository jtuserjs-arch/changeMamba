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


class ChangeMambaBCD(nn.Module):
    """
    BCD model for ChangeMamba.

    gate_mode:
        - "none": use original ChangeDecoder, no dynamic gate
        - "image": image-level dynamic gate
        - "pixel": pixel-level dynamic gate

    use_uncertainty:
        - False: normal binary change detection
        - True: return evidence / alpha / uncertainty when return_aux=True
    """

    def __init__(
        self,
        pretrained,
        gate_mode="pixel",
        use_uncertainty=False,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__()

        self.encoder = build_backbone(pretrained=pretrained, **kwargs)

        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(
            kwargs
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

        self.main_clf = build_head(
            out_channels=2,
            in_channels=128,
            dropout_rate=dropout_rate,
        )

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

    def forward(self, pre_data, post_data, return_aux=False):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

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

        logits = self.main_clf(change_feat)
        logits = resize_to_input(logits, pre_data)

        if not return_aux:
            return logits

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

        if gate_weights is not None:
            self.last_gate_weights = gate_weights.detach()
        else:
            self.last_gate_weights = None

        return aux