import torch
import torch.nn as nn
import torch.nn.functional as F

from .builders import build_backbone, resolve_decoder_components, build_head, resize_to_input
from .ChangeDecoder import ChangeDecoder
from .decoder_factory import DynamicChangeDecoder


class ChangeMambaBCD(nn.Module):
    def __init__(
        self,
        pretrained,
        gate_mode='pixel',
        use_uncertainty=False,
        dropout_rate=0.0,
        **kwargs
    ):
        super().__init__()

        self.encoder = build_backbone(pretrained=pretrained, **kwargs)

        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)

        self.gate_mode = gate_mode
        self.use_uncertainty = use_uncertainty

        # gate_mode='none' 时，使用原版 ChangeMamba decoder
        if gate_mode == 'none':
            self.decoder = ChangeDecoder(
                encoder_dims=self.encoder.dims,
                channel_first=self.encoder.channel_first,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                **clean_kwargs,
            )
            self.is_dynamic_decoder = False
        else:
            self.decoder = DynamicChangeDecoder(
                encoder_dims=self.encoder.dims,
                channel_first=self.encoder.channel_first,
                norm_layer=norm_layer,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,
                gate_mode=gate_mode,
                use_uncertainty=False,   # 不建议放 decoder 里
                **clean_kwargs,
            )
            self.is_dynamic_decoder = True

        self.main_clf = build_head(
            out_channels=2,
            dropout_rate=dropout_rate if dropout_rate > 0 else 0.0,
        )

        if self.use_uncertainty:
            self.evidence_head = nn.Conv2d(128, 2, kernel_size=1)
        else:
            self.evidence_head = None

        self.last_gate_weights = None
        self.last_uncertainty = None

    def forward(self, pre_data, post_data, return_aux=False):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        if self.is_dynamic_decoder:
            decoder_out = self.decoder(pre_features, post_features)

            # 兼容你现在 DynamicChangeDecoder 返回 3 个值的写法
            if isinstance(decoder_out, tuple):
                change_feat = decoder_out[0]
                gate_weights = decoder_out[-1]
            else:
                change_feat = decoder_out
                gate_weights = None
        else:
            change_feat = self.decoder(pre_features, post_features)
            gate_weights = None

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
            uncertainty = 2.0 / torch.sum(alpha, dim=1, keepdim=True)

            aux.update({
                "evidence": evidence,
                "alpha": alpha,
                "uncertainty": uncertainty,
            })

            self.last_uncertainty = uncertainty.detach()

        self.last_gate_weights = gate_weights.detach() if gate_weights is not None else None

        return aux