import torch
import torch.nn as nn
import torch.nn.functional as F
from .builders import build_backbone, resolve_decoder_components, build_head
from .decoder_factory import DynamicChangeDecoder   # 新增导入

def resize_to_input(output, input):
    return F.interpolate(output, size=input.shape[-2:], mode='bilinear')

class ChangeMambaBCD(nn.Module):
    def __init__(self, pretrained, gate_mode='pixel', use_uncertainty=False, **kwargs):
        super().__init__()
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)
        
        # 使用动态解码器
        self.decoder = DynamicChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            gate_mode=gate_mode,
            use_uncertainty=use_uncertainty,
            **clean_kwargs,
        )
        self.main_clf = build_head(out_channels=2)
        self.use_uncertainty = use_uncertainty
         # 如果启用不确定性，则分类头加入 Dropout
        if use_uncertainty:
            self.main_clf = build_head(out_channels=2, dropout_rate=dropout_rate)
        else:
            self.main_clf = build_head(out_channels=2)

    def forward(self, pre_data, post_data, return_uncertainty=False):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        out = self.decoder(pre_features, post_features)

        if self.use_uncertainty:
            change_feat, uncertainty, gate_weights = out
            # 存储到属性，方便外部获取（用于可视化）
            self.last_uncertainty = uncertainty
            self.last_gate_weights = gate_weights
        else:
            change_feat, _, _ = out   # gate_weights 为 None

        logits = self.main_clf(change_feat)

        # 添加这一行：上采样到输入图像的尺寸（例如 256×256）
        logits = F.interpolate(logits, size=pre_data.shape[-2:], mode='bilinear')

        if return_uncertainty and self.use_uncertainty:
            return logits, uncertainty, gate_weights
        else:
            return logits