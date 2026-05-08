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

# =========================
# Cross-Attention 模块
# =========================
class CrossAttention(nn.Module):
    """
    Cross-attention for two feature maps f1 <-> f2
    f1, f2: [B, C, H, W]
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5

        self.qkv1 = nn.Conv2d(dim, dim*3, 1)
        self.qkv2 = nn.Conv2d(dim, dim*3, 1)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape

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


# =========================
# ChangeMambaBCD
# =========================
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

        # -------------------------
        # encoder
        # -------------------------
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        self.encoder_dims = self.encoder.dims

        # cross-attention 对应 encoder 中间层
        # 假设选择 encoder 第二层 dim=encoder_dims[1]
        self.cross_attn = CrossAttention(dim=self.encoder_dims[1], num_heads=4)

        # -------------------------
        # decoder + gate
        # -------------------------
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

        # -------------------------
        # main classification head
        # -------------------------
        self.main_clf = build_head(
            out_channels=2,
            in_channels=128,
            dropout_rate=dropout_rate,
        )

        # -------------------------
        # uncertainty head
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

    # =========================
    # forward
    # =========================
    def forward(self, pre_data, post_data, return_aux=False):
        # -------------------------
        # encoder
        # -------------------------
        pre_features = self.encoder(pre_data)  # list of features
        post_features = self.encoder(post_data)

        # -------------------------
        # cross-attention for middle layer (index=1)
        # -------------------------
        pre_features[1], post_features[1] = self.cross_attn(pre_features[1], post_features[1])

        gate_weights = None

        # -------------------------
        # decoder
        # -------------------------
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
        # main classifier
        # -------------------------
        logits = self.main_clf(change_feat)
        logits = resize_to_input(logits, pre_data)

        if not return_aux:
            return logits

        # -------------------------
        # auxiliary outputs
        # -------------------------
        aux = {"logits": logits, "gate_weights": gate_weights}

        if self.use_uncertainty:
            evidence = F.softplus(self.evidence_head(change_feat))
            evidence = resize_to_input(evidence, pre_data)

            alpha = evidence + 1.0
            uncertainty = 2.0 / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

            aux.update({"evidence": evidence, "alpha": alpha, "uncertainty": uncertainty})

            self.last_uncertainty = uncertainty.detach()
        else:
            self.last_uncertainty = None

        if gate_weights is not None:
            self.last_gate_weights = gate_weights.detach()
        else:
            self.last_gate_weights = None

        return aux