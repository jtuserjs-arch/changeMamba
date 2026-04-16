
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