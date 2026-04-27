import torch
import torch.nn.functional as F

@torch.no_grad()
def mc_dropout_inference(model, pre_img, post_img, num_samples=30, return_uncertainty=True):
    """
    Monte Carlo Dropout 推理，用于不确定性估计。
    Args:
        model: ChangeMambaBCD 模型（已设置 use_uncertainty=True）
        pre_img, post_img: 输入图像 [B, C, H, W]
        num_samples: 采样次数
        return_uncertainty: 是否返回不确定性图
    Returns:
        pred: 最终预测 [B, H, W] (argmax)
        uncertainty: 不确定性图 [B, H, W] (预测熵) 或 None
        gate_weights: 最后一次前向的门控权重（可选）
    """
    model.train()  # 启用 Dropout
    logits_list = []
    gate_weights_list = []
    
    for _ in range(num_samples):
        # 如果模型支持返回 uncertainty 和 gate_weights，则获取
        out = model(pre_img, post_img, return_uncertainty=True)
        if len(out) == 3:
            logits, unc, gate = out
            gate_weights_list.append(gate)
        else:
            logits = out
        logits_list.append(logits)
    
    # 堆叠所有采样结果 [num_samples, B, 2, H, W]
    logits_stack = torch.stack(logits_list, dim=0)
    probs = F.softmax(logits_stack, dim=2)  # [num_samples, B, 2, H, W]
    mean_probs = probs.mean(dim=0)          # [B, 2, H, W]
    
    # 预测类别
    pred = mean_probs.argmax(dim=1)         # [B, H, W]
    
    if return_uncertainty:
        # 计算预测熵
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)  # [B, H, W]
        # 可选：返回最后一次的门控权重（或平均门控）
        gate_weights = gate_weights_list[-1] if gate_weights_list else None
        return pred, entropy, gate_weights
    else:
        return pred