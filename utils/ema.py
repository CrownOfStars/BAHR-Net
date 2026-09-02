"""
Exponential Moving Average (EMA) of model parameters.
验证和保存时使用 EMA 权重通常能带来 0.3-0.8% 的稳定提升。
"""
import copy
import torch
import torch.nn as nn


class ModelEMA(nn.Module):
    """
    模型参数的指数移动平均。
    shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self._update_steps = 0
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._update_steps += 1
        # 早期使用较小的有效衰减，避免 EMA 在训练初期严重滞后
        d = min(self.decay, (1.0 + self._update_steps) / (10.0 + self._update_steps))
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(model_p, alpha=1 - d)
        # 同步 buffers（BN 的 running_mean/var 等），否则推理时 BN 使用过时统计量导致异常
        for ema_buf, model_buf in zip(self.ema.buffers(), model.buffers()):
            ema_buf.copy_(model_buf)

    def forward(self, *args, **kwargs):
        return self.ema(*args, **kwargs)
