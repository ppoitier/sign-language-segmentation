from torch import nn, Tensor

from sls.losses.ce import CrossEntropyLoss
from sls.losses.smoothing_loss import SmoothingLoss


class MultiLayerLoss(nn.Module):
    def __init__(self, smoothing_coef: float = 0.15):
        super().__init__()
        self.smoothing_coef = smoothing_coef

        self.cls_loss = CrossEntropyLoss()
        self.smoothing_loss = SmoothingLoss()

    def forward(self, multilayer_logits: Tensor, targets: Tensor):
        """
        Args:
            multilayer_logits: tensor of shape (N_layers, N, T, C_out)
            targets: tensor of shape (N, T)

        Returns:
            loss
        """
        cls_loss = 0
        smoothing_loss = 0
        for logits in multilayer_logits:
            cls_loss += self.cls_loss(logits, targets)
            smoothing_loss += self.smoothing_loss(logits)
        return cls_loss + self.smoothing_coef * smoothing_loss
