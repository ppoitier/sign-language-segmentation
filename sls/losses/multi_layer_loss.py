from torch import nn, Tensor

from sls.losses.cls_with_smoothing import ClassificationLossWithSmoothing


# class MultiLayerLoss(nn.Module):
#     def __init__(self, smoothing_coef: float = 0.15):
#         super().__init__()
#         self.smoothing_coef = smoothing_coef
#
#         self.cls_loss = CrossEntropyLoss()
#         self.smoothing_loss = SmoothingLoss()
#
#     def forward(self, multilayer_logits: Tensor, targets: Tensor):
#         """
#         Args:
#             multilayer_logits: tensor of shape (N_layers, N, T, C_out)
#             targets: tensor of shape (N, T)
#
#         Returns:
#             loss
#         """
#         cls_loss = 0
#         smoothing_loss = 0
#         for logits in multilayer_logits:
#             cls_loss += self.cls_loss(logits, targets)
#             smoothing_loss += self.smoothing_loss(logits)
#         return cls_loss + self.smoothing_coef * smoothing_loss


class MultiLayerLoss(nn.Module):
    def __init__(self, single_layer_loss: nn.Module):
        super().__init__()
        self.loss = single_layer_loss

    def forward(self, multilayer_logits: Tensor, targets: Tensor):
        loss = 0
        for logits in multilayer_logits.unbind(0):
            loss += self.loss(logits, targets)
        return loss
