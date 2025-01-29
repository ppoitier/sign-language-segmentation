from torch import nn, Tensor

from sls.losses.smoothing_loss import SmoothingLoss


class ClassificationLossWithSmoothing(nn.Module):
    def __init__(
        self,
        cls_loss_fn: nn.Module,
        return_loss_components: bool = False,
        smoothing_coef: float = 0.15,
        smoothing_theta: float = 16,
    ):
        super().__init__()
        self.return_loss_components = return_loss_components
        self.smoothing_coef = smoothing_coef

        self.cls_loss_fn = cls_loss_fn
        self.smoothing_loss_fn = SmoothingLoss(smoothing_theta)

    def forward(self, logits: Tensor, targets: Tensor):
        cls_loss = self.cls_loss_fn(logits, targets)
        smoothing_loss = self.smoothing_loss_fn(logits)
        loss = cls_loss + self.smoothing_coef * smoothing_loss
        if self.return_loss_components:
            return loss, cls_loss, smoothing_loss
        return loss


class MultiLayerClassificationLossWithSmoothing(nn.Module):
    def __init__(
            self,
            cls_loss_fn: nn.Module,
            return_loss_components: bool = False,
            smoothing_coef: float = 0.15,
            smoothing_theta: float = 16,
    ):
        super().__init__()
        self.single_layer_loss_fn = ClassificationLossWithSmoothing(
            cls_loss_fn=cls_loss_fn,
            return_loss_components=return_loss_components,
            smoothing_coef=smoothing_coef,
            smoothing_theta=smoothing_theta,
        )
        self.return_loss_components = return_loss_components

    def forward(self, multilayer_logits: Tensor, targets: Tensor):
        if self.return_loss_components:
            loss = (0, 0, 0)
        else:
            loss = 0
        for logits in multilayer_logits.unbind(0):
            loss += self.single_layer_loss_fn(logits, targets)
        return loss
