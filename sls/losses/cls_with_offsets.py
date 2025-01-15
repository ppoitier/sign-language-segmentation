from torch import Tensor, nn

from sls.losses.ce import CrossEntropyLoss
from sls.losses.generalized_iou import GeneralizedIoU


class ClassificationWithOffsetsLoss(nn.Module):
    def __init__(self, n_classes: int, reg_loss_coef: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.reg_loss_coef = reg_loss_coef

        self.cls_loss = CrossEntropyLoss()
        self.reg_loss = GeneralizedIoU()

    def forward(self, multilayer_logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
            Args:
                multilayer_logits: tensor of shape (L_layers, N, T, C_cls + C_reg)
                targets: tensor of shape (N, T, 1 + C_reg)

            Returns:
                loss
            """
        cls_loss = 0
        reg_loss = 0

        cls_targets = targets[:, :, 0].long()
        reg_targets = targets[:, :, 1:]

        for logits in multilayer_logits.unbind():
            cls_logits = logits[:, :, :self.n_classes]
            cls_loss += self.cls_loss(cls_logits, cls_targets)

            reg_logits = logits[:, :, self.n_classes:]
            reg_loss += self.reg_loss(reg_logits, reg_targets)

        return cls_loss + self.reg_loss_coef * reg_loss



