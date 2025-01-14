from torch import Tensor, nn

from sls.losses.generalized_iou import GeneralizedIoU


class ClassificationWithOffsetsLoss(nn.Module):
    def __init__(self, n_classes: int, reg_loss_coef: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.reg_loss = GeneralizedIoU()
        self.reg_loss_coef = reg_loss_coef

    def forward(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """

            Args:
                logits: tensor of shape (L, N, C_cls + C_reg, T)
                targets: tensor of shape (N, C_cls + C_reg, T)

            Returns:
                loss
            """
        cls_loss = 0
        reg_loss = 0

        cls_targets = targets[:, 0].long()
        reg_targets = targets[:, 1:]

        for layer in logits.unbind():
            cls_logits = layer[:, :self.n_classes]
            cls_loss += self.cls_loss(cls_logits, cls_targets)

            reg_logits = layer[:, self.n_classes:]
            reg_loss += self.reg_loss(
                reg_logits.transpose(-1, -2).contiguous(),
                reg_targets.transpose(-1, -2).contiguous(),
            )

        return cls_loss + self.reg_loss_coef * reg_loss, cls_loss, reg_loss



