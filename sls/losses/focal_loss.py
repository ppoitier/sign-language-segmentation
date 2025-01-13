from torch import nn, exp, Tensor
from torch.nn.functional import cross_entropy


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weights: Tensor | None = None,
        reduction: str = "mean",
        **kwargs,
    ):
        """Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf"""
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weights", weights)
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            targets,
            weight=self.weights,
            reduction="none",
            ignore_index=-1,
            **self.kwargs
        )
        p_t = exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
