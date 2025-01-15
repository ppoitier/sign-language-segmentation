from torch import nn, Tensor
import torch.nn.functional as F


class SmoothingLoss(nn.Module):
    def __init__(self, theta: float = 16):
        super().__init__()
        self.theta = theta

    def forward(self, logits: Tensor) -> Tensor:
        """
        Args:
            logits: tensor of shape (N, T, C_out)

        Returns:
            loss
        """
        return (
            F.mse_loss(
                logits[:, 1:].log_softmax(dim=-1),
                logits.detach()[:, :-1].log_softmax(dim=-1),
                reduction='none',
            )
            .clamp(min=0, max=self.theta)
            .mean()
        )
