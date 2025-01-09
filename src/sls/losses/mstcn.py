from torch import nn
import torch.nn.functional as F


def compute_smoothing_loss(logits, theta=16):
    # p: (N, C, L)
    return F.mse_loss(
        logits[:, :, 1:].log_softmax(dim=1),
        logits.detach()[:, :, :-1].log_softmax(dim=1)
    ).clamp(0, theta).mean()


def compute_loss(logits_per_layer, targets, smoothing_coef=0.15):
    # logits: (num_layers, batch_size, c, length)
    ce_loss = 0
    smoothing_loss = 0
    for logits in logits_per_layer:
        ce_loss += F.cross_entropy(logits, targets, ignore_index=-1)
        smoothing_loss += compute_smoothing_loss(logits)
    return ce_loss + smoothing_coef * smoothing_loss, ce_loss, smoothing_loss


class MSTCNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        loss, _, _ = compute_loss(logits, targets)
        return loss
