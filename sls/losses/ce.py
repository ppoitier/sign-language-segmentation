from torch import nn
from torch.nn.functional import cross_entropy


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        return cross_entropy(logits.permute(0, 2, 1).contiguous(), targets, ignore_index=-1)
