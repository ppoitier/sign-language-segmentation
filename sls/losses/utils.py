from torch import Tensor

from .focal_loss import FocalLoss
from .mstcn import MSTCNLoss
from .ce import CrossEntropyLoss
from .cls_with_offsets import ClassificationWithOffsetsLoss


def get_loss_function(criterion: str, criterion_weights: Tensor | None = None):
    if criterion == 'mstcn':
        return MSTCNLoss()
    if criterion == 'ce':
        return CrossEntropyLoss(weights=criterion_weights)
    if criterion == 'focal_loss':
        return FocalLoss(weights=criterion_weights)
    if criterion == 'offsets+actionness':
        return ClassificationWithOffsetsLoss(n_classes=2)
    raise ValueError(f'Unknown criterion: {criterion}.')
