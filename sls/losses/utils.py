from torch import Tensor

from .ce import CrossEntropyLoss
from .focal_loss import FocalLoss
from .multi_layer_loss import MultiLayerLoss
from .cls_with_smoothing import MultiLayerClassificationLossWithSmoothing


def get_loss_function(
        criterion: str,
        criterion_weights: Tensor | None = None,
):
    if criterion == 'multi-layer+ce+smoothing':
        return MultiLayerClassificationLossWithSmoothing(
            cls_loss_fn=CrossEntropyLoss(weights=criterion_weights),
            return_loss_components=False,
        )
    elif criterion == 'multi-layer+fl+smoothing':
        return MultiLayerClassificationLossWithSmoothing(
            cls_loss_fn=FocalLoss(weights=criterion_weights),
            return_loss_components=False,
        )
    elif criterion == 'multi-layer+ce':
        return MultiLayerLoss(CrossEntropyLoss(weights=criterion_weights))
    elif criterion == 'multi-layer+fl':
        return MultiLayerLoss(FocalLoss(weights=criterion_weights))
    raise ValueError(f'Unknown criterion: {criterion}.')
