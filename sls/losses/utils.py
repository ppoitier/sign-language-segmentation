from torch import Tensor

from .ce import CrossEntropyLoss
from .focal_loss import FocalLoss
from .multi_layer_loss import MultiLayerLoss
from .cls_with_smoothing import ClassificationLossWithSmoothing, MultiLayerClassificationLossWithSmoothing
from .cls_with_offsets import MultiLayerClassificationWithOffsetsLoss
from .generalized_iou import GeneralizedIoU


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
    elif criterion == 'multi-layer+ce+smoothing+offsets':
        return MultiLayerClassificationWithOffsetsLoss(
            cls_loss_fn=ClassificationLossWithSmoothing(CrossEntropyLoss(weights=criterion_weights)),
            reg_loss_fn=GeneralizedIoU(),
            n_classes=2,    # TODO
            return_loss_components=True,
        )
    elif criterion == 'multi-layer+fl+smoothing+offsets':
        return MultiLayerClassificationWithOffsetsLoss(
            cls_loss_fn=ClassificationLossWithSmoothing(FocalLoss(weights=criterion_weights)),
            reg_loss_fn=GeneralizedIoU(),
            n_classes=2,    # TODO
            return_loss_components=True,
        )
    elif criterion == 'multi-layer+ce':
        return MultiLayerLoss(CrossEntropyLoss(weights=criterion_weights))
    elif criterion == 'multi-layer+fl':
        return MultiLayerLoss(FocalLoss(weights=criterion_weights))
    raise ValueError(f'Unknown criterion: {criterion}.')
