import torch
from torchmetrics import MetricCollection

from .segments import (
    MeanIoUF1Score,
    MeanMaxIoUOverSegments,
    SegmentMatching,
    SegmentProportion,
    MeanCenterDistF1Score,
)


class PerSegmentMetrics(MetricCollection):
    def __init__(self, **kwargs):
        metrics = {
            "mIoU": MeanMaxIoUOverSegments(),
            "mF1s@20": MeanIoUF1Score(thresholds=torch.tensor([0.2])),
            "mF1s@50": MeanIoUF1Score(thresholds=torch.tensor([0.5])),
            "mF1s@80": MeanIoUF1Score(thresholds=torch.tensor([0.8])),
            "mF1s@40-75-05": MeanIoUF1Score(thresholds=torch.arange(0.4, 0.76, 0.05)),
            "mF1b@5": MeanCenterDistF1Score(dist_thresholds=torch.tensor([5.0])),
            "mF1b@10": MeanCenterDistF1Score(dist_thresholds=torch.tensor([10.0])),
            "mF1b@20": MeanCenterDistF1Score(dist_thresholds=torch.tensor([20.0])),
            "mF1b@1-20": MeanCenterDistF1Score(dist_thresholds=torch.arange(1, 21)),
            "matching@50": SegmentMatching(threshold=0.5, relative=True),
            "proportion@50": SegmentProportion(),
        }
        super().__init__(metrics, **kwargs)
