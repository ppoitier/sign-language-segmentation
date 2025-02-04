import torch
from torchmetrics import MetricCollection

from .segments import MeanF1ScoreOverSegments, MeanMaxIoUOverSegments, SegmentMatching, SegmentProportion


class PerSegmentMetrics(MetricCollection):
    def __init__(self, **kwargs):
        metrics = {
            "mIoU": MeanMaxIoUOverSegments(),
            "mF1@20": MeanF1ScoreOverSegments(thresholds=torch.tensor([0.2])),
            "mF1@50": MeanF1ScoreOverSegments(thresholds=torch.tensor([0.5])),
            "mF1@80": MeanF1ScoreOverSegments(thresholds=torch.tensor([0.8])),
            "mF1@40-75-05": MeanF1ScoreOverSegments(thresholds=torch.arange(0.4, 0.76, 0.05)),
            "matching@50": SegmentMatching(threshold=0.5),
            "proportion@50": SegmentProportion(),
        }
        super().__init__(metrics, **kwargs)
