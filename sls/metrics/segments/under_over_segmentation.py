import torch
from torch import Tensor
from torchmetrics import Metric

from sls.metrics.segments.functional import compute_iou_matrix


class UnderOverSegmentation(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("n_overlaps", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_gts", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_preds", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_segments: list[Tensor], gt_segments: list[Tensor]) -> None:
        """
        Args:
            pred_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
            gt_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
        """
        for pred_segments_b, gt_segments_b in zip(pred_segments, gt_segments):
            iou_matrix = compute_iou_matrix(pred_segments_b, gt_segments_b)
            overlaps = iou_matrix > 0.0
            n_overlaps = overlaps.sum()
            self.n_overlaps += n_overlaps
            self.n_gts += gt_segments_b.shape[1]
            self.n_preds += pred_segments_b.shape[0]

    def compute(self) -> Tensor:
        under_segmentation_ratio = self.n_overlaps / self.n_preds if self.n_preds > 0 else 0
        over_segmentation_ratio = self.n_overlaps / self.n_gts if self.n_gts > 0 else 0
        return torch.stack([under_segmentation_ratio, over_segmentation_ratio])


if __name__ == '__main__':
    gt = torch.tensor([
        [0, 10],
        [13, 16],
    ])
    pred = torch.tensor([
        [1, 3],
        [13, 16],
        [6, 10],
        [13, 15],
        [3, 6],
        [0, 16],
    ])
    metric = UnderOverSegmentation()
    print(metric([pred], [gt]))
    metric.reset()
    print(metric([pred], [gt]))
