import torch
from torch import Tensor
from torchmetrics import Metric


class SegmentProportion(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("n_preds", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_gts", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_segments: list[Tensor], gt_segments: list[Tensor]) -> None:
        """
        Args:
            pred_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
            gt_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
        """
        for pred_segments_b, gt_segments_b in zip(pred_segments, gt_segments):
            self.n_preds += pred_segments_b.shape[0]
            self.n_gts += gt_segments_b.shape[0]

    def compute(self) -> Tensor:
        if self.n_gts == 0:
            return torch.tensor(0, dtype=torch.float32, device=self.device)
        return self.n_preds / self.n_gts
