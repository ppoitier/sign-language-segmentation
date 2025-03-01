import torch
from torch import Tensor
from torchmetrics import Metric

from .functional import tp_fp_fn


class SegmentMatching(Metric):
    def __init__(self, threshold: float, relative: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.relative = relative
        self.add_state("thresholds", default=torch.tensor([threshold], dtype=torch.float), persistent=True)
        self.add_state("n_preds", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred_segments: list[Tensor], gt_segments: list[Tensor]) -> None:
        """
        Args:
            pred_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
            gt_segments: batch of tensors of shape (N, 2) for the start and the end of N predicted segments.
        """
        for pred_segments_b, gt_segments_b in zip(pred_segments, gt_segments):
            tp, fp, fn = tp_fp_fn(pred_segments_b, gt_segments_b, self.thresholds)
            self.n_preds += pred_segments_b.shape[0]
            self.tp += tp[0]
            self.fp += fp[0]
            self.fn += fn[0]

    def compute(self) -> Tensor:
        tp, fp, fn = self.tp, self.fp, self.fn
        if self.relative:
            tp = tp / self.n_preds if self.n_preds > 0 else 0
            fp = fp / self.n_preds if self.n_preds > 0 else 0
            fn = fn / self.n_preds if self.n_preds > 0 else 0
        return torch.tensor([tp, fp, fn], device=self.device)

