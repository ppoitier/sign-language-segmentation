import numpy as np

from sls.utils.nms import soft_nms
from sls.utils.proposals import generate_proposals
from .base import TargetDecoder


class OffsetsDecoder(TargetDecoder):
    def __init__(
        self,
        n_classes: int = 2,
        soft_nms_method: str = 'gaussian',
        soft_nms_sigma: float = 0.2,
        soft_nms_threshold: float = 0.2,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.soft_nms_method = soft_nms_method
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        probs = encoded[:, : self.n_classes]
        pred_offsets = encoded[:, self.n_classes :]
        proposals, proposal_indices = generate_proposals(
            start_offsets=pred_offsets[:, 0],
            end_offsets=pred_offsets[:, 1],
            return_indices=True,
        )
        proposal_scores = 1 - probs[proposal_indices, 0]
        proposals, _ = soft_nms(
            proposals,
            scores=proposal_scores,
            method=self.soft_nms_method,
            sigma=self.soft_nms_sigma,
            threshold=self.soft_nms_threshold,
        )
        return proposals.round().astype('int32')
