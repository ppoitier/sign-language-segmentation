import numpy as np

from .base import TargetDecoder
from sls.utils.proposals import generate_proposals
from sls.utils.nms import soft_nms


class OffsetsDecoder(TargetDecoder):
    def __init__(
            self,
            soft_nms_sigma: float = 0.5,
            soft_nms_threshold: float = 0.2,
    ):
        super().__init__()
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Offset decoder not implemented yet.")

        per_frame_probs = encoded[0]
        start_offsets = encoded[1]
        end_offsets = encoded[2]

        scores = ...
        generated_proposals = generate_proposals(start_offsets, end_offsets)
        filtered_proposals, _ = soft_nms(generated_proposals, scores, sigma=self.soft_nms_sigma, threshold=self.soft_nms_threshold)

        return filtered_proposals
