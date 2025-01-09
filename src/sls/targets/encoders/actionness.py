import numpy as np

from sign_language_tools.annotations.transforms import SegmentsToSegmentationVector
from .base import TargetEncoder


class ActionnessEncoder(TargetEncoder):
    def __init__(self, length: int):
        super().__init__(length)
        self.transform = SegmentsToSegmentationVector(
            vector_size=length,
            background_label=0,
            use_annotation_labels=False,
        )

    def encode(self, segments: np.ndarray) -> np.ndarray:
        vector = self.transform(segments)
        return vector
