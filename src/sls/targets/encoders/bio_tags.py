import numpy as np

from sign_language_tools.annotations.transforms import ScaleSegments, SegmentsToSegmentationVector
from .base import TargetEncoder


class BIOTagEncoder(TargetEncoder):
    def __init__(self, length: int, begin_width: float | int = 0.3):
        super().__init__(length)
        self._to_begin = ScaleSegments(factor=begin_width, location="start")
        self._to_inside = ScaleSegments(factor=1.0 - begin_width, location="end")
        self._to_segmentation = SegmentsToSegmentationVector(
            vector_size=length,
            use_annotation_labels=False,
            background_label=0,
            fill_label=1,
        )

    def encode(self, segments: np.ndarray) -> np.ndarray:
        b_tags = self._to_segmentation(self._to_begin(segments))
        i_tags = self._to_segmentation(self._to_inside(segments))
        b_indices = b_tags > 0
        i_tags[b_indices] = 2
        return i_tags
