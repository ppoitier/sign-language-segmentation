import numpy as np

from sign_language_tools.annotations.transforms import (
    SegmentsToBoundaries,
    FilterShortSilence,
    RemoveOverlapping,
    SegmentsToSegmentationVector,
)
from sign_language_tools.common.transforms import Compose
from .base import TargetEncoder


class BoundariesEncoder(TargetEncoder):
    def __init__(self, length: int, boundary_width: float | int = 2):
        super().__init__(length)
        self.boundary_width = boundary_width
        self.transform = Compose(
            [
                # FilterShortSilence(min_duration=8),
                SegmentsToBoundaries(
                    width=boundary_width if isinstance(boundary_width, int) else None,
                    relative_width=(
                        boundary_width if isinstance(boundary_width, float) else None
                    ),
                    min_width=2,
                    max_end=length,
                    exclude_edges=True,
                    discrete_gap=True,
                    rounded=True,
                ),
                # RemoveOverlapping(min_gap=1),
                SegmentsToSegmentationVector(
                    vector_size=length,
                    use_annotation_labels=False,
                    background_label=0,
                    fill_label=1,
                ),
            ]
        )

    def encode(self, segments: np.ndarray) -> np.ndarray:
        return self.transform(segments)
