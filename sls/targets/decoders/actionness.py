import numpy as np

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments
from .base import TargetDecoder


class ActionnessDecoder(TargetDecoder):
    def __init__(self):
        super().__init__()
        self.transform = SegmentationVectorToSegments(
            background_classes=(-1, 0),
            use_annotation_labels=False,
        )

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        return self.transform(encoded)
