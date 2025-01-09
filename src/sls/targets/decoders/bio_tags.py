import numpy as np

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments
from .base import TargetDecoder


class BIOTagDecoder(TargetDecoder):
    def __init__(self):
        super().__init__()
        self.transform = SegmentationVectorToSegments(
            background_classes=(-1, 0),
            use_annotation_labels=False,
        )

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        # noinspection PyTypeChecker
        binary_segmentation: np.ndarray = (encoded == 2) | (encoded == 1)
        return self.transform(binary_segmentation)
