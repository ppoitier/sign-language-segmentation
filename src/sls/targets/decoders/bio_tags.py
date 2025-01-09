import numpy as np

from sign_language_tools.annotations.transforms import SegmentationVectorToSegments
from .base import TargetDecoder


class BIOTagEncoder(TargetDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        binary_segmentation = (encoded == 2) | (encoded == 1)
