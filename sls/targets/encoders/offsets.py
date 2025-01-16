import numpy as np

from sign_language_tools.annotations.transforms import (
    LinearBoundaryOffset,
)
from .base import TargetEncoder


class OffsetsEncoder(TargetEncoder):
    def __init__(self, length: int):
        super().__init__(length)
        self.to_start_offset = LinearBoundaryOffset(length, ref_location='start', background_class=-1)
        self.to_end_offset = LinearBoundaryOffset(length, ref_location='end', background_class=-1)

    def encode(self, segments: np.ndarray) -> np.ndarray:
        start_offsets = self.to_start_offset.transform(segments)
        end_offsets = self.to_end_offset.transform(segments)
        return np.stack([start_offsets, end_offsets], axis=-1)


class OffsetsWithSegmentationEncoder(TargetEncoder):
    def __init__(self, length: int, segmentation_encoder: TargetEncoder):
        super().__init__(length)
        self.offset_encoder = OffsetsEncoder(length)
        self.segments_encoder = segmentation_encoder

    def encode(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: array of shape (M, 2) for M segments with a start and a end

        Returns:
            encoded_target: array of shape (T, C_cls + C_reg)
        """
        segmentation = self.segments_encoder(segments)
        offsets = self.offset_encoder(segments)
        return np.concatenate([segmentation[:, None], offsets], axis=-1)
