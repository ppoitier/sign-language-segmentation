from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from sign_language_tools.annotations.transforms import (
    ScaleSegments,
    SegmentsToSegmentationVector,
    SegmentationVectorToSegments,
    SegmentsToBoundaries,
    RemoveOverlapping,
    LinearBoundaryOffset,
    MergeSegmentsOnTransition,
    FilterShortSilence,
    FillBetween,
)
from sign_language_tools.common.transforms import Compose


class SegmentEncoder(ABC):
    def __init__(self, segmentation_size: int):
        self.segmentation_size = segmentation_size

    def __call__(self, x: Tensor) -> Tensor:
        x_arr = x.detach().cpu().numpy()
        return torch.from_numpy(self.encode_to_segmentation(x_arr)).to(x.device)

    def encode_to_segments(self, segments: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Invalid encoding operation.")

    @abstractmethod
    def encode_to_segmentation(
        self, segments: np.ndarray, length: int | None = None
    ) -> np.ndarray:
        raise NotImplementedError("Invalid encoding operation.")

    def decode(self, x: Tensor) -> list[Tensor]:
        return [
            torch.from_numpy(self.decode_numpy(sub_x.detach().cpu().numpy())).to(
                x.device
            )
            for sub_x in x.unbind(0)
        ]

    @abstractmethod
    def decode_numpy(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Invalid decoding operation.")


class ActionnessEncoder(SegmentEncoder):
    def __init__(self, segmentation_size: int):
        super().__init__(segmentation_size)
        self._to_segmentation = SegmentsToSegmentationVector(
            vector_size=segmentation_size,
            use_annotation_labels=False,
            background_label=0,
            fill_label=1,
        )
        self._to_segments = SegmentationVectorToSegments(
            background_classes=(0, -1),
            use_annotation_labels=False,
        )

    def encode_to_segmentation(
        self, segments: np.ndarray, length: int | None = None
    ) -> np.ndarray:
        segmentation = self._to_segmentation(segments)
        if length is not None:
            segmentation[length:] = -1
        return segmentation

    def decode_numpy(self, segmentation: np.ndarray) -> np.ndarray:
        return self._to_segments(segmentation)


class BIOTagEncoder(SegmentEncoder):
    def __init__(self, segmentation_size: int, begin_width: float = 0.3):
        super().__init__(segmentation_size)
        self._to_begin = ScaleSegments(factor=begin_width, location="start")
        self._to_inside = ScaleSegments(factor=1.0 - begin_width, location="end")
        self._to_segmentation = SegmentsToSegmentationVector(
            vector_size=segmentation_size,
            use_annotation_labels=False,
            background_label=0,
            fill_label=1,
        )
        self._to_segments = SegmentationVectorToSegments(
            background_classes=(0, -1),
            use_annotation_labels=True,
        )
        self._bio_tags_to_segments = Compose(
            [
                SegmentationVectorToSegments(
                    background_classes=(0, -1), use_annotation_labels=True
                ),
                FilterShortSilence(4),
                MergeSegmentsOnTransition([(2, 1)], new_value=1),
            ]
        )

    def encode_to_segmentation(
        self, segments: np.ndarray, length: int | None = None
    ) -> np.ndarray:
        b_tags = self._to_segmentation(self._to_begin(segments))
        i_tags = self._to_segmentation(self._to_inside(segments))
        b_indices = b_tags > 0
        i_tags[b_indices] = 2
        if length is not None:
            i_tags[length:] = -1
        return i_tags

    def encode_to_segments(self, segments: np.ndarray) -> np.ndarray:
        segmentation = self.encode_to_segmentation(segments)
        return self._to_segments(segmentation)

    def decode_numpy(self, bio_tags: np.ndarray) -> np.ndarray:
        return self._bio_tags_to_segments(bio_tags)


class BoundariesEncoder(SegmentEncoder):
    def __init__(
        self, segmentation_size: int, relative_width: float, min_width: int = 2
    ):
        super().__init__(segmentation_size)
        self.to_boundaries = Compose(
            [
                SegmentsToBoundaries(
                    relative_width=relative_width,
                    min_width=min_width,
                    max_end=segmentation_size,
                    exclude_edges=False,
                ),
                RemoveOverlapping(min_gap=2),
            ]
        )
        self._to_segmentation = SegmentsToSegmentationVector(
            vector_size=segmentation_size, background_label=0, fill_label=1, use_annotation_labels=False,
        )
        self._to_segments = SegmentationVectorToSegments(
            background_classes=(0, -1), use_annotation_labels=True
        )
        self._boundaries_to_signs = Compose(
            [
                FillBetween(start_value=1, end_value=1, fill_value=2, alternate=True),
                MergeSegmentsOnTransition(transitions=[(1, 2)], new_value=2),
                MergeSegmentsOnTransition(transitions=[(2, 1)], new_value=1),
            ]
        )

    def encode_to_segments(self, segments: np.ndarray) -> np.ndarray:
        return self.to_boundaries(segments)

    def encode_to_segmentation(
        self, segments: np.ndarray, length: int | None = None
    ) -> np.ndarray:
        segmentation = self._to_segmentation(self.encode_to_segments(segments))
        if length is not None:
            segmentation[length:] = -1
        return segmentation

    def decode_numpy(self, segmentation: np.ndarray) -> np.ndarray:
        segments = self._to_segments(segmentation)
        return self._boundaries_to_signs(segments)


class OffsetsEncoder(SegmentEncoder):
    def __init__(self, segmentation_size: int):
        super().__init__(segmentation_size)
        self._start_offset = LinearBoundaryOffset(
            sequence_length=segmentation_size, background_class=-1, ref_location="start"
        )
        self._end_offset = LinearBoundaryOffset(
            sequence_length=segmentation_size, background_class=-1, ref_location="end"
        )

    def encode_to_segmentation(
        self, segments: np.ndarray, length: int | None = None
    ) -> np.ndarray:
        start_offsets = self._start_offset(segments)
        end_offsets = self._end_offset(segments)
        return np.stack((start_offsets, end_offsets), axis=-1)

    def decode_numpy(self, offsets: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO.")
