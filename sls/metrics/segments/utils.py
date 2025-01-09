import torch
from torch import Tensor


def compute_iou_matrix(segments_1: Tensor, segments_2: Tensor) -> Tensor:
    """
    Computes the Intersection over Union (IoU) matrix between two arrays of segments.

    Args:
    - segments_1: Tensor of shape (M, 2), where each row is [start, end]
    - segments_2: Tensor of shape (N, 2), where each row is [start, end]

    Returns:
    - iou_matrix: Tensor of shape (M, N), where each element is the IoU between segments_1[i] and segments_2[j]
    """
    starts_1 = segments_1[:, 0].unsqueeze(1)  # Shape: (M, 1)
    ends_1 = segments_1[:, 1].unsqueeze(1)    # Shape: (M, 1)
    starts_2 = segments_2[:, 0].unsqueeze(0)  # Shape: (1, N)
    ends_2 = segments_2[:, 1].unsqueeze(0)    # Shape: (1, N)

    # Compute the intersection
    inter_start = torch.max(starts_1, starts_2)  # Shape: (M, N)
    inter_end = torch.min(ends_1, ends_2)        # Shape: (M, N)
    intersection = torch.clamp(inter_end - inter_start, min=0)  # Shape: (M, N)

    # Compute the union
    lengths_1 = ends_1 - starts_1 + 1  # Shape: (M, 1)
    lengths_2 = ends_2 - starts_2 + 1  # Shape: (1, N)
    union = lengths_1 + lengths_2 - intersection    # Shape: (M, N)

    # Avoid division by zero
    iou_matrix = torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(intersection, dtype=torch.float32)
    )

    return iou_matrix
