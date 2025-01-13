import numpy as np


def compute_iou_matrix(segments_1: np.ndarray, segments_2: np.ndarray) -> np.ndarray:
    starts_1 = segments_1[:, 0, None]   # Shape: (M, 1)
    ends_1 = segments_1[:, 1, None]     # Shape: (M, 1)
    starts_2 = segments_2[None, :, 0]   # Shape: (1, N)
    ends_2 = segments_2[None, :, 1]     # Shape: (1, N)

    # Compute the intersection
    inter_start = np.maximum(starts_1, starts_2)  # Shape: (M, N)
    inter_end = np.minimum(ends_1, ends_2)        # Shape: (M, N)
    intersection = np.maximum(inter_end - inter_start + 1, 0)  # Shape: (M, N)

    # Compute the union
    lengths_1 = ends_1 - starts_1 + 1  # Shape: (M, 1)
    lengths_2 = ends_2 - starts_2 + 1  # Shape: (1, N)
    union = lengths_1 + lengths_2 - intersection    # Shape: (M, N)

    # Avoid division by zero
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
