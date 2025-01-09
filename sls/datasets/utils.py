import numpy as np


def compute_window_indices(sequence_length: int, window_size: int, stride: int) -> np.ndarray:
    """
    Compute the start and end indices for each window, including a possibly shorter last window.

    Args:
        sequence_length: Total length of the sequence
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        np.ndarray: Array of shape (n_windows, 2) containing start and end indices for each window
    """
    start_indices = np.arange(0, sequence_length, stride)
    end_indices = np.minimum(start_indices + window_size, sequence_length)
    valid_windows = start_indices != end_indices
    return np.column_stack((start_indices[valid_windows], end_indices[valid_windows]))


def get_window_from_instance(instance: dict, start: int, end: int):
    if isinstance(instance, np.ndarray) or isinstance(instance, list):
        return instance[start:end]
    elif isinstance(instance, dict):
        return {
            k: get_window_from_instance(v, start, end)
            for k, v in instance.items()
        }
    else:
        return instance
