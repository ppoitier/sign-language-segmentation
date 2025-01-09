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


def get_segments_in_range(segments, start, end):
    return segments[(segments[:, 0] < end) & (segments[:, 1] >= start)]


def get_window_from_instance(instance: dict, start: int, end: int):
    if isinstance(instance, np.ndarray) or isinstance(instance, list):
        return instance[start:end]
    elif isinstance(instance, dict):
        new_instance = dict()
        for k, v in instance.items():
            if k == 'segments':
                new_instance[k] = get_segments_in_range(v, start, end)
            elif k != 'segment_classes':
                new_instance[k] = get_window_from_instance(v, start, end)
        return new_instance
    else:
        return instance


def subdivide_instance_into_windows(instance: dict, window_indices: np.ndarray):
    return [get_window_from_instance(instance, start, end) for start, end in window_indices]


def convert_instances_to_windows(instances: list[dict], window_size: int, stride: int):
    new_instances = []
    for instance in instances:
        seq_len = instance['targets']['ground_truth']['segmentation'].shape[0]
        window_indices = compute_window_indices(seq_len, window_size, stride)
        new_instances += subdivide_instance_into_windows(instance, window_indices)
    return new_instances
