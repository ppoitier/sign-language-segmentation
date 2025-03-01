import torch
from torch import Tensor

from scipy.optimize import linear_sum_assignment


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
    ends_1 = segments_1[:, 1].unsqueeze(1)  # Shape: (M, 1)
    starts_2 = segments_2[:, 0].unsqueeze(0)  # Shape: (1, N)
    ends_2 = segments_2[:, 1].unsqueeze(0)  # Shape: (1, N)

    # Compute the intersection
    inter_starts = torch.max(starts_1, starts_2)  # Shape: (M, N)
    inter_ends = torch.min(ends_1, ends_2)  # Shape: (M, N)
    # We add 1 to end - start because the last index is included in this code.
    intersections = torch.clamp(inter_ends - inter_starts + 1, min=0)  # Shape: (M, N)

    # Compute the union
    lengths_1 = ends_1 - starts_1 + 1  # Shape: (M, 1)
    lengths_2 = ends_2 - starts_2 + 1  # Shape: (1, N)
    unions = lengths_1 + lengths_2 - intersections  # Shape: (M, N)

    # Avoid division by zero
    iou_matrix = torch.where(
        unions > 0,
        intersections / unions,
        torch.zeros_like(intersections, dtype=torch.float32),
    )
    return iou_matrix


def greedy_bipartite_iou_matching(iou_matrix: Tensor) -> Tensor:
    device = iou_matrix.device
    n_pred, n_gt = iou_matrix.shape
    pred_mask = torch.ones(n_pred, dtype=torch.bool, device=device)
    gt_mask = torch.ones(n_gt, dtype=torch.bool, device=device)
    matched_ious = []
    for _ in range(min(n_pred, n_gt)):
        # Get active IoUs and find global max
        active_ious = iou_matrix[pred_mask, :][:, gt_mask]
        if active_ious.numel() == 0:
            break
        max_row, argmax_row = active_ious.max(dim=1)
        max_iou, argmax_iou = max_row.max(dim=0)
        if max_iou <= 0.0:
            break
        remaining_pred = torch.where(pred_mask)[0]
        remaining_gt = torch.where(gt_mask)[0]
        pred_idx = remaining_pred[argmax_iou]
        gt_idx = remaining_gt[argmax_row[argmax_iou]]
        matched_ious.append(max_iou)
        pred_mask[pred_idx] = False
        gt_mask[gt_idx] = False
    return torch.stack(matched_ious)


def hungarian_bipartite_matching(cost_matrix: Tensor):
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    return row_ind, col_ind


def tp_fp_fn(
    pred_segments: Tensor,
    gt_segments: Tensor,
    iou_thresholds: Tensor,
):
    n_pred, n_gt = pred_segments.shape[0], gt_segments.shape[0]
    n_thresh = iou_thresholds.shape[0]
    device = pred_segments.device
    if n_pred == 0:
        tp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fn = torch.full((n_thresh,), fill_value=n_gt, dtype=torch.long, device=device)
        return tp, fp, fn
    if n_gt == 0:
        tp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fp = torch.full((n_thresh,), fill_value=n_pred, dtype=torch.long, device=device)
        fn = torch.zeros(n_thresh, dtype=torch.long, device=device)
        return tp, fp, fn
    iou_matrix = compute_iou_matrix(pred_segments, gt_segments)
    cost_matrix = 1 - iou_matrix
    selected_rows, selected_columns = hungarian_bipartite_matching(cost_matrix)
    matched_ious = iou_matrix[selected_rows, selected_columns]
    matched_ious = matched_ious[matched_ious > 0]
    matches = matched_ious[None, :] >= iou_thresholds[:, None]
    tp = matches.sum(dim=1)  # All matched IoUs over the given thresholds
    fp = n_pred - tp  # All predictions either matched (via tp) or excess
    fn = n_gt - tp  # All GTs not covered by matches
    return tp, fp, fn


def compute_center_dist_matrix(segments_1: Tensor, segments_2: Tensor) -> Tensor:
    lengths_1 = segments_1[:, 1] - segments_1[:, 0]
    centers_1 = segments_1[:, 0] + lengths_1 / 2
    lengths_2 = segments_2[:, 1] - segments_2[:, 0]
    centers_2 = segments_2[:, 0] + lengths_2 / 2
    center_dist_matrix = (centers_2[:, None] - centers_1[None, :]).abs()
    return center_dist_matrix


def tp_fp_fn_center_dists(
    pred_segments: Tensor,
    gt_segments: Tensor,
    dist_thresholds: Tensor,
):
    n_pred, n_gt = pred_segments.shape[0], gt_segments.shape[0]
    n_thresh = dist_thresholds.shape[0]
    device = pred_segments.device

    if n_pred == 0:
        tp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fn = torch.full((n_thresh,), fill_value=n_gt, device=device)
        return tp, fp, fn
    if n_gt == 0:
        tp = torch.zeros(n_thresh, dtype=torch.long, device=device)
        fp = torch.full((n_thresh,), fill_value=n_pred, dtype=torch.long, device=device)
        fn = torch.zeros(n_thresh, dtype=torch.long, device=device)
        return tp, fp, fn
    dist_matrix = compute_center_dist_matrix(pred_segments, gt_segments)
    selected_rows, selected_columns = hungarian_bipartite_matching(dist_matrix)
    matched_dists = dist_matrix[selected_rows, selected_columns]
    matches = matched_dists[None, :] <= dist_thresholds[:, None]
    tp = matches.sum(dim=1)  # All matched over the given thresholds
    fp = n_pred - tp  # All predictions either matched (via tp) or excess
    fn = n_gt - tp  # All GTs not covered by matches
    return tp, fp, fn


if __name__ == "__main__":
    pred = torch.tensor([
        [1, 4],
        [0, 9],
        [3, 5],
        [5, 10],
    ], dtype=torch.float32)
    gt = torch.tensor([
        # [2, 5],
        # [3, 10],
        # [4, 14],
        # [0, 12],
        # [5, 6],
    ], dtype=torch.float32)

    # print(tp_fp_fn_center_dists(pred, gt, dist_thresholds=torch.tensor([1.0])))

    print(tp_fp_fn(pred, gt, torch.tensor([0.4, 0.5, 0.8])))

