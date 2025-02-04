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


def greedy_bipartite_matching(iou_matrix: Tensor) -> Tensor:
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
        # global_max_iou = torch.max(active_ious)
        # if global_max_iou <= 0:
        #     break

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

        # # Find ALL coordinates with max_iou and pick the first pair
        # max_mask = (active_ious == max_iou)
        # rows, cols = torch.where(max_mask)
        # row_idx = rows[0]  # First occurrence
        # col_idx = cols[0]
        #
        # # Map back to original indices
        # remaining_pred = torch.where(pred_mask)[0]
        # remaining_gt = torch.where(gt_mask)[0]
        # pred_idx = remaining_pred[row_idx]
        # gt_idx = remaining_gt[col_idx]
        #
        # # Update masks and results
        # matched_ious.append(iou_matrix[pred_idx, gt_idx])
        # pred_mask[pred_idx] = False
        # gt_mask[gt_idx] = False

    return torch.stack(matched_ious)


def hungarian_bipartite_matching(iou_matrix: Tensor, return_indices: bool = False):
    cost_matrix = 1 - iou_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if return_indices:
        return row_ind, col_ind
    selected_ious = iou_matrix[row_ind, col_ind]
    selected_ious = selected_ious[selected_ious > 0]
    return selected_ious


def tp_fp_fn(
    pred_segments: Tensor,
    gt_segments: Tensor,
    thresholds: Tensor,
    algorithm: str = "hungarian",
):
    n_pred, n_gt = pred_segments.shape[0], gt_segments.shape[0]
    n_thresh = thresholds.shape[0]
    device = pred_segments.device
    if n_pred == 0:
        tp = torch.zeros(n_thresh, device=device)
        fp = torch.zeros(n_thresh, device=device)
        fn = torch.full((n_thresh,), fill_value=n_gt, device=device)
        return tp, fp, fn
    if n_gt == 0:
        tp = torch.zeros(n_thresh, device=device)
        fp = torch.full((n_thresh,), fill_value=n_pred, device=device)
        fn = torch.zeros(n_thresh, device=device)
        return tp, fp, fn
    iou_matrix = compute_iou_matrix(pred_segments, gt_segments)
    if algorithm == "greedy":
        matched_ious = greedy_bipartite_matching(iou_matrix)
    elif algorithm == "hungarian":
        matched_ious = hungarian_bipartite_matching(iou_matrix)
    else:
        raise ValueError(f"Unknown bipartite matching algorithm: {algorithm}")

    matches = matched_ious[None, :] >= thresholds[:, None]
    tp = matches.sum(dim=1)  # All matched IoUs over the given thresholds
    fp = n_pred - tp  # All predictions either matched (via tp) or excess
    fn = n_gt - tp  # All GTs not covered by matches
    return tp, fp, fn


if __name__ == "__main__":
    pred = torch.tensor([
        [1, 4],
        [0, 9],
        [3, 5],
        [5, 10],
    ])
    gt = torch.tensor([
        [1, 4],
        [0, 9],
        [3, 5],
        [5, 12],
    ])

    # gt = torch.tensor(
    #     [
    #         [0, 10],
    #         [1, 10],
    #         [14, 17],
    #     ]
    # )
    # pred = torch.tensor(
    #     [
    #         [0, 8],
    #         [13, 17],
    #         [14, 16],
    #     ]
    # )

    print(tp_fp_fn(pred, gt, torch.tensor([0.4, 0.5, 0.8]), algorithm='greedy'))

