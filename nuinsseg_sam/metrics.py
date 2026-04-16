from __future__ import annotations

from itertools import permutations
from typing import Dict, Iterable, List, Tuple

import numpy as np

scipy_linear_sum_assignment = None
_scipy_import_attempted = False


def _load_scipy_linear_sum_assignment():
    global scipy_linear_sum_assignment, _scipy_import_attempted
    if _scipy_import_attempted:
        return scipy_linear_sum_assignment
    _scipy_import_attempted = True
    try:
        from scipy.optimize import linear_sum_assignment as scipy_assignment

        scipy_linear_sum_assignment = scipy_assignment
    except Exception:
        scipy_linear_sum_assignment = None
    return scipy_linear_sum_assignment


def _greedy_linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if cost.shape[0] <= 8 and cost.shape[1] <= 8:
        rows = np.arange(cost.shape[0], dtype=int)
        cols = np.arange(cost.shape[1], dtype=int)
        choose_cols = cols if cost.shape[0] <= cost.shape[1] else rows
        best_pairs: List[Tuple[int, int]] = []
        best_cost = float("inf")
        if cost.shape[0] <= cost.shape[1]:
            for perm in permutations(cols.tolist(), len(rows)):
                total = sum(cost[row, col] for row, col in enumerate(perm))
                if total < best_cost:
                    best_cost = float(total)
                    best_pairs = list(zip(rows.tolist(), perm))
        else:
            for perm in permutations(rows.tolist(), len(cols)):
                total = sum(cost[row, col] for row, col in zip(perm, cols.tolist()))
                if total < best_cost:
                    best_cost = float(total)
                    best_pairs = list(zip(list(perm), cols.tolist()))
        if not best_pairs:
            return np.array([], dtype=int), np.array([], dtype=int)
        row_ind = np.array([pair[0] for pair in best_pairs], dtype=int)
        col_ind = np.array([pair[1] for pair in best_pairs], dtype=int)
        return row_ind, col_ind

    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    pairs: List[Tuple[int, int]] = []
    flat_pairs = [
        (float(cost[row, col]), int(row), int(col))
        for row in range(cost.shape[0])
        for col in range(cost.shape[1])
    ]
    flat_pairs.sort(key=lambda item: item[0])
    for _, row, col in flat_pairs:
        if row not in remaining_rows or col not in remaining_cols:
            continue
        remaining_rows.remove(row)
        remaining_cols.remove(col)
        pairs.append((row, col))
        if not remaining_rows or not remaining_cols:
            break
    row_ind = np.array([pair[0] for pair in pairs], dtype=int)
    col_ind = np.array([pair[1] for pair in pairs], dtype=int)
    return row_ind, col_ind


def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scipy_assignment = _load_scipy_linear_sum_assignment()
    if scipy_assignment is not None:
        return scipy_assignment(cost)
    return _greedy_linear_sum_assignment(cost)


def apply_ignore(mask: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    output = np.array(mask, copy=True)
    output[ignore_mask.astype(bool)] = 0
    return output


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.int32)
    labels = [label for label in np.unique(mask) if label > 0]
    for new_label, old_label in enumerate(labels, start=1):
        output[mask == old_label] = new_label
    return output


def dice_score(
    pred_binary: np.ndarray, target_binary: np.ndarray, ignore_mask: np.ndarray | None = None
) -> float:
    pred = pred_binary.astype(bool)
    target = target_binary.astype(bool)
    if ignore_mask is not None:
        valid = ~ignore_mask.astype(bool)
        pred = pred & valid
        target = target & valid
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection) / max(pred_sum + target_sum, 1))


def _labels(mask: np.ndarray) -> List[int]:
    return [int(label) for label in np.unique(mask) if label > 0]


def _pairwise_stats(gt: np.ndarray, pred: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    gt_ids = _labels(gt)
    pred_ids = _labels(pred)
    intersections = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    unions = np.zeros_like(intersections)
    if not gt_ids or not pred_ids:
        return gt_ids, pred_ids, intersections, unions
    for gt_index, gt_id in enumerate(gt_ids):
        gt_mask = gt == gt_id
        overlapping_pred_ids = [pred_id for pred_id in np.unique(pred[gt_mask]) if pred_id > 0]
        for pred_id in overlapping_pred_ids:
            pred_index = pred_ids.index(int(pred_id))
            pred_mask = pred == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            intersections[gt_index, pred_index] = intersection
            unions[gt_index, pred_index] = union
    return gt_ids, pred_ids, intersections, unions


def aggregate_jaccard_index(
    gt_instances: np.ndarray, pred_instances: np.ndarray, ignore_mask: np.ndarray | None = None
) -> float:
    gt = relabel_sequential(
        apply_ignore(gt_instances, ignore_mask if ignore_mask is not None else np.zeros_like(gt_instances))
    )
    pred = relabel_sequential(
        apply_ignore(pred_instances, ignore_mask if ignore_mask is not None else np.zeros_like(pred_instances))
    )
    gt_ids, pred_ids, intersections, unions = _pairwise_stats(gt, pred)
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0

    ious = np.divide(
        intersections,
        np.maximum(unions, 1),
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    cost = 1.0 - ious
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_gt = set()
    matched_pred = set()
    intersection_sum = 0.0
    union_sum = 0.0

    for gt_index, pred_index in zip(row_ind, col_ind):
        if intersections[gt_index, pred_index] <= 0:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        intersection_sum += intersections[gt_index, pred_index]
        union_sum += unions[gt_index, pred_index]

    for gt_index, gt_id in enumerate(gt_ids):
        if gt_index not in matched_gt:
            union_sum += float((gt == gt_id).sum())
    for pred_index, pred_id in enumerate(pred_ids):
        if pred_index not in matched_pred:
            union_sum += float((pred == pred_id).sum())
    if union_sum == 0:
        return 1.0
    return float(intersection_sum / union_sum)


def panoptic_quality(
    gt_instances: np.ndarray, pred_instances: np.ndarray, ignore_mask: np.ndarray | None = None
) -> float:
    gt = relabel_sequential(
        apply_ignore(gt_instances, ignore_mask if ignore_mask is not None else np.zeros_like(gt_instances))
    )
    pred = relabel_sequential(
        apply_ignore(pred_instances, ignore_mask if ignore_mask is not None else np.zeros_like(pred_instances))
    )
    gt_ids, pred_ids, intersections, unions = _pairwise_stats(gt, pred)
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0

    ious = np.divide(
        intersections,
        np.maximum(unions, 1),
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    match_scores = np.where(ious > 0.5, ious, 0.0)
    if np.any(match_scores > 0):
        cost = -match_scores
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)

    tp = 0
    matched_iou = 0.0
    matched_gt = set()
    matched_pred = set()
    for gt_index, pred_index in zip(row_ind, col_ind):
        if ious[gt_index, pred_index] <= 0.5:
            continue
        tp += 1
        matched_iou += ious[gt_index, pred_index]
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    if tp == 0:
        return 0.0
    segmentation_quality = matched_iou / tp
    recognition_quality = tp / (tp + 0.5 * fp + 0.5 * fn)
    return float(segmentation_quality * recognition_quality)


def summarize_metric_rows(rows: Iterable[Dict[str, float]], metric_names: Iterable[str]) -> Dict[str, float]:
    rows = list(rows)
    summary: Dict[str, float] = {"count": float(len(rows))}
    for metric_name in metric_names:
        values = np.array([row[metric_name] for row in rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean()) if len(values) else 0.0
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
    return summary
