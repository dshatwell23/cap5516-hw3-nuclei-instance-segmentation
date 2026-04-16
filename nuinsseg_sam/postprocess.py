from __future__ import annotations

import numpy as np

from .metrics import relabel_sequential


def probability_to_binary(
    probability_map: np.ndarray, threshold: float, ignore_mask: np.ndarray | None = None
) -> np.ndarray:
    binary = probability_map >= threshold
    if ignore_mask is not None:
        binary = np.logical_and(binary, ~ignore_mask.astype(bool))
    return binary


def probability_to_instances(
    probability_map: np.ndarray,
    threshold: float,
    min_object_size: int,
    peak_min_distance: int,
    peak_threshold_abs: float,
    ignore_mask: np.ndarray | None = None,
) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    foreground = probability_to_binary(probability_map, threshold=threshold, ignore_mask=ignore_mask)
    if min_object_size > 1:
        foreground = remove_small_objects(foreground, min_size=min_object_size)
    if not np.any(foreground):
        return np.zeros_like(probability_map, dtype=np.int32)

    distance = ndi.distance_transform_edt(foreground)
    peak_coordinates = peak_local_max(
        distance,
        labels=foreground.astype(np.uint8),
        min_distance=max(1, peak_min_distance),
        threshold_abs=peak_threshold_abs,
        exclude_border=False,
    )
    markers = np.zeros_like(probability_map, dtype=np.int32)
    for marker_id, (row, col) in enumerate(peak_coordinates, start=1):
        markers[row, col] = marker_id
    if markers.max() == 0:
        markers, _ = ndi.label(foreground)
    instances = watershed(-distance, markers, mask=foreground)
    if min_object_size > 1:
        cleaned = np.zeros_like(instances, dtype=np.int32)
        next_label = 1
        for label_id in [label for label in np.unique(instances) if label > 0]:
            label_mask = instances == label_id
            if int(label_mask.sum()) < min_object_size:
                continue
            cleaned[label_mask] = next_label
            next_label += 1
        instances = cleaned
    return relabel_sequential(instances)
