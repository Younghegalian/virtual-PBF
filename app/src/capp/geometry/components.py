from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def remove_small_components(
    binary: NDArray[np.bool_],
    *,
    min_size: int = 50,
    connectivity: int = 1,
) -> NDArray[np.bool_]:
    """Keep connected voxel components that meet the minimum size."""
    try:
        from scipy import ndimage
    except ImportError:
        return np.asarray(binary, dtype=bool)

    data = np.asarray(binary, dtype=bool)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, count = ndimage.label(data, structure=structure)
    if count == 0:
        return np.zeros_like(data, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(sizes >= min_size)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(data, dtype=bool)
    return np.isin(labeled, keep_labels)
