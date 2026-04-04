"""Type definitions for MetVane."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

try:
    import torch

    ArrayLike = Union[np.ndarray, torch.Tensor, "xr.DataArray"]
except ImportError:
    ArrayLike = Union[np.ndarray, "xr.DataArray"]  # type: ignore[misc]

AxisType = Optional[Union[int, Sequence[int]]]
FlexibleDims = Optional[Union[str, Sequence[str]]]


def normalize_axis(axis: AxisType, ndim: int) -> Optional[tuple[int, ...]]:
    """Normalize *axis* to a tuple of non-negative ints, or ``None``."""
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)
