"""Type definitions for MetVane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import torch/xarray only for type checkers
    import torch
    import xarray as xr

    ArrayLike = Union[np.ndarray, "torch.Tensor", "xr.DataArray"]
else:
    ArrayLike = Any

AxisType = Optional[Union[int, Sequence[int]]]
FlexibleDims = Optional[Union[str, Sequence[str]]]


def normalize_axis(axis: AxisType, ndim: int) -> Optional[tuple[int, ...]]:
    """Normalize *axis* (see :func:`metvane.core.prepare.normalize_axis`)."""
    from .prepare import normalize_axis as _normalize
    return _normalize(axis, ndim)
