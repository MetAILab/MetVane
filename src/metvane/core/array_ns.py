"""Array namespace adapter.

Provides a thin abstraction over numpy and torch so that metric
implementations can be written once and work with both backends.

Only the operations that *differ* between numpy and torch are wrapped here.
Arithmetic operators (``+  -  *  **``), comparison operators (``>=  >``),
and logical operators (``&  |  ~``) are already universal and need no
adapter.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np


class NumpyNS:
    """Numpy array namespace."""

    name = "numpy"

    @staticmethod
    def sqrt(x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    @staticmethod
    def abs(x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    @staticmethod
    def sum(x: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    @staticmethod
    def mean(x: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.mean(x, axis=axis)

    @staticmethod
    def nanmean(x: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.nanmean(x, axis=axis)

    @staticmethod
    def nansum(x: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.nansum(x, axis=axis)

    @staticmethod
    def isfinite(x: np.ndarray) -> np.ndarray:
        return np.isfinite(x)

    @staticmethod
    def where(condition: np.ndarray, x: Any, y: Any) -> np.ndarray:
        return np.where(condition, x, y)

    @staticmethod
    def zeros_like(x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @staticmethod
    def ones_like(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    @staticmethod
    def full_like(x: np.ndarray, value: float) -> np.ndarray:
        return np.full_like(x, value)

    @staticmethod
    def stack(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    @staticmethod
    def broadcast_to(x: np.ndarray, shape: tuple) -> np.ndarray:
        return np.broadcast_to(x, shape)

    @staticmethod
    def as_float(x: np.ndarray) -> np.ndarray:
        if not np.issubdtype(x.dtype, np.floating):
            return x.astype(np.float64)
        return x

    @staticmethod
    def to_scalar(x: np.ndarray) -> float:
        return float(x)


def _is_empty_axis(axis: Any) -> bool:
    """``axis=()`` means "reduce over no axes" (numpy semantics).

    ``torch.sum(x, dim=())`` instead reduces over *all* dims, so the torch
    namespace must special-case it — otherwise a fully preserved grid
    (e.g. per-grid-point statistics) silently collapses to a scalar on GPU.
    """
    return axis is not None and not isinstance(axis, int) and len(tuple(axis)) == 0


class TorchNS:
    """Torch array namespace — only instantiated when torch is available."""

    name = "torch"

    @staticmethod
    def sqrt(x: Any) -> Any:
        import torch
        return torch.sqrt(x)

    @staticmethod
    def abs(x: Any) -> Any:
        import torch
        return torch.abs(x)

    @staticmethod
    def sum(x: Any, axis: Any = None) -> Any:
        import torch
        if axis is None:
            return torch.sum(x)
        if _is_empty_axis(axis):
            # numpy: np.sum(x, axis=()) is an element-wise copy (bool -> int64)
            return x.to(torch.int64) if x.dtype == torch.bool else x.clone()
        return torch.sum(x, dim=axis)

    @staticmethod
    def mean(x: Any, axis: Any = None) -> Any:
        if axis is None:
            return x.float().mean()
        if _is_empty_axis(axis):
            return x.float().clone()
        return x.float().mean(dim=axis)

    @staticmethod
    def nanmean(x: Any, axis: Any = None) -> Any:
        import torch
        if axis is None:
            return torch.nanmean(x.float())
        if _is_empty_axis(axis):
            return x.float().clone()          # NaN stays NaN, as np.nanmean(x, axis=())
        return torch.nanmean(x.float(), dim=axis)

    @staticmethod
    def nansum(x: Any, axis: Any = None) -> Any:
        import torch
        if axis is None:
            return torch.nansum(x)
        if _is_empty_axis(axis):
            return torch.where(torch.isnan(x), torch.zeros_like(x), x)   # np.nansum(x, axis=()) -> NaN->0
        return torch.nansum(x, dim=axis)

    @staticmethod
    def isfinite(x: Any) -> Any:
        import torch
        return torch.isfinite(x)

    @staticmethod
    def where(condition: Any, x: Any, y: Any) -> Any:
        import torch
        return torch.where(condition, x, y)

    @staticmethod
    def zeros_like(x: Any) -> Any:
        import torch
        return torch.zeros_like(x)

    @staticmethod
    def ones_like(x: Any) -> Any:
        import torch
        return torch.ones_like(x)

    @staticmethod
    def full_like(x: Any, value: float) -> Any:
        import torch
        return torch.full_like(x, value)

    @staticmethod
    def stack(arrays: Sequence, axis: int = 0) -> Any:
        import torch
        return torch.stack(list(arrays), dim=axis)

    @staticmethod
    def broadcast_to(x: Any, shape: tuple) -> Any:
        return x.expand(shape)

    @staticmethod
    def as_float(x: Any) -> Any:
        return x.float()

    @staticmethod
    def to_scalar(x: Any) -> float:
        return x.item()


_numpy_ns = NumpyNS()
_torch_ns: Optional[TorchNS] = None


def get_namespace(*arrays: Any) -> Union[NumpyNS, TorchNS]:
    """Return the appropriate namespace for the given arrays."""
    global _torch_ns
    for arr in arrays:
        if type(arr).__module__.startswith("torch"):
            if _torch_ns is None:
                _torch_ns = TorchNS()
            return _torch_ns
    return _numpy_ns
