"""Array namespace adapter.

Provides a thin abstraction over numpy and torch so that metric
implementations can be written once and work with both backends.

Only the operations that *differ* between numpy and torch are wrapped here.
Arithmetic operators (``+  -  *  **``), comparison operators (``>=  >``),
and logical operators (``&  |  ~``) are already universal and need no
adapter.

Dtype policy (:meth:`as_float`)
-------------------------------
* float16 / bfloat16  -> float32 (sums of squares overflow in half precision)
* float32 / float64   -> unchanged (torch float64 is **not** down-cast)
* bool / integer      -> float64 (numpy) / float32 (torch)
* ``numpy.ma.MaskedArray`` -> masked elements become NaN
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np


def _is_empty_axis(axis: Any) -> bool:
    """``axis=()`` means "reduce over no axes" (numpy semantics).

    ``torch.sum(x, dim=())`` instead reduces over *all* dims, so the torch
    namespace must special-case it — otherwise a fully preserved grid
    (e.g. per-grid-point statistics) silently collapses to a scalar on GPU.
    """
    return axis is not None and not isinstance(axis, int) and len(tuple(axis)) == 0


class NumpyNS:
    """Numpy array namespace."""

    name = "numpy"
    float64 = np.float64
    int64 = np.int64

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def sum(x, axis: Any = None, *, keepdims: bool = False, dtype: Any = None):
        return np.sum(x, axis=axis, keepdims=keepdims, dtype=dtype)

    @staticmethod
    def mean(x, axis: Any = None, *, keepdims: bool = False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def nanmean(x, axis: Any = None, *, keepdims: bool = False):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice -> NaN
            return np.nanmean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def nansum(x, axis: Any = None):
        return np.nansum(x, axis=axis)

    @staticmethod
    def isfinite(x):
        return np.isfinite(x)

    @staticmethod
    def isnan(x):
        return np.isnan(x)

    @staticmethod
    def any(x) -> bool:
        return bool(np.any(x))

    @staticmethod
    def all(x, axis: Any = None, *, keepdims: bool = False):
        return np.all(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def where(condition, x: Any, y: Any):
        return np.where(condition, x, y)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def ones_like(x):
        return np.ones_like(x)

    @staticmethod
    def full_like(x, value: float):
        return np.full_like(x, value)

    @staticmethod
    def nan_like(x):
        return np.full(np.shape(x), np.nan, dtype=np.result_type(x, np.float32))

    @staticmethod
    def stack(arrays: Sequence, axis: int = 0):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def broadcast_to(x, shape: tuple):
        return np.broadcast_to(x, shape)

    @staticmethod
    def moveaxis(x, src: int, dst: int):
        return np.moveaxis(x, src, dst)

    @staticmethod
    def sort(x, axis: int = 0):
        return np.sort(x, axis=axis)

    @staticmethod
    def cumsum(x, axis: int):
        return np.cumsum(x, axis=axis)

    @staticmethod
    def pad_last2(x, before: int, after: int):
        """Zero-pad the last two axes by (before, after) on each side."""
        width = [(0, 0)] * (x.ndim - 2) + [(before, after), (before, after)]
        return np.pad(x, width, mode="constant")

    @staticmethod
    def arange(n: int, like=None):
        return np.arange(n, dtype=np.float64 if like is None else like.dtype)

    @staticmethod
    def astype(x, dtype):
        return np.asarray(x).astype(dtype)

    @staticmethod
    def as_float(x):
        if isinstance(x, np.ma.MaskedArray):
            dt = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
            x = np.ma.filled(x.astype(dt), np.nan)
        x = np.asarray(x)
        if x.dtype == np.float16:
            return x.astype(np.float32)
        if not np.issubdtype(x.dtype, np.floating):
            return x.astype(np.float64)
        return x

    @staticmethod
    def is_float(x) -> bool:
        return np.issubdtype(np.asarray(x).dtype, np.floating)

    @staticmethod
    def to_scalar(x) -> float:
        return float(x)


class TorchNS:
    """Torch array namespace — only instantiated when torch is available."""

    name = "torch"

    def __init__(self):
        import torch
        self.float64 = torch.float64
        self.int64 = torch.int64

    @staticmethod
    def _dims(x, axis):
        if axis is None:
            return tuple(range(x.dim()))
        if isinstance(axis, int):
            return (axis,)
        return tuple(int(a) for a in axis)

    @staticmethod
    def sqrt(x):
        import torch
        return torch.sqrt(x)

    @staticmethod
    def abs(x):
        import torch
        return torch.abs(x)

    def sum(self, x, axis: Any = None, *, keepdims: bool = False, dtype: Any = None):
        import torch
        if _is_empty_axis(axis):
            # numpy: np.sum(x, axis=()) is an element-wise copy (bool -> int64)
            out = x.to(torch.int64) if x.dtype == torch.bool else x.clone()
            return out if dtype is None else out.to(dtype)
        if axis is None and not keepdims:
            return torch.sum(x, dtype=dtype)
        return torch.sum(x, dim=self._dims(x, axis), keepdim=keepdims, dtype=dtype)

    def mean(self, x, axis: Any = None, *, keepdims: bool = False):
        x = x if x.is_floating_point() else x.float()
        if _is_empty_axis(axis):
            return x.clone()
        if axis is None and not keepdims:
            return x.mean()
        return x.mean(dim=self._dims(x, axis), keepdim=keepdims)

    def nanmean(self, x, axis: Any = None, *, keepdims: bool = False):
        import torch
        x = x if x.is_floating_point() else x.float()
        if _is_empty_axis(axis):
            return x.clone()          # NaN stays NaN, as np.nanmean(x, axis=())
        if axis is None and not keepdims:
            return torch.nanmean(x)
        return torch.nanmean(x, dim=self._dims(x, axis), keepdim=keepdims)

    def nansum(self, x, axis: Any = None):
        import torch
        if axis is None:
            return torch.nansum(x)
        if _is_empty_axis(axis):
            return torch.where(torch.isnan(x), torch.zeros_like(x), x)   # np.nansum(x, axis=()) -> NaN->0
        return torch.nansum(x, dim=self._dims(x, axis))

    @staticmethod
    def isfinite(x):
        import torch
        return torch.isfinite(x)

    @staticmethod
    def isnan(x):
        import torch
        return torch.isnan(x)

    @staticmethod
    def any(x) -> bool:
        import torch
        return bool(torch.any(x))

    def all(self, x, axis: Any = None, *, keepdims: bool = False):
        import torch
        if axis is None and not keepdims:
            return torch.all(x)
        out = x
        for d in sorted(self._dims(x, axis), reverse=True):
            out = torch.all(out, dim=d, keepdim=keepdims)
        return out

    @staticmethod
    def where(condition, x: Any, y: Any):
        import torch
        return torch.where(condition, x, y)

    @staticmethod
    def zeros_like(x):
        import torch
        return torch.zeros_like(x)

    @staticmethod
    def ones_like(x):
        import torch
        return torch.ones_like(x)

    @staticmethod
    def full_like(x, value: float):
        import torch
        return torch.full_like(x, value)

    @staticmethod
    def nan_like(x):
        import torch
        dt = x.dtype if x.is_floating_point() else torch.float32
        return torch.full(tuple(x.shape), float("nan"), dtype=dt, device=x.device)

    @staticmethod
    def stack(arrays: Sequence, axis: int = 0):
        import torch
        return torch.stack(list(arrays), dim=axis)

    @staticmethod
    def broadcast_to(x, shape: tuple):
        return x.expand(shape)

    @staticmethod
    def moveaxis(x, src: int, dst: int):
        import torch
        return torch.movedim(x, src, dst)

    @staticmethod
    def sort(x, axis: int = 0):
        import torch
        return torch.sort(x, dim=axis).values

    @staticmethod
    def cumsum(x, axis: int):
        import torch
        return torch.cumsum(x, dim=axis)

    @staticmethod
    def pad_last2(x, before: int, after: int):
        import torch.nn.functional as F
        return F.pad(x, (before, after, before, after), mode="constant", value=0)

    @staticmethod
    def arange(n: int, like=None):
        import torch
        if like is None:
            return torch.arange(n, dtype=torch.float64)
        return torch.arange(n, dtype=like.dtype, device=like.device)

    @staticmethod
    def astype(x, dtype):
        return x.to(dtype)

    @staticmethod
    def as_float(x):
        import torch
        if x.dtype in (torch.float16, torch.bfloat16):
            return x.float()
        if x.is_floating_point():
            return x
        return x.float()

    @staticmethod
    def is_float(x) -> bool:
        return x.is_floating_point()

    @staticmethod
    def to_scalar(x) -> float:
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
