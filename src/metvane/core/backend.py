"""Backend detection, data conversion and device management."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class Backend(Enum):
    """Compute backend of an array (``detect_backend`` / ``convert`` / ``backend=``)."""

    NUMPY = "numpy"
    TORCH = "torch"
    XARRAY = "xarray"


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _is_xarray(x: Any) -> bool:
    return type(x).__module__.startswith("xarray")


def detect_backend(*arrays: Any) -> Backend:
    """Infer the backend from input arrays.

    Priority (independent of argument order): torch > xarray > numpy.
    """
    if any(_is_torch(a) for a in arrays):
        return Backend.TORCH
    if any(_is_xarray(a) for a in arrays):
        return Backend.XARRAY
    return Backend.NUMPY


def _to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    if _is_xarray(x):
        return x.values
    if isinstance(x, np.ma.MaskedArray):
        return x                      # masks are resolved by the namespace's as_float
    return np.asarray(x)


def convert(x: Any, target: Backend, *, device: Any = None) -> Any:
    """Convert *x* to the *target* backend.

    numpy -> torch handles negative-stride views and Python scalars/lists.
    """
    source = detect_backend(x)
    if source == target and target != Backend.NUMPY:
        if target == Backend.TORCH and device is not None:
            return x.to(device)
        return x

    if target == Backend.NUMPY:
        return _to_numpy(x)

    if target == Backend.TORCH:
        import torch

        arr = _to_numpy(x)
        if isinstance(arr, np.ma.MaskedArray):
            dt = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64
            arr = np.ma.filled(arr.astype(dt), np.nan)
        if any(s < 0 for s in arr.strides):
            arr = np.ascontiguousarray(arr)
        return torch.as_tensor(arr, device=device)

    if target == Backend.XARRAY:
        import xarray as xr

        return xr.DataArray(_to_numpy(x))

    return x  # pragma: no cover


def reject_xarray(*arrays: Any, where: str = "函数式 API") -> None:
    """The numpy/torch core computes by *position*; xarray inputs must go through metvane.xr_api."""
    for a in arrays:
        if _is_xarray(a):
            raise TypeError(
                f"metvane {where} 不接受 xarray 对象（{type(a).__name__}）：这里按位置计算、会忽略坐标与维名，"
                f"纬度顺序或维顺序不一致时静默算错。请改用 metvane.xr_api（按维名/坐标对齐），"
                f"或先 xr.align(..., join='exact') 并 transpose 后传入 .values")


def ensure_same_backend(*arrays: Any) -> tuple[Any, ...]:
    """Ensure all arrays share the same backend.

    If any array is a torch Tensor the others are promoted to torch on the
    same device (all torch inputs must already share one device).  Otherwise
    everything is converted to numpy.  ``None`` entries are passed through.
    xarray objects are rejected (see :func:`reject_xarray`).
    """
    present = [a for a in arrays if a is not None]
    reject_xarray(*present)
    torch_inputs = [a for a in present if _is_torch(a)]
    if torch_inputs:
        devices = {str(a.device) for a in torch_inputs}
        if len(devices) > 1:
            raise ValueError(f"torch 输入位于不同设备 {sorted(devices)}；请先把它们放到同一设备上")
        device = torch_inputs[0].device
        return tuple(None if a is None else convert(a, Backend.TORCH, device=device) for a in arrays)
    return tuple(None if a is None else convert(a, Backend.NUMPY) for a in arrays)
