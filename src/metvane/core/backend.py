"""Backend detection, data conversion and device management."""

from __future__ import annotations

from enum import Enum
from typing import Any


class Backend(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    XARRAY = "xarray"


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _is_xarray(x: Any) -> bool:
    return type(x).__module__.startswith("xarray")


def detect_backend(*arrays: Any) -> Backend:
    """Infer the backend from input arrays.

    Priority: torch > xarray > numpy.
    """
    for arr in arrays:
        if _is_torch(arr):
            return Backend.TORCH
        if _is_xarray(arr):
            return Backend.XARRAY
    return Backend.NUMPY


def convert(x: Any, target: Backend, *, device: Any = None) -> Any:
    """Convert *x* to the *target* backend."""
    source = detect_backend(x)
    if source == target:
        if target == Backend.TORCH and device is not None:
            return x.to(device)
        return x

    if target == Backend.NUMPY:
        if source == Backend.TORCH:
            return x.detach().cpu().numpy()
        if source == Backend.XARRAY:
            return x.values
        return x

    if target == Backend.TORCH:
        import torch

        np_arr = convert(x, Backend.NUMPY)
        t = torch.as_tensor(np_arr)
        return t.to(device) if device is not None else t

    if target == Backend.XARRAY:
        import xarray as xr

        np_arr = convert(x, Backend.NUMPY)
        return xr.DataArray(np_arr)

    return x  # pragma: no cover


def ensure_same_backend(*arrays: Any) -> tuple[Any, ...]:
    """Ensure all arrays share the same backend.

    If any array is a torch Tensor the others are promoted to torch on the
    same device.  Otherwise everything is converted to numpy.
    """
    backends = [detect_backend(a) for a in arrays]

    if Backend.TORCH in backends:
        device = None
        for a, be in zip(arrays, backends):
            if be == Backend.TORCH:
                device = a.device
                break
        return tuple(convert(a, Backend.TORCH, device=device) for a in arrays)

    return tuple(convert(a, Backend.NUMPY) for a in arrays)
