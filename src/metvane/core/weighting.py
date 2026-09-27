"""Meteorology-specific weighting and region utilities."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from .array_ns import get_namespace
from .backend import convert, Backend


REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (-90, 90, 0, 360),
    "NH": (20, 90, 0, 360),
    "SH": (-90, -20, 0, 360),
    "tropics": (-20, 20, 0, 360),
    "NH_extratropics": (30, 90, 0, 360),
    "SH_extratropics": (-90, -30, 0, 360),
    "east_asia": (15, 55, 70, 140),
    "europe": (35, 75, -15, 45),
    "north_america": (15, 75, -145, -50),
}


def latitude_weights(
    lat: Any,
    *,
    normalize: bool = True,
) -> Any:
    """Cosine latitude area weights.

    Parameters
    ----------
    lat : array-like
        1-D latitude array in degrees.
    normalize : bool
        If *True* (default), scale weights so that their mean equals 1.

    Returns
    -------
    Same type as *lat* (numpy or torch), in *lat*'s floating dtype
    (float64 for integer / list input; float32 for integer tensors).

    Notes
    -----
    Cosines are always evaluated in float64 and cast back afterwards:
    in float32, ``cos(deg2rad(90))`` is -4.4e-8 and would be clipped to 0,
    giving the polar rows zero weight (a latitude-preserving weighted mean
    then returns 0 there).  In float64 it is 6.1e-17 > 0.
    """
    lat_np = np.asarray(convert(lat, Backend.NUMPY))
    w = np.cos(np.deg2rad(lat_np.astype(np.float64)))
    w = np.clip(w, 0.0, None)
    if normalize:
        w = w / w.mean()
    try:
        import torch
        if isinstance(lat, torch.Tensor):
            dtype = lat.dtype if lat.is_floating_point() else torch.float32
            return torch.as_tensor(w, dtype=dtype, device=lat.device)
    except ImportError:
        pass
    if np.issubdtype(lat_np.dtype, np.floating):
        return w.astype(lat_np.dtype)
    return w


def broadcast_weights(
    weights: Any,
    target_shape: tuple[int, ...],
    *,
    lat_axis: int = -2,
) -> Any:
    """Broadcast 1-D weights to *target_shape*.

    Parameters
    ----------
    weights : 1-D array
        Typically the output of :func:`latitude_weights`.
    target_shape : tuple
        The shape to broadcast to.
    lat_axis : int
        Which axis of *target_shape* corresponds to latitude.
    """
    ndim = len(target_shape)
    lat_axis_pos = lat_axis if lat_axis >= 0 else ndim + lat_axis
    shape = [1] * ndim
    shape[lat_axis_pos] = -1
    xp = get_namespace(weights)
    reshaped = weights.reshape(shape)
    return xp.broadcast_to(reshaped, target_shape)


def region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    region: Union[str, tuple[float, float, float, float]],
) -> np.ndarray:
    """Create a boolean mask for a rectangular lat/lon region.

    Parameters
    ----------
    lat, lon : 1-D arrays
        Coordinate arrays.
    region : str or 4-tuple
        Region name (see :data:`REGIONS`) or ``(lat_min, lat_max, lon_min, lon_max)``.
    """
    lat = convert(lat, Backend.NUMPY)
    lon = convert(lon, Backend.NUMPY)
    if isinstance(region, str):
        region = REGIONS[region]
    lat_min, lat_max, lon_min, lon_max = region
    lat_ok = (lat >= lat_min) & (lat <= lat_max)
    lon_ok = (lon >= lon_min) & (lon <= lon_max)
    return lat_ok[:, None] & lon_ok[None, :]
