"""Meteorology-specific weighting and region utilities."""

from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np

from .array_ns import get_namespace
from .backend import Backend, _is_torch, _is_xarray, convert


# (lat_min, lat_max, lon_min, lon_max); longitudes in -180..180, matched modulo 360 so that
# 0..360 grids, -180..180 grids and dateline-crossing boxes (lon_min > lon_max) all work.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (-90, 90, -180, 180),
    "NH": (20, 90, -180, 180),
    "SH": (-90, -20, -180, 180),
    "tropics": (-20, 20, -180, 180),
    "NH_extratropics": (30, 90, -180, 180),
    "SH_extratropics": (-90, -30, -180, 180),
    "east_asia": (15, 55, 70, 140),
    "europe": (35, 75, -15, 45),
    "north_america": (15, 75, -145, -50),
}


def _lat_numpy(lat: Any) -> np.ndarray:
    return np.asarray(convert(lat, Backend.NUMPY), dtype=np.float64)


def _check_lat(lat64: np.ndarray) -> None:
    if lat64.size == 0:
        raise ValueError("lat 为空")
    amax = float(np.nanmax(np.abs(lat64)))
    if amax > 90 + 1e-6:
        raise ValueError(f"纬度超出 [-90, 90]（|lat| 最大 {amax:.3f}）：请传入以度为单位的纬度（不是余纬或索引）")
    if lat64.size > 3 and 0 < amax <= np.pi / 2 + 1e-9:
        warnings.warn(f"纬度的绝对值都不超过 π/2（最大 {amax:.4f}），疑似弧度；latitude_weights 需要以度为单位的纬度",
                      UserWarning, stacklevel=3)


def _area_weights(lat64: np.ndarray) -> np.ndarray:
    """Cell-area weights ∝ |sin(φ_hi) - sin(φ_lo)| with cell edges at midpoints (clipped to ±90)."""
    order = np.argsort(lat64)
    s = lat64[order]
    if s.size == 1:
        return np.ones(1)
    mid = (s[1:] + s[:-1]) / 2
    edges = np.concatenate([[max(-90.0, s[0] - (mid[0] - s[0]))], mid, [min(90.0, s[-1] + (s[-1] - mid[-1]))]])
    edges = np.clip(edges, -90.0, 90.0)
    w_sorted = np.abs(np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1])))
    w = np.empty_like(w_sorted)
    w[order] = w_sorted
    return w


def latitude_weights(
    lat: Any,
    *,
    normalize: bool = True,
    method: str = "cos",
    dtype: Any = None,
) -> Any:
    """Latitude area weights.

    Parameters
    ----------
    lat : array-like or xr.DataArray
        1-D latitude array in **degrees**.
    normalize : bool
        If *True* (default), scale weights so that their **mean equals 1**.  When weighting
        yourself, always divide by the sum of the weights actually used, e.g.
        ``(w * e).sum() / w.sum()`` (for a regional subset, the subset's own weight sum).
    method : {'cos', 'area'}
        ``'cos'`` (default): cos(lat) at the grid points (WeatherBench convention; polar rows
        get ~0 weight).  ``'area'``: exact cell area between mid-latitude edges — preferable
        for irregular latitude spacing and for polar rows.
    dtype : optional
        Output dtype.  Default: *lat*'s floating dtype (float64 for integer / list input,
        float32 for integer tensors).

    Returns
    -------
    Same type as *lat*: numpy array, torch tensor (same device) or ``xr.DataArray``
    (same dimension and coordinate).

    Notes
    -----
    Cosines are always evaluated in float64 and cast back afterwards: in float32
    ``cos(deg2rad(90))`` is -4.4e-8 and would be clipped to 0.
    """
    lat64 = _lat_numpy(lat)
    if lat64.ndim != 1:
        raise ValueError(f"lat 必须是 1 维，得到形状 {lat64.shape}")
    _check_lat(lat64)
    if method == "cos":
        w = np.clip(np.cos(np.deg2rad(lat64)), 0.0, None)
        d = np.diff(lat64)
        if d.size > 1 and np.ptp(np.abs(d)) > 1e-3 * np.max(np.abs(d)):
            warnings.warn("纬度间隔不均匀：cos(lat) 点权重不等于单元面积，建议 method='area'",
                          UserWarning, stacklevel=2)
    elif method == "area":
        w = _area_weights(lat64)
    else:
        raise ValueError(f"method={method!r} 不受支持，可选 'cos' / 'area'")
    if normalize:
        w = w / w.mean()

    if _is_xarray(lat):
        import xarray as xr
        name = lat.dims[0]
        out = xr.DataArray(w if dtype is None else w.astype(dtype), dims=(name,),
                           coords={name: lat.values}, name="latitude_weights")
        if dtype is None and np.issubdtype(lat.dtype, np.floating):
            out = out.astype(lat.dtype)
        return out
    if _is_torch(lat):
        import torch
        tdtype = dtype if dtype is not None else (lat.dtype if lat.is_floating_point() else torch.float32)
        return torch.as_tensor(w, dtype=tdtype, device=lat.device)
    src = np.asarray(lat)
    if dtype is not None:
        return w.astype(dtype)
    if np.issubdtype(src.dtype, np.floating):
        return w.astype(src.dtype)
    return w


def broadcast_weights(
    weights: Any,
    target_shape: tuple[int, ...],
    *,
    lat_axis: int = -2,
) -> Any:
    """Broadcast 1-D (latitude) weights to *target_shape* along *lat_axis* (returns a view).

    Raises if *lat_axis* is out of range or the weight length does not match
    ``target_shape[lat_axis]``.
    """
    target_shape = tuple(int(s) for s in target_shape)
    ndim = len(target_shape)
    if not -ndim <= lat_axis < ndim:
        raise ValueError(f"lat_axis={lat_axis} 超出 {ndim} 维目标形状 {target_shape} 的范围")
    pos = lat_axis % ndim
    if _is_xarray(weights):
        weights = weights.values
    if not hasattr(weights, "shape"):
        weights = np.asarray(weights)
    n = int(np.prod(tuple(weights.shape)))
    if len(tuple(weights.shape)) != 1:
        raise ValueError(f"broadcast_weights 需要 1 维权重，得到形状 {tuple(weights.shape)}")
    if n != target_shape[pos]:
        raise ValueError(f"权重长度 {n} 与 target_shape[{lat_axis}]={target_shape[pos]} 不一致（lat_axis 指错了轴？）")
    shape = [1] * ndim
    shape[pos] = -1
    xp = get_namespace(weights)
    return xp.broadcast_to(weights.reshape(shape), target_shape)


def region_mask(
    lat: Any,
    lon: Any,
    region: Union[str, tuple[float, float, float, float]],
) -> Any:
    """Boolean mask for a rectangular lat/lon region.

    Parameters
    ----------
    lat, lon : 1-D arrays (mask shape ``(nlat, nlon)``) or 2-D arrays of the same shape
        (curvilinear grids; mask has that shape).  Longitudes may use 0..360 or -180..180.
    region : str or 4-tuple
        Name in :data:`REGIONS` or ``(lat_min, lat_max, lon_min, lon_max)``.  Longitudes are
        compared modulo 360; ``lon_min > lon_max`` selects a box crossing the dateline and a
        span of 360° or more selects all longitudes.

    Returns a numpy bool array, or a torch bool tensor on *lat*'s device for torch input.
    An empty selection raises for built-in regions and warns for custom ones.
    """
    builtin = isinstance(region, str)
    if builtin:
        if region not in REGIONS:
            raise KeyError(f"未知区域 {region!r}，可选 {sorted(REGIONS)}")
        box = REGIONS[region]
    else:
        box = tuple(float(v) for v in region)
        if len(box) != 4:
            raise ValueError("region 需为 (lat_min, lat_max, lon_min, lon_max)")
    lat_min, lat_max, lon_min, lon_max = box
    if lat_min > lat_max:
        raise ValueError(f"lat_min={lat_min} > lat_max={lat_max}")

    la = np.asarray(convert(lat, Backend.NUMPY), dtype=np.float64)
    lo = np.asarray(convert(lon, Backend.NUMPY), dtype=np.float64)
    span = lon_max - lon_min
    if span >= 360:
        def lon_ok(x):
            return np.ones(x.shape, dtype=bool)
    else:
        width = np.mod(span, 360.0)

        def lon_ok(x):
            return np.mod(x - lon_min, 360.0) <= width + 1e-9

    if la.ndim == 1 and lo.ndim == 1:
        mask = ((la >= lat_min) & (la <= lat_max))[:, None] & lon_ok(lo)[None, :]
    elif la.shape == lo.shape:
        mask = (la >= lat_min) & (la <= lat_max) & lon_ok(lo)
    else:
        raise ValueError(f"lat/lon 须同为 1 维，或同形状的 2 维数组；得到 {la.shape} 与 {lo.shape}")

    if not mask.any():
        msg = (f"区域 {region!r} 在该网格上没有选中任何格点（网格纬度 {la.min():.2f}..{la.max():.2f}，"
               f"经度 {lo.min():.2f}..{lo.max():.2f}）")
        if builtin:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    if _is_torch(lat):
        import torch
        return torch.as_tensor(mask, device=lat.device)
    return mask
