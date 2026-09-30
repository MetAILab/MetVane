"""Unified input preparation shared by every metric, accumulator and the xarray layer.

One place decides, for all entry points:

* backend / device handling (``backend=`` / ``device=``; xarray rejected in the positional core)
* dtype policy (see :mod:`metvane.core.array_ns`)
* axis normalisation (negative axes, ranges, duplicates, ``()`` = no reduction)
* shape checks (forecast and observation must have identical shapes — no silent broadcasting)
* auxiliary arrays (weights, climatology, masks) converted to the data's backend/device/dtype
  and checked for unambiguous broadcasting (a 1-D array is only accepted for 1-D data)
* the validity mask (NaN = missing; ``inf`` is a value, not missing) and safe division
"""

from __future__ import annotations

import operator as _op
import weakref
from collections import OrderedDict
from typing import Any, Optional, Sequence

import numpy as np

from .array_ns import get_namespace
from .backend import Backend, convert, ensure_same_backend

try:  # numpy >= 1.25
    from numpy.exceptions import AxisError
except ImportError:  # pragma: no cover
    AxisError = np.AxisError  # type: ignore[attr-defined]

WEIGHT_MODES = ("mean", "multiply")

OPS = {">=": _op.ge, ">": _op.gt, "<=": _op.le, "<": _op.lt, "==": _op.eq}


# ----------------------------------------------------------------------------- axis / shape
def normalize_axis(axis: Any, ndim: int, *, name: str = "axis") -> Optional[tuple[int, ...]]:
    """Normalise *axis* to a sorted tuple of non-negative ints.

    ``None`` -> ``None`` (reduce all); ``()`` -> ``()`` (reduce nothing); ints, numpy integers,
    lists and tuples are accepted.  Out-of-range axes raise ``AxisError``, duplicates ``ValueError``.
    """
    if axis is None:
        return None
    if isinstance(axis, (int, np.integer)):
        axis = (int(axis),)
    try:
        axis = tuple(int(a) for a in axis)
    except TypeError as e:
        raise TypeError(f"{name} 必须是 int、int 序列或 None，得到 {axis!r}") from e
    out = []
    for a in axis:
        if not -ndim <= a < ndim:
            raise AxisError(f"{name}={a} 超出 {ndim} 维数据的范围")
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} 含重复轴: {axis}")
    return tuple(sorted(out))


def complement_axes(keep: Sequence[int], ndim: int) -> tuple[int, ...]:
    keep = set(keep)
    return tuple(i for i in range(ndim) if i not in keep)


def check_same_shape(ref: Any, *others: tuple[str, Any], ref_name: str = "fcst") -> None:
    for name, x in others:
        if tuple(x.shape) != tuple(ref.shape):
            raise ValueError(f"{name} 形状 {tuple(x.shape)} 与 {ref_name} 形状 {tuple(ref.shape)} 不一致；"
                             f"MetVane 不做隐式广播（(N,) 与 (N,1) 混用会让计数膨胀为 N²）")


# ----------------------------------------------------------------------------- backend
def resolve_backend(arrays: Sequence[Any], backend: Optional[str], device: Any) -> tuple[Any, ...]:
    """Apply the ``backend=`` / ``device=`` arguments of the functional API."""
    from .backend import reject_xarray
    reject_xarray(*[a for a in arrays if a is not None])
    if backend is None:
        if device is not None:
            raise ValueError("只给了 device= 而没有 backend=：请同时指定 backend='torch'，"
                             "或直接传入目标设备上的 torch 张量")
        return tuple(arrays)
    if backend == "xarray":
        import warnings
        warnings.warn("backend='xarray' 已弃用，将在 0.3 中报错：函数式 API 按位置计算，结果为 numpy；"
                      "xarray 输入请使用 metvane.xr_api", DeprecationWarning, stacklevel=3)
        backend = "numpy"
    target = Backend(backend)
    return tuple(None if a is None else convert(a, target, device=device) for a in arrays)


def prepare(*arrays: Any, backend: Optional[str] = None, device: Any = None, as_float: bool = True):
    """Common entry: backend/device -> same backend (xarray rejected) -> dtype policy.

    Returns ``(xp, *arrays)``; ``None`` entries stay ``None``.
    """
    arrays = resolve_backend(arrays, backend, device)
    arrays = ensure_same_backend(*arrays)
    xp = get_namespace(*[a for a in arrays if a is not None])
    if as_float:
        arrays = tuple(None if a is None else xp.as_float(a) for a in arrays)
    else:
        arrays = tuple(None if a is None else _resolve_masked(xp, a) for a in arrays)
    return (xp, *arrays)


def _resolve_masked(xp, a):
    if isinstance(a, np.ma.MaskedArray):
        return xp.as_float(a)
    return a


def align_to(x: Any, like: Any, xp, *, name: str, dtype: Any = None, allow_1d: bool = False):
    """Convert an auxiliary array (weights / climatology / mask) to *like*'s backend, device and dtype.

    Only the compact array is converted (broadcast views are not materialised).  Broadcasting to
    *like* must follow numpy rules exactly to ``like.shape``.  A 1-D array for N-D data (N > 1) is
    rejected because numpy would align it with the **last** axis (longitude), not latitude —
    use ``broadcast_weights(w, shape, lat_axis=...)`` or ``w[:, None]``.
    """
    if x is None:
        return None
    from .backend import _is_torch, reject_xarray
    reject_xarray(x, where=f"函数式 API 的 {name}")
    if xp.name == "torch":
        import torch
        if not _is_torch(x):
            x = _cached_to_torch(x, like)
        elif x.device != like.device:
            raise ValueError(f"{name} 在设备 {x.device}，数据在 {like.device}；请先放到同一设备")
        target = dtype if dtype is not None else (like.dtype if like.is_floating_point() else torch.float32)
        if x.dtype != torch.bool or dtype is not None:
            x = x.to(target)
    else:
        if _is_torch(x):
            x = convert(x, Backend.NUMPY)
        x = xp.as_float(x) if dtype is None else np.asarray(x).astype(dtype)
        if dtype is None and x.dtype != like.dtype and np.issubdtype(like.dtype, np.floating):
            x = x.astype(like.dtype)
    shape = tuple(x.shape)
    if len(shape) == 1 and len(like.shape) > 1 and not allow_1d and shape[0] != 1:
        raise ValueError(f"{name} 是 1 维 (长度 {shape[0]})，而数据是 {len(like.shape)} 维 {tuple(like.shape)}：numpy "
                         f"会把它对齐到最后一维（经度）而不是纬度，方形网格上会静默算错。请用 "
                         f"broadcast_weights(w, data.shape, lat_axis=-2) 或 w[:, None]")
    try:
        out_shape = np.broadcast_shapes(shape, tuple(like.shape))
    except ValueError as e:
        raise ValueError(f"{name} 形状 {shape} 不能广播到数据形状 {tuple(like.shape)}") from e
    if tuple(out_shape) != tuple(like.shape):
        raise ValueError(f"{name} 形状 {shape} 会把数据 {tuple(like.shape)} 广播成 {tuple(out_shape)}")
    return x


_TORCH_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_TORCH_CACHE_SIZE = 16


def _cached_to_torch(x: Any, like: Any):
    """numpy -> torch (on *like*'s device and float dtype) with a small cache.

    Repeated calls with the same weight / climatology array (typical in per-lead loops) reuse
    the device copy instead of re-uploading it.  Entries are keyed by object identity and
    validated through a weak reference and the array's (shape, dtype, data pointer); read-only
    arrays and arrays whose buffer is unchanged are safe to reuse.  Mutating a cached array in
    place is not detected — pass a new array (or call ``clear_conversion_cache()``).
    """
    import torch
    arr = x if isinstance(x, np.ndarray) and not isinstance(x, np.ma.MaskedArray) else None
    target = like.dtype if like.is_floating_point() else torch.float32
    if arr is None or arr.size < 1024:
        return convert(x, Backend.TORCH, device=like.device)
    key = (id(arr), str(like.device), str(target))
    hit = _TORCH_CACHE.get(key)
    sig = (arr.shape, arr.dtype.str, arr.__array_interface__["data"][0], arr.strides)
    if hit is not None:
        ref, hsig, t = hit
        if ref() is arr and hsig == sig:
            _TORCH_CACHE.move_to_end(key)
            return t
    t = convert(arr, Backend.TORCH, device=like.device)
    if t.is_floating_point() or t.dtype != torch.bool:
        t = t.to(target)
    try:
        _TORCH_CACHE[key] = (weakref.ref(arr), sig, t)
    except TypeError:          # object without weakref support: do not cache
        return t
    while len(_TORCH_CACHE) > _TORCH_CACHE_SIZE:
        _TORCH_CACHE.popitem(last=False)
    return t


def clear_conversion_cache() -> None:
    """Drop cached numpy -> torch conversions of weights / climatology."""
    _TORCH_CACHE.clear()


def align_mask(mask: Any, like: Any, xp):
    """Boolean evaluation mask (True = evaluate) aligned to *like*'s backend/device; shape-checked."""
    if mask is None:
        return None
    return _align_bool(mask, like, xp)


def _align_bool(mask, like, xp):
    from .backend import _is_torch, reject_xarray
    reject_xarray(mask, where="函数式 API 的 mask")
    if xp.name == "torch":
        import torch
        m = mask if _is_torch(mask) else convert(mask, Backend.TORCH, device=like.device)
        m = m.to(device=like.device).bool()
    else:
        m = np.asarray(convert(mask, Backend.NUMPY)).astype(bool)
    shape = tuple(m.shape)
    if len(shape) == 1 and len(like.shape) > 1 and shape[0] != 1:
        raise ValueError(f"mask 是 1 维，而数据是 {len(like.shape)} 维：请给出可明确广播的形状（如 (nlat, nlon)）")
    if tuple(np.broadcast_shapes(shape, tuple(like.shape))) != tuple(like.shape):
        raise ValueError(f"mask 形状 {shape} 不能广播到数据形状 {tuple(like.shape)}")
    return m


# ----------------------------------------------------------------------------- validity / division
def validity(xp, *arrays, mask=None, skipna: bool = True):
    """Validity mask: not-NaN in every array (``inf`` counts as a value), combined with *mask*.

    Returns ``None`` when every element is valid (fast path, nothing to mask).
    """
    valid = None
    if skipna:
        for a in arrays:
            if a is None or not xp.is_float(a):
                continue
            nan = xp.isnan(a)
            if xp.any(nan):
                valid = ~nan if valid is None else (valid & ~nan)
    if mask is not None:
        valid = mask if valid is None else (valid & mask)
    return valid


def safe_divide(xp, num, den):
    """``num / den`` where ``den != 0``, NaN elsewhere — without 0/0 warnings."""
    nz = den != 0
    safe = xp.where(nz, den, xp.ones_like(den))
    out = num / safe
    return xp.where(nz, out, xp.nan_like(out))


# ----------------------------------------------------------------------------- thresholds / ops
def normalize_thresholds(thresholds: Any) -> list[float]:
    if isinstance(thresholds, (int, float, np.integer, np.floating)):
        return [float(thresholds)]
    try:
        th = [float(t) for t in thresholds]
    except TypeError as e:
        raise TypeError(f"thresholds 必须是数或数的序列，得到 {thresholds!r}") from e
    if not th:
        raise ValueError("thresholds 不能为空")
    return th


def get_op(op: str):
    if op not in OPS:
        raise ValueError(f"op={op!r} 不受支持，可选 {sorted(OPS)}")
    return OPS[op]


def check_weight_mode(weight_mode: str) -> None:
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}")


def normalize_metrics(metrics: Any, valid: set, *, what: str = "metrics") -> tuple[str, ...]:
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    unknown = set(metrics) - set(valid)
    if unknown:
        raise ValueError(f"Unknown {what}: {sorted(unknown)}. Valid: {sorted(valid)}")
    return metrics
