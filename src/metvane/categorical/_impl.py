"""Categorical metrics — single implementation for numpy and torch.

Core routine: exact **int64** TP / FP / FN / TN counts from continuous forecasts and
observations for one or more thresholds.

* missing values: ``NaN`` in either field (and masked-array elements, and points where
  ``mask`` is False) are excluded from **all four** counts when ``skipna=True``.  With
  ``skipna=False`` a slice containing NaN yields NaN counts (float64).
* comparisons are done in the data's own dtype (the threshold is cast to it), so every
  entry point makes the same event decision at the threshold boundary.
* ``fcst`` and ``obs`` must have identical shapes (no implicit broadcasting).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from ..core.prepare import (align_mask, check_same_shape, get_op, normalize_axis,
                            normalize_thresholds, prepare, validity)


def _full(xp, x, shape):
    return x if tuple(x.shape) == tuple(shape) else xp.broadcast_to(x, tuple(shape))


def compute_contingency(
    fcst: Any,
    obs: Any,
    thresholds: Any,
    *,
    op: str = ">=",
    axis: Any = None,
    skipna: bool = True,
    mask: Optional[Any] = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute TP, FP, FN, TN for each threshold.

    Returns four arrays of shape ``(n_thresholds, *kept_dims)`` — int64, or float64 with NaN
    when ``skipna=False`` and a slice contains NaN.
    """
    cmp = get_op(op)
    ths = normalize_thresholds(thresholds)
    xp, f, o = prepare(fcst, obs, as_float=False)
    check_same_shape(f, ("obs", o))
    ax = normalize_axis(axis, f.ndim)
    m = align_mask(mask, f, xp)
    valid = validity(xp, f, o, mask=m, skipna=True)
    shape = tuple(f.shape)
    i64 = xp.int64

    # valid is None  <=>  no NaN and no mask, so (f == f) is all True
    n_valid = xp.sum(_full(xp, valid, shape) if valid is not None else (f == f), ax, dtype=i64)

    tps, fps, fns, tns = [], [], [], []
    for t in ths:
        pb, ob = cmp(f, t), cmp(o, t)
        if valid is not None:
            pb, ob = pb & valid, ob & valid
        tp = xp.sum(pb & ob, ax, dtype=i64)
        npred = xp.sum(pb, ax, dtype=i64)
        nobs = xp.sum(ob, ax, dtype=i64)
        tps.append(tp)
        fps.append(npred - tp)
        fns.append(nobs - tp)
        tns.append(n_valid - npred - nobs + tp)
    out = [xp.stack(c) for c in (tps, fps, fns, tns)]
    _warn_degenerate(xp, op, ths, out)

    if not skipna:
        nan_any = None
        for a in (f, o):
            if xp.is_float(a):
                nn = xp.isnan(a)
                nan_any = nn if nan_any is None else (nan_any | nn)
        if nan_any is not None and xp.any(nan_any):
            if m is not None:
                nan_any = nan_any & m
            has_nan = xp.sum(nan_any, ax, dtype=i64) > 0
            out = [xp.where(has_nan, xp.nan_like(xp.astype(c, xp.float64)), xp.astype(c, xp.float64))
                   for c in out]
    return tuple(out)


def _warn_degenerate(xp, op, ths, counts) -> None:
    tp, fp, fn, tn = counts
    if op == ">=" and any(t == 0 for t in ths):
        warnings.warn("op='>=' 且阈值为 0：对非负变量（降水等）所有格点都是事件，CSI/POD 恒为 1。"
                      "降水请用 op='>' 或阈值 0.1", UserWarning, stacklevel=4)
        return
    for i, t in enumerate(ths):
        n_i = int(xp.to_scalar(xp.sum(tp[i] + fp[i] + fn[i] + tn[i])))
        if n_i > 0 and int(xp.to_scalar(xp.sum(tp[i]))) == n_i:
            warnings.warn(f"列联表退化：阈值 {t} (op='{op}') 下全部有效格点在预报和观测中都是事件"
                          "（HSS/ETS 无定义，返回 NaN）；请检查阈值与 op", UserWarning, stacklevel=4)
            return
