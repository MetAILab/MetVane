"""Continuous (regression) metrics — single implementation for numpy and torch.

All metrics are built from **weighted sums** (:func:`weighted_sums`) that the chunked
accumulators reuse, so functional and accumulator results follow the same rules:

* missing values: ``NaN`` (and masked-array elements) are excluded when ``skipna=True``;
  ``inf`` is a value and propagates (a diverging model is not hidden)
* weights: converted to the data's backend/device/dtype; 1-D weights are only accepted for
  1-D data (use :func:`metvane.broadcast_weights`); zero-weight points never produce ``0*inf``
* ``weight_mode='mean'``: Σw·x / Σw (normalised weighted mean; weights cancel on kept axes)
* ``weight_mode='multiply'``: Σw·x / N (weights kept on preserved axes, e.g. per-grid
  latitude-weighted fields)
* a slice whose denominator is 0 (no weight / no valid point) is NaN
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from ..core.prepare import (WEIGHT_MODES, align_mask, align_to, check_same_shape,  # noqa: F401
                            check_weight_mode, normalize_axis, prepare, safe_divide, validity)


# ---------------------------------------------------------------------------
# Shared building blocks (also used by the accumulators and FSS/CRPS)
# ---------------------------------------------------------------------------

def reduced_count(num, data_shape, axis) -> int:
    n = 1
    axes = range(len(data_shape)) if axis is None else axis
    for a in axes:
        n *= int(data_shape[a])
    return n


def sum_broadcast(xp, x, data_shape, axis, *, keepdims: bool = False, dtype: Any = None):
    """``sum(broadcast_to(x, data_shape), axis)`` computed on the compact *x* (no full-size pass)."""
    ndim = len(data_shape)
    if tuple(x.shape) == tuple(data_shape):
        return xp.sum(x, axis, keepdims=keepdims, dtype=dtype)
    xs = (1,) * (ndim - len(x.shape)) + tuple(x.shape)
    x = x.reshape(xs)
    red = tuple(range(ndim)) if axis is None else tuple(axis)
    own = tuple(a for a in red if xs[a] > 1)
    factor = 1
    for a in red:
        if xs[a] == 1:
            factor *= int(data_shape[a])
    s = xp.sum(x, own, keepdims=True, dtype=dtype) * factor
    kd = tuple(1 if a in red else int(data_shape[a]) for a in range(ndim))
    s = xp.broadcast_to(s, kd)
    if keepdims:
        return s
    out_shape = tuple(int(data_shape[a]) for a in range(ndim) if a not in red)
    return s.reshape(out_shape)


def weighted_sums(xp, data, *, axis, weights=None, valid=None, weight_mode: str = "mean",
                  dtype: Any = None, keepdims: bool = False):
    """Return ``(Σ w·x, denominator)`` over *axis*.

    *data* is a full-shape float array, *weights* a compact array broadcastable to it (or None),
    *valid* a boolean array broadcastable to it (or None = all valid).  The denominator is
    Σ w over valid points (``'mean'``) or the number of valid points (``'multiply'`` /
    unweighted).  *dtype* sets the accumulation dtype (float64 in accumulators).
    """
    x = data
    if valid is not None:
        x = xp.where(valid, x, xp.zeros_like(x))
    if weights is not None:
        w = weights
        if valid is not None:
            w = xp.where(valid, w, xp.zeros_like(w))
        if xp.any(weights == 0) or valid is not None:
            x = xp.where(w != 0, x, xp.zeros_like(x))      # 0 * inf -> NaN otherwise
        num = xp.sum(x * w, axis, keepdims=keepdims, dtype=dtype)
        if weight_mode == "mean":
            return num, sum_broadcast(xp, w, tuple(data.shape), axis, keepdims=keepdims, dtype=dtype)
    else:
        num = xp.sum(x, axis, keepdims=keepdims, dtype=dtype)
    if valid is None:
        den = xp.zeros_like(num) + reduced_count(num, data.shape, axis)
    else:
        den = sum_broadcast(xp, valid, tuple(data.shape), axis, keepdims=keepdims,
                            dtype=dtype if dtype is not None else num.dtype)
    return num, den


def multiply_mode_check(xp, weights, valid, data_shape, axis) -> None:
    """Warn when ``weight_mode='multiply'`` is used where it no longer equals a weighted mean."""
    if weights is None:
        return
    ndim = len(data_shape)
    off = ndim - len(weights.shape)
    reduced = set(range(ndim)) if axis is None else set(axis)
    # axes along which the weights actually vary (a broadcast full-shape array is constant
    # along most of its axes)
    varying = set()
    for a in range(len(weights.shape)):
        if weights.shape[a] > 1 and xp.any(xp.abs(weights - xp.mean(weights, a, keepdims=True)) > 1e-12):
            varying.add(a + off)
    if not (varying & reduced):
        return
    if valid is not None:
        warnings.warn("weight_mode='multiply' 且在变权重的轴上归约时存在缺测/掩码：结果 Σw·x/N 不再是加权平均"
                      "（区域、掩码、含缺测时请用 weight_mode='mean'）", UserWarning, stacklevel=4)
        return
    if not varying <= reduced:
        return                      # weights kept on some axis: per-grid contribution field (intended use)
    m = xp.mean(weights, tuple(sorted(a - off for a in varying)))
    if xp.any(xp.abs(m - 1) > 1e-6):
        warnings.warn("weight_mode='multiply'：被归约轴上的权重均值不为 1（如区域子集使用全局归一化权重），"
                      "结果不等于加权平均；如需加权平均请用 weight_mode='mean'", UserWarning, stacklevel=4)


def squeeze_axes(xp, x, axes, ndim):
    axes = range(ndim) if axes is None else axes
    for a in sorted(axes, reverse=True):
        x = x.squeeze(a) if xp.name == "torch" else x.squeeze(axis=a)
    return x


def _setup(fcst, obs, *, axis, weights, mask, extra=()):
    xp, f, o, *ex = prepare(fcst, obs, *extra)
    check_same_shape(f, ("obs", o))
    ax = normalize_axis(axis, f.ndim)
    w = align_to(weights, f, xp, name="weights")
    m = align_mask(mask, f, xp)
    return xp, f, o, ex, ax, w, m


def _mean_metric(kind, fcst, obs, *, axis, weights, skipna, weight_mode, mask):
    check_weight_mode(weight_mode)
    xp, f, o, _, ax, w, m = _setup(fcst, obs, axis=axis, weights=weights, mask=mask)
    err = f - o
    valid = validity(xp, err, w, mask=m, skipna=skipna)
    if kind == "sq":
        if xp.name == "numpy":
            x = np.multiply(err, err, out=err)       # err is a fresh temporary: square in place
        else:
            x = err * err
    else:
        x = xp.abs(err) if kind == "abs" else err
    if weight_mode == "multiply":
        multiply_mode_check(xp, w, valid, tuple(f.shape), ax)
    num, den = weighted_sums(xp, x, axis=ax, weights=w, valid=valid, weight_mode=weight_mode)
    return safe_divide(xp, num, den)


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def mse(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
        weight_mode: str = "mean", mask: Optional[Any] = None) -> Any:
    """Mean Squared Error (see module docstring for NaN / weight rules)."""
    return _mean_metric("sq", fcst, obs, axis=axis, weights=weights, skipna=skipna,
                        weight_mode=weight_mode, mask=mask)


def rmse(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
         weight_mode: str = "mean", mask: Optional[Any] = None) -> Any:
    """Root Mean Squared Error = sqrt(mse) over *axis*."""
    out = mse(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask)
    from ..core.array_ns import get_namespace
    return get_namespace(out).sqrt(out)


def mae(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
        weight_mode: str = "mean", mask: Optional[Any] = None) -> Any:
    """Mean Absolute Error."""
    return _mean_metric("abs", fcst, obs, axis=axis, weights=weights, skipna=skipna,
                        weight_mode=weight_mode, mask=mask)


def bias(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
         weight_mode: str = "mean", mask: Optional[Any] = None) -> Any:
    """Mean error (additive bias, fcst − obs).  Not the categorical frequency bias."""
    return _mean_metric("err", fcst, obs, axis=axis, weights=weights, skipna=skipna,
                        weight_mode=weight_mode, mask=mask)


def continuous_scores(fcst: Any, obs: Any, metrics: Any = ("rmse", "mae", "bias"), *, axis: Any = None,
                      weights: Optional[Any] = None, skipna: bool = True, weight_mode: str = "mean",
                      mask: Optional[Any] = None) -> dict:
    """Several of rmse / mse / mae / bias in one pass (shared error field, validity and weights)."""
    from ..core.prepare import normalize_metrics
    ms = normalize_metrics(metrics, {"rmse", "mse", "mae", "bias"})
    check_weight_mode(weight_mode)
    xp, f, o, _, ax, w, m = _setup(fcst, obs, axis=axis, weights=weights, mask=mask)
    err = f - o
    valid = validity(xp, err, w, mask=m, skipna=skipna)
    if weight_mode == "multiply":
        multiply_mode_check(xp, w, valid, tuple(f.shape), ax)
    out = {}

    def mean_of(x):
        num, den = weighted_sums(xp, x, axis=ax, weights=w, valid=valid, weight_mode=weight_mode)
        return safe_divide(xp, num, den)

    if "bias" in ms:
        out["bias"] = mean_of(err)
    if "mae" in ms:
        out["mae"] = mean_of(xp.abs(err))
    if "mse" in ms or "rmse" in ms:
        sq = np.multiply(err, err, out=err) if xp.name == "numpy" else err * err
        mse_ = mean_of(sq)
        if "mse" in ms:
            out["mse"] = mse_
        if "rmse" in ms:
            out["rmse"] = xp.sqrt(mse_)
    return {k: out[k] for k in ms}


def correlation_sums(xp, fa, oa, *, axis, weights, valid, centered: bool, dtype=None, keepdims=False):
    """Sufficient statistics for (anomaly) correlation over *axis*.

    Returns dict with ``fo, ff, oo`` (Σw·f'·o', Σw·f'², Σw·o'²) and, if *centered*, ``sf, so, sw``.
    """
    w = weights
    if valid is not None:
        fa = xp.where(valid, fa, xp.zeros_like(fa))
        oa = xp.where(valid, oa, xp.zeros_like(oa))
        if w is not None:
            w = xp.where(valid, w, xp.zeros_like(w))
    if w is not None and (valid is not None or xp.any(weights == 0)):
        fa = xp.where(w != 0, fa, xp.zeros_like(fa))
        oa = xp.where(w != 0, oa, xp.zeros_like(oa))

    def S(x):
        return xp.sum(x if w is None else x * w, axis, keepdims=keepdims, dtype=dtype)

    out = {"fo": S(fa * oa), "ff": S(fa * fa), "oo": S(oa * oa)}
    if centered:
        out["sf"], out["so"] = S(fa), S(oa)
        if w is None:
            if valid is None:
                out["sw"] = xp.zeros_like(out["sf"]) + reduced_count(None, fa.shape, axis)
            else:
                out["sw"] = sum_broadcast(xp, valid, tuple(fa.shape), axis, keepdims=keepdims,
                                          dtype=dtype if dtype is not None else fa.dtype)
        else:
            out["sw"] = sum_broadcast(xp, w, tuple(fa.shape), axis, keepdims=keepdims, dtype=dtype)
    return out


def correlation_from_sums(xp, s, centered: bool):
    fo, ff, oo = s["fo"], s["ff"], s["oo"]
    if centered:
        sw = s["sw"]
        mf = safe_divide(xp, s["sf"], sw)
        mo = safe_divide(xp, s["so"], sw)
        fo = fo - mf * s["so"]
        ff = ff - mf * s["sf"]
        oo = oo - mo * s["so"]
    den = xp.sqrt(xp.where(ff * oo > 0, ff * oo, xp.zeros_like(ff)))
    return safe_divide(xp, fo, den)


def _correlation(fcst, obs, climatology, *, axis, weights, skipna, mask, centered, mean_over):
    extra = () if climatology is None else (climatology,)
    xp, f, o, ex, ax, w, m = _setup(fcst, obs, axis=axis, weights=weights, mask=mask, extra=extra)
    if climatology is not None:
        c = align_to(ex[0], f, xp, name="climatology")
        fa, oa = f - c, o - c
    else:
        fa, oa = f, o
    valid = validity(xp, fa, oa, w, mask=m, skipna=skipna)
    if centered:
        # two-pass centring on each slice (numerically safer than raw sums in float32)
        s = correlation_sums(xp, fa, oa, axis=ax, weights=w, valid=valid, centered=True, keepdims=True)
        fa = fa - safe_divide(xp, s["sf"], s["sw"])
        oa = oa - safe_divide(xp, s["so"], s["sw"])
    s = correlation_sums(xp, fa, oa, axis=ax, weights=w, valid=valid, centered=False, keepdims=True)
    score = correlation_from_sums(xp, s, centered=False)
    ndim = f.ndim
    reduced = set(range(ndim)) if ax is None else set(ax)
    if mean_over is not None:
        mo = normalize_axis(mean_over, ndim, name="mean_over")
        if set(mo) & reduced:
            raise ValueError(f"mean_over={mo} 与 axis={ax} 重叠：mean_over 应为样本（起报/时间）轴，axis 为空间轴")
        score = xp.nanmean(score, mo, keepdims=True)
        reduced |= set(mo)
    return squeeze_axes(xp, score, sorted(reduced), ndim)


def acc(fcst: Any, obs: Any, climatology: Any, *, axis: Any = None, weights: Optional[Any] = None,
        skipna: bool = True, mask: Optional[Any] = None, centered: bool = False,
        mean_over: Any = None) -> Any:
    """Anomaly Correlation Coefficient.

    .. math:: ACC = \\frac{\\sum w f' o'}{\\sqrt{\\sum w f'^2 \\sum w o'^2}},\\quad f'=f-c,\\ o'=o-c

    * ``axis``: axes pooled into one correlation (e.g. the spatial axes).
    * ``mean_over``: sample axes (e.g. init time) over which the per-sample ACC is averaged
      afterwards — ``acc(f, o, c, axis=(2, 3), mean_over=0)`` is the WeatherBench-2 / ECMWF
      convention; putting the sample axis into ``axis`` instead gives a *pooled* ACC dominated by
      large-anomaly cases.
    * ``centered=True`` subtracts the (weighted) mean anomaly of each slice first.
    * *climatology* must broadcast to *fcst* and correspond to each **verification time**.
    """
    return _correlation(fcst, obs, climatology, axis=axis, weights=weights, skipna=skipna, mask=mask,
                        centered=centered, mean_over=mean_over)


def pearson_correlation(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None,
                        skipna: bool = True, mask: Optional[Any] = None, mean_over: Any = None) -> Any:
    """Weighted Pearson correlation coefficient (centred correlation of the raw fields)."""
    return _correlation(fcst, obs, None, axis=axis, weights=weights, skipna=skipna, mask=mask,
                        centered=True, mean_over=mean_over)


def wind_vector_rmse(u_fcst: Any, v_fcst: Any, u_obs: Any, v_obs: Any, *, axis: Any = None,
                     weights: Optional[Any] = None, skipna: bool = True, weight_mode: str = "mean",
                     mask: Optional[Any] = None) -> Any:
    """Wind vector RMSE: sqrt(mean_w[(u_f-u_o)² + (v_f-v_o)²])."""
    check_weight_mode(weight_mode)
    xp, uf, vf, uo, vo = prepare(u_fcst, v_fcst, u_obs, v_obs)
    check_same_shape(uf, ("v_fcst", vf), ("u_obs", uo), ("v_obs", vo), ref_name="u_fcst")
    ax = normalize_axis(axis, uf.ndim)
    w = align_to(weights, uf, xp, name="weights")
    m = align_mask(mask, uf, xp)
    sq = (uf - uo) ** 2 + (vf - vo) ** 2
    valid = validity(xp, sq, w, mask=m, skipna=skipna)
    if weight_mode == "multiply":
        multiply_mode_check(xp, w, valid, tuple(uf.shape), ax)
    num, den = weighted_sums(xp, sq, axis=ax, weights=w, valid=valid, weight_mode=weight_mode)
    return xp.sqrt(safe_divide(xp, num, den))
