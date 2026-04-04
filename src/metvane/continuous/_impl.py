"""Continuous (regression) metrics — single implementation for numpy and torch.

Every function accepts numpy arrays *or* torch tensors and transparently
dispatches through the array-namespace adapter.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.array_ns import get_namespace
from ..core.backend import ensure_same_backend


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_mean(
    data: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    xp: Any,
) -> Any:
    """Weighted mean — the workhorse behind all regression metrics."""
    if weights is not None:
        weights = xp.broadcast_to(weights, data.shape)
        if skipna:
            mask = xp.isfinite(data)
            data = xp.where(mask, data, xp.zeros_like(data))
            weights = xp.where(mask, weights, xp.zeros_like(weights))
        denom = xp.sum(weights, axis=axis)
        denom = xp.where(denom > 0, denom, xp.ones_like(denom))
        return xp.sum(data * weights, axis=axis) / denom
    return xp.nanmean(data, axis=axis) if skipna else xp.mean(data, axis=axis)


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def mse(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
) -> Any:
    """Mean Squared Error."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        (fcst - obs) ** 2, axis=axis, weights=weights, skipna=skipna, xp=xp,
    )


def rmse(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
) -> Any:
    """Root Mean Squared Error."""
    xp = get_namespace(fcst)
    return xp.sqrt(
        mse(fcst, obs, axis=axis, weights=weights, skipna=skipna)
    )


def mae(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
) -> Any:
    """Mean Absolute Error."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        xp.abs(fcst - obs), axis=axis, weights=weights, skipna=skipna, xp=xp,
    )


def bias(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
) -> Any:
    """Additive bias (fcst − obs)."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        fcst - obs, axis=axis, weights=weights, skipna=skipna, xp=xp,
    )


def acc(
    fcst: Any,
    obs: Any,
    climatology: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
) -> Any:
    """Anomaly Correlation Coefficient.

    .. math::
        ACC = \\frac{\\sum w \\, f' \\, o'}{\\sqrt{\\sum w \\, f'^2 \\;\\sum w \\, o'^2}}

    where :math:`f' = \\text{fcst} - \\text{clim}` and
    :math:`o' = \\text{obs} - \\text{clim}`.
    """
    fcst, obs, clim = ensure_same_backend(fcst, obs, climatology)
    xp = get_namespace(fcst)
    fcst, obs, clim = xp.as_float(fcst), xp.as_float(obs), xp.as_float(clim)

    f_anom = fcst - clim
    o_anom = obs - clim

    if weights is not None:
        weights = xp.broadcast_to(weights, fcst.shape)
    else:
        weights = xp.ones_like(fcst)

    num = xp.sum(weights * f_anom * o_anom, axis=axis)
    den = xp.sqrt(
        xp.sum(weights * f_anom ** 2, axis=axis)
        * xp.sum(weights * o_anom ** 2, axis=axis)
    )
    zero = xp.zeros_like(num)
    nan_val = zero + float("nan")
    return xp.where(den > 0, num / den, nan_val)


def pearson_correlation(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
) -> Any:
    """Weighted Pearson correlation coefficient."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)

    if weights is not None:
        weights = xp.broadcast_to(weights, fcst.shape)
    else:
        weights = xp.ones_like(fcst)

    w_sum = xp.sum(weights, axis=axis)
    f_mean = xp.sum(weights * fcst, axis=axis) / w_sum
    o_mean = xp.sum(weights * obs, axis=axis) / w_sum

    # Expand means for broadcasting
    if axis is not None:
        if isinstance(axis, int):
            ax_tuple = (axis,)
        else:
            ax_tuple = tuple(axis)
        for a in sorted(ax_tuple):
            f_mean = _expand_dim(f_mean, a, xp)
            o_mean = _expand_dim(o_mean, a, xp)

    f_dev = fcst - f_mean
    o_dev = obs - o_mean

    num = xp.sum(weights * f_dev * o_dev, axis=axis)
    den = xp.sqrt(
        xp.sum(weights * f_dev ** 2, axis=axis)
        * xp.sum(weights * o_dev ** 2, axis=axis)
    )
    zero = xp.zeros_like(num)
    return xp.where(den > 0, num / den, zero + float("nan"))


def wind_vector_rmse(
    u_fcst: Any,
    v_fcst: Any,
    u_obs: Any,
    v_obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
) -> Any:
    """Wind vector RMSE.

    .. math::
        \\sqrt{\\overline{w \\left[(u_f - u_o)^2 + (v_f - v_o)^2\\right]}}
    """
    u_fcst, u_obs = ensure_same_backend(u_fcst, u_obs)
    v_fcst, v_obs = ensure_same_backend(v_fcst, v_obs)
    xp = get_namespace(u_fcst)
    sq_err = (xp.as_float(u_fcst) - xp.as_float(u_obs)) ** 2 + \
             (xp.as_float(v_fcst) - xp.as_float(v_obs)) ** 2
    return xp.sqrt(
        _weighted_mean(sq_err, axis=axis, weights=weights, skipna=True, xp=xp)
    )


# ---------------------------------------------------------------------------

def _expand_dim(arr: Any, axis: int, xp: Any) -> Any:
    """Insert a size-1 dimension at *axis* (works for numpy and torch)."""
    if xp.name == "torch":
        return arr.unsqueeze(axis)
    import numpy as np
    return np.expand_dims(arr, axis)
