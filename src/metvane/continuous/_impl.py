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

WEIGHT_MODES = ("mean", "multiply")


def check_weight_mode(weight_mode: str) -> None:
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}")


def _weighted_mean(
    data: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    weight_mode: str = "mean",
    xp: Any,
) -> Any:
    """Weighted mean — the workhorse behind all regression metrics.

    weight_mode
        ``"mean"`` (default): ``sum(w * x) / sum(w)`` over the reduced axes —
        a normalized weighted mean.  Along a *preserved* axis the weights
        cancel, e.g. keeping latitude yields the unweighted per-latitude value.
        ``"multiply"``: ``sum(w * x) / n`` (``n`` = number of valid points over
        the reduced axes) — weights act as multiplicative factors and are kept
        on preserved axes.  With mean-normalized latitude weights and latitude
        preserved this is the per-grid "area contribution" field whose plain
        spatial mean equals the latitude-weighted global score (WeatherBench /
        eval_xrv4 ``multiply_mean1``); with latitude fully reduced on a global
        grid it equals ``"mean"``.

    Slices whose denominator is not positive — summed weights (``"mean"``) or
    number of valid points (``"multiply"``), e.g. fully masked regions or
    all-NaN slices — are undefined and return NaN (like the unweighted
    ``nanmean``), not 0.
    """
    check_weight_mode(weight_mode)
    if weights is not None:
        weights = xp.broadcast_to(weights, data.shape)
        if skipna:
            mask = xp.isfinite(data)
            data = xp.where(mask, data, xp.zeros_like(data))
            weights = xp.where(mask, weights, xp.zeros_like(weights))
        if weight_mode == "mean":
            denom = xp.sum(weights, axis=axis)
        else:  # "multiply": divide by the number of (valid) points
            ones = xp.ones_like(data)
            denom = xp.sum(xp.where(mask, ones, xp.zeros_like(data)) if skipna else ones, axis=axis)
        num = xp.sum(data * weights, axis=axis)
        safe = xp.where(denom > 0, denom, xp.ones_like(denom))
        return xp.where(denom > 0, num / safe, xp.zeros_like(num) + float("nan"))
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
    weight_mode: str = "mean",
) -> Any:
    """Mean Squared Error."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        (fcst - obs) ** 2, axis=axis, weights=weights, skipna=skipna,
        weight_mode=weight_mode, xp=xp,
    )


def rmse(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    weight_mode: str = "mean",
) -> Any:
    """Root Mean Squared Error."""
    xp = get_namespace(fcst)
    return xp.sqrt(
        mse(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode)
    )


def mae(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    weight_mode: str = "mean",
) -> Any:
    """Mean Absolute Error."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        xp.abs(fcst - obs), axis=axis, weights=weights, skipna=skipna,
        weight_mode=weight_mode, xp=xp,
    )


def bias(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    weight_mode: str = "mean",
) -> Any:
    """Additive bias (fcst − obs)."""
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    return _weighted_mean(
        fcst - obs, axis=axis, weights=weights, skipna=skipna,
        weight_mode=weight_mode, xp=xp,
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
    safe_den = xp.where(den > 0, den, xp.ones_like(den))     # 先换安全分母再除，避免 0/0 警告；结果仍为 NaN
    return xp.where(den > 0, num / safe_den, nan_val)


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
    w_sum = xp.where(w_sum > 0, w_sum, xp.ones_like(w_sum))   # 权重和为 0 时下方 den=0 -> NaN
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
    safe_den = xp.where(den > 0, den, xp.ones_like(den))
    return xp.where(den > 0, num / safe_den, zero + float("nan"))


def wind_vector_rmse(
    u_fcst: Any,
    v_fcst: Any,
    u_obs: Any,
    v_obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    weight_mode: str = "mean",
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
        _weighted_mean(sq_err, axis=axis, weights=weights, skipna=True,
                       weight_mode=weight_mode, xp=xp)
    )


# ---------------------------------------------------------------------------

def _expand_dim(arr: Any, axis: int, xp: Any) -> Any:
    """Insert a size-1 dimension at *axis* (works for numpy and torch)."""
    if xp.name == "torch":
        return arr.unsqueeze(axis)
    import numpy as np
    return np.expand_dims(arr, axis)
