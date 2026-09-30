"""Continuous (regression) evaluation metrics.

Functions
---------
rmse, mse, mae, bias, acc, pearson_correlation, wind_vector_rmse, continuous_scores

Common parameters
-----------------
fcst, obs : numpy array or torch tensor of **identical shape** (no implicit broadcasting).
    xarray objects are rejected here — use :mod:`metvane.xr_api`, which aligns by
    dimension name and coordinate.
axis : int or tuple of int, optional
    Axes to reduce; ``None`` reduces all, ``()`` reduces nothing (per-grid-point values).
weights : array, optional
    Broadcastable to the data (a 1-D array is only accepted for 1-D data; use
    :func:`metvane.broadcast_weights` or ``w[:, None]`` for latitude weights).
    Converted automatically to the data's backend, device and dtype.
skipna : bool, default True
    Exclude NaN (and masked-array elements).  ``inf`` is kept as a value.
weight_mode : {'mean', 'multiply'}
    ``'mean'`` = Σw·x/Σw (weighted mean).  ``'multiply'`` = Σw·x/N, keeping weights on
    preserved axes (per-grid "area contribution" fields); only equal to the weighted mean
    for a fully reduced global latitude axis with mean-1 weights and no missing data.
mask : bool array, optional
    Evaluation mask (True = include), broadcastable to the data.
backend : {'numpy', 'torch'}, optional
    Force a backend; ``device`` is only valid together with ``backend='torch'``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.prepare import resolve_backend
from . import _impl


def _pre(arrays, backend, device):
    return resolve_backend(arrays, backend, device)


def rmse(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
         weight_mode: str = "mean", mask: Optional[Any] = None, backend: Optional[str] = None,
         device: Any = None) -> Any:
    """Root Mean Squared Error, ``sqrt(Σw(f-o)² / Σw)`` over *axis* (see module docstring).

    Across samples this is sqrt of the pooled MSE (the WeatherBench convention), not the
    mean of per-sample RMSEs.
    """
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.rmse(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask)


def mse(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
        weight_mode: str = "mean", mask: Optional[Any] = None, backend: Optional[str] = None,
        device: Any = None) -> Any:
    """Mean Squared Error (parameters: see module docstring)."""
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.mse(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask)


def mae(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
        weight_mode: str = "mean", mask: Optional[Any] = None, backend: Optional[str] = None,
        device: Any = None) -> Any:
    """Mean Absolute Error (parameters: see module docstring)."""
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.mae(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask)


def bias(fcst: Any, obs: Any, *, axis: Any = None, weights: Optional[Any] = None, skipna: bool = True,
         weight_mode: str = "mean", mask: Optional[Any] = None, backend: Optional[str] = None,
         device: Any = None) -> Any:
    """Mean error ``mean(fcst - obs)`` (additive bias, "ME").

    Not to be confused with the categorical frequency bias
    (:func:`metvane.frequency_bias` / ``ContingencyTable.bias_score``).
    """
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.bias(fcst, obs, axis=axis, weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask)


def acc(fcst: Any, obs: Any, climatology: Any, *, axis: Any = None, weights: Any = None,
        skipna: bool = True, mask: Optional[Any] = None, centered: bool = False, mean_over: Any = None,
        backend: Any = None, device: Any = None) -> Any:
    """Anomaly Correlation Coefficient (uncentred by default).

    Parameters
    ----------
    climatology : array or scalar
        Broadcastable to *fcst*; must correspond to each verification time.
    axis : axes pooled into one correlation (typically the spatial axes).
    mean_over : sample axes (init/valid time) averaged **after** computing the per-sample ACC.
        ``acc(f, o, c, axis=(2, 3), mean_over=0)`` = WeatherBench-2 / ECMWF convention;
        ``acc(f, o, c, axis=(0, 2, 3))`` = pooled ACC (dominated by large-anomaly cases).
    centered : subtract each slice's weighted mean anomaly first (centred ACC).
    Other parameters as in the module docstring.
    """
    fcst, obs, climatology, weights, mask = _pre((fcst, obs, climatology, weights, mask), backend, device)
    return _impl.acc(fcst, obs, climatology, axis=axis, weights=weights, skipna=skipna, mask=mask,
                     centered=centered, mean_over=mean_over)


def pearson_correlation(fcst: Any, obs: Any, *, axis: Any = None, weights: Any = None, skipna: bool = True,
                        mask: Optional[Any] = None, mean_over: Any = None, backend: Any = None,
                        device: Any = None) -> Any:
    """Weighted Pearson correlation over *axis* (``mean_over`` as in :func:`acc`)."""
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.pearson_correlation(fcst, obs, axis=axis, weights=weights, skipna=skipna, mask=mask,
                                     mean_over=mean_over)


def wind_vector_rmse(u_fcst: Any, v_fcst: Any, u_obs: Any, v_obs: Any, *, axis: Any = None,
                     weights: Any = None, skipna: bool = True, weight_mode: str = "mean",
                     mask: Optional[Any] = None, backend: Any = None, device: Any = None) -> Any:
    """Wind vector RMSE ``sqrt(mean_w[(u_f-u_o)² + (v_f-v_o)²])``."""
    u_fcst, v_fcst, u_obs, v_obs, weights, mask = _pre((u_fcst, v_fcst, u_obs, v_obs, weights, mask),
                                                       backend, device)
    return _impl.wind_vector_rmse(u_fcst, v_fcst, u_obs, v_obs, axis=axis, weights=weights, skipna=skipna,
                                  weight_mode=weight_mode, mask=mask)


def continuous_scores(fcst: Any, obs: Any, metrics: Any = ("rmse", "mae", "bias"), *, axis: Any = None,
                      weights: Optional[Any] = None, skipna: bool = True, weight_mode: str = "mean",
                      mask: Optional[Any] = None, backend: Optional[str] = None, device: Any = None) -> dict:
    """Compute several of ``'rmse', 'mse', 'mae', 'bias'`` in one pass (cheaper than separate calls).

    Returns ``{name: array}``; parameters as in the module docstring.
    """
    fcst, obs, weights, mask = _pre((fcst, obs, weights, mask), backend, device)
    return _impl.continuous_scores(fcst, obs, metrics, axis=axis, weights=weights, skipna=skipna,
                                   weight_mode=weight_mode, mask=mask)


__all__ = ["rmse", "mse", "mae", "bias", "acc", "pearson_correlation", "wind_vector_rmse", "continuous_scores"]
