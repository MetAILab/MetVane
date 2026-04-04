"""Continuous (regression) evaluation metrics.

Functions
---------
rmse, mse, mae, bias, acc, pearson_correlation, wind_vector_rmse

All functions share a unified signature::

    metric(fcst, obs, *, axis=None, weights=None, backend=None, device=None)
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.backend import Backend, convert, detect_backend
from . import _impl


def _prepare(
    *arrays: Any,
    backend: Optional[str],
    device: Any,
) -> tuple[Any, ...]:
    if backend is not None:
        target = Backend(backend)
        return tuple(convert(a, target, device=device) for a in arrays)
    return arrays


def rmse(
    fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    backend: Optional[str] = None,
    device: Any = None,
) -> Any:
    """Root Mean Squared Error.

    Parameters
    ----------
    fcst, obs : array-like
        Forecast and observation arrays (numpy, torch, or xarray).
    axis : int or tuple of ints, optional
        Dimensions to reduce.  ``None`` reduces all.
    weights : array-like, optional
        Weighting array (e.g. latitude weights).
    skipna : bool
        Skip NaN values.
    backend : {'numpy', 'torch'}, optional
        Force a specific backend.
    device : str, optional
        Torch device (only when *backend='torch'*).
    """
    fcst, obs = _prepare(fcst, obs, backend=backend, device=device)
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.rmse(fcst, obs, axis=axis, weights=weights, skipna=skipna)


def mse(fcst: Any, obs: Any, *, axis: Any = None, weights: Any = None,
        skipna: bool = True, backend: Any = None, device: Any = None) -> Any:
    """Mean Squared Error."""
    fcst, obs = _prepare(fcst, obs, backend=backend, device=device)
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.mse(fcst, obs, axis=axis, weights=weights, skipna=skipna)


def mae(fcst: Any, obs: Any, *, axis: Any = None, weights: Any = None,
        skipna: bool = True, backend: Any = None, device: Any = None) -> Any:
    """Mean Absolute Error."""
    fcst, obs = _prepare(fcst, obs, backend=backend, device=device)
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.mae(fcst, obs, axis=axis, weights=weights, skipna=skipna)


def bias(fcst: Any, obs: Any, *, axis: Any = None, weights: Any = None,
         skipna: bool = True, backend: Any = None, device: Any = None) -> Any:
    """Additive bias (forecast minus observation)."""
    fcst, obs = _prepare(fcst, obs, backend=backend, device=device)
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.bias(fcst, obs, axis=axis, weights=weights, skipna=skipna)


def acc(fcst: Any, obs: Any, climatology: Any, *, axis: Any = None,
        weights: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Anomaly Correlation Coefficient."""
    fcst, obs, climatology = _prepare(
        fcst, obs, climatology, backend=backend, device=device,
    )
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.acc(fcst, obs, climatology, axis=axis, weights=weights)


def pearson_correlation(
    fcst: Any, obs: Any, *, axis: Any = None, weights: Any = None,
    backend: Any = None, device: Any = None,
) -> Any:
    """Weighted Pearson correlation coefficient."""
    fcst, obs = _prepare(fcst, obs, backend=backend, device=device)
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.pearson_correlation(fcst, obs, axis=axis, weights=weights)


def wind_vector_rmse(
    u_fcst: Any, v_fcst: Any, u_obs: Any, v_obs: Any, *,
    axis: Any = None, weights: Any = None,
    backend: Any = None, device: Any = None,
) -> Any:
    """Wind vector RMSE."""
    u_fcst, v_fcst, u_obs, v_obs = _prepare(
        u_fcst, v_fcst, u_obs, v_obs, backend=backend, device=device,
    )
    if weights is not None:
        (weights,) = _prepare(weights, backend=backend, device=device)
    return _impl.wind_vector_rmse(
        u_fcst, v_fcst, u_obs, v_obs, axis=axis, weights=weights,
    )
