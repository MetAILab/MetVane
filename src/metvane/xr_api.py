"""xarray convenience layer.

Wraps the numpy-core metric functions with named-dimension handling
(``reduce_dims`` / ``preserve_dims``), returning ``xr.DataArray`` or
``xr.Dataset`` results.

Usage::

    import metvane.xr_api as mxr

    result = mxr.rmse(fcst_da, obs_da, preserve_dims='lead_time', weights=w_da)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

FlexibleDims = Optional[Union[str, Sequence[str]]]


def _to_tuple(x: Optional[Union[str, Sequence[str]]]) -> Optional[tuple[str, ...]]:
    if x is None:
        return None
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _gather_reduce_dims(
    all_dims: tuple[str, ...],
    reduce_dims: FlexibleDims = None,
    preserve_dims: FlexibleDims = None,
) -> tuple[str, ...]:
    """Resolve which dimensions to reduce.

    Exactly one of *reduce_dims* / *preserve_dims* may be set, or
    neither (reduce all).
    """
    rd = _to_tuple(reduce_dims)
    pd = _to_tuple(preserve_dims)
    if rd is not None and pd is not None:
        raise ValueError("Specify only one of reduce_dims / preserve_dims")

    if rd is not None:
        return tuple(d for d in all_dims if d in rd)
    if pd is not None:
        if pd == ("all",):
            return ()
        return tuple(d for d in all_dims if d not in pd)
    return all_dims


def _dims_to_axes(da: Any, dims: tuple[str, ...]) -> tuple[int, ...]:
    """Axes to reduce.  An empty tuple means *no* reduction (e.g.
    ``preserve_dims='all'`` or every dimension preserved) — it must not be
    turned into ``None``, which would reduce everything."""
    return tuple(list(da.dims).index(d) for d in dims if d in da.dims)


def _get_weights_values(weights: Any, reference: Any = None) -> Any:
    """Extract numpy values from weights, broadcasting to reference shape if needed."""
    if weights is None:
        return None
    if hasattr(weights, "values") and hasattr(weights, "dims") and reference is not None:
        import xarray as xr
        if isinstance(weights, xr.DataArray) and isinstance(reference, xr.DataArray):
            aligned = weights.broadcast_like(reference)
            return aligned.values
    if hasattr(weights, "values"):
        return weights.values
    return weights


def _wrap_result(
    result_np: Any,
    reference_da: Any,
    reduced_dims: tuple[str, ...],
) -> Any:
    """Wrap a numpy result back into an xr.DataArray.

    The kept dimensions are those of *reference_da* that were not reduced,
    in their original order — whether they were chosen via ``preserve_dims``
    or ``reduce_dims``.  Coordinates living only on kept dimensions are kept.
    """
    import numpy as np
    import xarray as xr

    result_np = np.asarray(result_np)
    kept = [d for d in reference_da.dims if d not in reduced_dims]
    if result_np.ndim == 0 or not kept:
        return xr.DataArray(result_np)
    coords = {k: v for k, v in reference_da.coords.items() if set(v.dims) <= set(kept)}
    return xr.DataArray(result_np, dims=kept, coords=coords)


def _apply_to_dataset(fn, fcst_ds, obs_ds, **kwargs):
    """Apply a metric function to each common variable in two Datasets."""
    import xarray as xr

    common_vars = sorted(set(fcst_ds.data_vars) & set(obs_ds.data_vars))
    results = {}
    for v in common_vars:
        results[v] = fn(fcst_ds[v], obs_ds[v], **kwargs)
    return xr.Dataset(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rmse(
    fcst: Any,
    obs: Any,
    *,
    reduce_dims: FlexibleDims = None,
    preserve_dims: FlexibleDims = None,
    weights: Any = None,
    skipna: bool = True,
    weight_mode: str = "mean",
) -> Any:
    """RMSE with named dimension handling (``weight_mode`` as in :func:`metvane.rmse`)."""
    import xarray as xr
    from .continuous import _impl

    if isinstance(fcst, xr.Dataset):
        return _apply_to_dataset(
            rmse, fcst, obs,
            reduce_dims=reduce_dims, preserve_dims=preserve_dims,
            weights=weights, skipna=skipna, weight_mode=weight_mode,
        )

    dims = _gather_reduce_dims(fcst.dims, reduce_dims, preserve_dims)
    axes = _dims_to_axes(fcst, dims)
    result = _impl.rmse(
        fcst.values, obs.values, axis=axes,
        weights=_get_weights_values(weights, fcst), skipna=skipna,
        weight_mode=weight_mode,
    )
    return _wrap_result(result, fcst, dims)


def mse(fcst: Any, obs: Any, *, reduce_dims: FlexibleDims = None,
        preserve_dims: FlexibleDims = None, weights: Any = None,
        skipna: bool = True, weight_mode: str = "mean") -> Any:
    """MSE with named dimension handling."""
    import xarray as xr
    from .continuous import _impl

    if isinstance(fcst, xr.Dataset):
        return _apply_to_dataset(
            mse, fcst, obs, reduce_dims=reduce_dims,
            preserve_dims=preserve_dims, weights=weights, skipna=skipna,
            weight_mode=weight_mode,
        )
    dims = _gather_reduce_dims(fcst.dims, reduce_dims, preserve_dims)
    axes = _dims_to_axes(fcst, dims)
    result = _impl.mse(fcst.values, obs.values, axis=axes,
                       weights=_get_weights_values(weights, fcst), skipna=skipna,
                       weight_mode=weight_mode)
    return _wrap_result(result, fcst, dims)


def mae(fcst: Any, obs: Any, *, reduce_dims: FlexibleDims = None,
        preserve_dims: FlexibleDims = None, weights: Any = None,
        skipna: bool = True, weight_mode: str = "mean") -> Any:
    """MAE with named dimension handling."""
    import xarray as xr
    from .continuous import _impl

    if isinstance(fcst, xr.Dataset):
        return _apply_to_dataset(
            mae, fcst, obs, reduce_dims=reduce_dims,
            preserve_dims=preserve_dims, weights=weights, skipna=skipna,
            weight_mode=weight_mode,
        )
    dims = _gather_reduce_dims(fcst.dims, reduce_dims, preserve_dims)
    axes = _dims_to_axes(fcst, dims)
    result = _impl.mae(fcst.values, obs.values, axis=axes,
                       weights=_get_weights_values(weights, fcst), skipna=skipna,
                       weight_mode=weight_mode)
    return _wrap_result(result, fcst, dims)


def acc(fcst: Any, obs: Any, climatology: Any, *,
        reduce_dims: FlexibleDims = None, preserve_dims: FlexibleDims = None,
        weights: Any = None) -> Any:
    """ACC with named dimension handling."""
    import xarray as xr
    from .continuous import _impl

    dims = _gather_reduce_dims(fcst.dims, reduce_dims, preserve_dims)
    axes = _dims_to_axes(fcst, dims)
    clim_vals = climatology.values if hasattr(climatology, "values") else climatology
    result = _impl.acc(fcst.values, obs.values, clim_vals, axis=axes,
                       weights=_get_weights_values(weights, fcst))
    return _wrap_result(result, fcst, dims)


def bias(fcst: Any, obs: Any, *, reduce_dims: FlexibleDims = None,
         preserve_dims: FlexibleDims = None, weights: Any = None,
         skipna: bool = True, weight_mode: str = "mean") -> Any:
    """Bias with named dimension handling."""
    import xarray as xr
    from .continuous import _impl

    if isinstance(fcst, xr.Dataset):
        return _apply_to_dataset(
            bias, fcst, obs, reduce_dims=reduce_dims,
            preserve_dims=preserve_dims, weights=weights, skipna=skipna,
            weight_mode=weight_mode,
        )
    dims = _gather_reduce_dims(fcst.dims, reduce_dims, preserve_dims)
    axes = _dims_to_axes(fcst, dims)
    result = _impl.bias(fcst.values, obs.values, axis=axes,
                        weights=_get_weights_values(weights, fcst), skipna=skipna,
                        weight_mode=weight_mode)
    return _wrap_result(result, fcst, dims)
