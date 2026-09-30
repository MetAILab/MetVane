"""xarray layer — named dimensions, coordinate alignment, lazy (dask) evaluation.

Every entry point:

1. **aligns by name and coordinate**, never by position: *obs* must have the same set of
   dimensions as *fcst* and is transposed to ``fcst.dims``; coordinates that are the same set
   in a different order (e.g. latitude 90→-90 vs -90→90) are re-ordered to *fcst*'s order
   (``join='reorder'``, default), identical coordinates are required with ``join='exact'``,
   and ``join='inner'`` evaluates the common coordinates only.  Numeric coordinates match
   within ``coord_tol``.  Climatology / weights / masks may have a subset of *fcst*'s dims
   (and a larger coordinate range, e.g. global weights for a regional field) and are
   aligned the same way on the shared dims.
2. validates ``reduce_dims`` / ``preserve_dims`` (unknown names raise);
3. computes with **xarray-native** operations (weighted sums, then ratios), so dask-backed
   inputs stay lazy — call ``.compute()`` on the result;
4. uses the same rules as the numpy/torch core: NaN = missing (``skipna``), ``inf`` is a
   value, zero denominators → NaN.

Usage::

    import metvane.xr_api as mxr
    w = metvane.latitude_weights(fcst.latitude)            # DataArray on 'latitude'
    mxr.rmse(fcst, obs, preserve_dims="lead_time", weights=w)
    mxr.acc(fcst, obs, clim, reduce_dims=["latitude", "longitude"], mean_over="time", weights=w)
    mxr.categorical_scores(fcst, obs, [0.1, 1, 5], preserve_dims="lead_time")
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np

from .continuous._impl import check_weight_mode
from .core.prepare import get_op, normalize_thresholds

FlexibleDims = Optional[Union[str, Sequence[str]]]
JOINS = ("reorder", "exact", "inner")

__all__ = [
    "FlexibleDims", "rmse", "mse", "mae", "bias", "acc", "pearson_correlation", "wind_vector_rmse",
    "contingency_table", "scores_from_counts", "categorical_scores", "align_inputs",
]

_DIM_HINT = "（常见命名差异：lat/latitude、lon/longitude、step/lead_time/prediction_timedelta、time/init_time）"


# ============================================================================ dims
def _to_tuple(x: FlexibleDims) -> Optional[tuple[str, ...]]:
    if x is None:
        return None
    if isinstance(x, str):
        return (x,)
    return tuple(x)


def _resolve_reduce_dims(all_dims: Sequence[str], reduce_dims: FlexibleDims = None,
                         preserve_dims: FlexibleDims = None, *, strict: bool = True,
                         what: str = "fcst") -> tuple[str, ...]:
    """Dims to reduce.  Exactly one of *reduce_dims* / *preserve_dims* (or neither = all).

    ``preserve_dims='all'`` keeps every dimension (unless a dimension is literally named
    ``'all'``, which is ambiguous and raises).  Unknown names raise (``strict``) or are ignored.
    """
    all_dims = tuple(all_dims)
    rd, pd = _to_tuple(reduce_dims), _to_tuple(preserve_dims)
    if rd is not None and pd is not None:
        raise ValueError("Specify only one of reduce_dims / preserve_dims（两者只能给一个）")
    if pd == ("all",):
        if "all" in all_dims:
            raise ValueError("preserve_dims='all' 有歧义：数据中存在名为 'all' 的维；请显式列出要保留的维")
        return ()
    names = rd if rd is not None else pd
    if names is not None and strict:
        unknown = [d for d in names if d not in all_dims]
        if unknown:
            raise ValueError(f"{'reduce_dims' if rd is not None else 'preserve_dims'} 中的维 {unknown} "
                             f"不在 {what} 的维 {all_dims} 中{_DIM_HINT}")
    if rd is not None:
        return tuple(d for d in all_dims if d in rd)
    if pd is not None:
        return tuple(d for d in all_dims if d not in pd)
    return all_dims


# ============================================================================ alignment
def _match_positions(ref_vals: np.ndarray, x_vals: np.ndarray, tol: float) -> Optional[np.ndarray]:
    """Positions in *x_vals* matching each *ref_vals* entry (exact, or within *tol* for numbers)."""
    import pandas as pd
    idx = pd.Index(x_vals)
    if not idx.is_unique:
        return None
    pos = np.asarray(idx.get_indexer(ref_vals))
    if (pos >= 0).all():
        return pos
    xv, rv = np.asarray(x_vals), np.asarray(ref_vals)
    if tol and np.issubdtype(xv.dtype, np.number) and np.issubdtype(rv.dtype, np.number) and len(xv):
        xv, rv = xv.astype(np.float64), rv.astype(np.float64)
        order = np.argsort(xv)
        xs = xv[order]
        i = np.searchsorted(xs, rv)
        cand = np.stack([np.clip(i - 1, 0, len(xs) - 1), np.clip(i, 0, len(xs) - 1)])
        dist = np.abs(xs[cand] - rv[None])
        best = cand[np.argmin(dist, axis=0), np.arange(len(rv))]
        if (np.abs(xs[best] - rv) <= tol).all() and len(np.unique(best)) == len(best):
            return order[best]
    return None


def _align_to_ref(x: Any, ref: Any, *, name: str, join: str, coord_tol: float,
                  same_dims: bool, allow_superset: bool) -> Any:
    """Align DataArray *x* to *ref* (dims subset / equal, coordinates re-ordered / sub-selected)."""
    extra = [d for d in x.dims if d not in ref.dims]
    if extra:
        raise ValueError(f"{name} 含 fcst 没有的维 {extra}：{name}.dims={tuple(x.dims)}，fcst.dims={tuple(ref.dims)}"
                         f"{_DIM_HINT}")
    if same_dims and set(x.dims) != set(ref.dims):
        raise ValueError(f"{name} 的维 {tuple(x.dims)} 与 fcst 的维 {tuple(ref.dims)} 不一致（obs 必须与 fcst 同维，"
                         f"不做隐式广播）{_DIM_HINT}")
    for d in x.dims:
        if not (d in x.indexes and d in ref.indexes):
            if x.sizes[d] != ref.sizes[d]:
                raise ValueError(f"{name} 的维 {d!r} 长度 {x.sizes[d]} 与 fcst 的 {ref.sizes[d]} 不一致（且没有坐标可对齐）")
            continue
        rv, xv = ref.indexes[d].values, x.indexes[d].values
        if len(rv) == len(xv) and np.array_equal(rv, xv):
            continue
        if join == "exact":
            raise ValueError(f"{name} 的 {d!r} 坐标与 fcst 不同（join='exact'）")
        pos = _match_positions(rv, xv, coord_tol)
        if pos is None:
            raise ValueError(f"{name} 的 {d!r} 坐标无法与 fcst 对齐（既不是同一集合的重排，也不包含 fcst 的全部坐标；"
                             f"coord_tol={coord_tol}）。fcst[{d!r}]: {rv[:3]}...{rv[-3:]}，"
                             f"{name}[{d!r}]: {xv[:3]}...{xv[-3:]}")
        if len(xv) != len(rv) and not allow_superset:
            raise ValueError(f"{name} 的 {d!r} 坐标比 fcst 多（{len(xv)} vs {len(rv)}）；只评估公共部分请用 join='inner'")
        x = x.isel({d: pos}).assign_coords({d: ref[d].values})
    return x.transpose(*[d for d in ref.dims if d in x.dims])


def align_inputs(fcst: Any, obs: Any, *others: tuple[str, Any], join: str = "reorder",
                 coord_tol: float = 1e-4) -> tuple[Any, ...]:
    """Align *obs* and auxiliary DataArrays (``(name, array)`` pairs) to *fcst*.

    Returns ``(fcst, obs, *others)``; see module docstring for the rules.  Scalars and ``None``
    in *others* are passed through; plain arrays are only accepted with exactly ``fcst.shape``.
    """
    import xarray as xr
    if join not in JOINS:
        raise ValueError(f"join={join!r} 不受支持，可选 {JOINS}")
    if not isinstance(fcst, xr.DataArray) or not isinstance(obs, xr.DataArray):
        raise TypeError("fcst 和 obs 必须都是 xr.DataArray（或同为 xr.Dataset）")
    if join == "inner":
        if set(obs.dims) != set(fcst.dims):
            raise ValueError(f"obs 的维 {tuple(obs.dims)} 与 fcst 的维 {tuple(fcst.dims)} 不一致{_DIM_HINT}")
        n0 = fcst.size
        fcst, obs = xr.align(fcst, obs.transpose(*fcst.dims), join="inner")
        if fcst.size == 0:
            raise ValueError("join='inner' 后 fcst 与 obs 没有公共坐标")
        if fcst.size != n0:
            warnings.warn(f"join='inner'：只评估公共坐标（{fcst.size}/{n0} 个点）", UserWarning, stacklevel=3)
    obs = _align_to_ref(obs, fcst, name="obs", join=join, coord_tol=coord_tol, same_dims=True,
                        allow_superset=False)
    out = []
    for name, x in others:
        if x is None or isinstance(x, (int, float, np.number)):
            out.append(x)
        elif isinstance(x, xr.DataArray):
            out.append(_align_to_ref(x, fcst, name=name, join="reorder" if join == "inner" else join,
                                     coord_tol=coord_tol, same_dims=False, allow_superset=True))
        elif isinstance(x, xr.Dataset):
            raise TypeError(f"{name} 是 Dataset，而 fcst 是 DataArray")
        else:
            arr = x if hasattr(x, "shape") else np.asarray(x)
            if tuple(arr.shape) != tuple(fcst.shape):
                raise ValueError(f"{name} 是无维名的数组 {tuple(arr.shape)}：xr_api 只接受与 fcst 形状完全相同的数组"
                                 f" {tuple(fcst.shape)}，否则请传 DataArray（如 metvane.latitude_weights(fcst.latitude)）")
            out.append(xr.DataArray(np.asarray(arr), dims=fcst.dims, coords=fcst.coords))
    return (fcst, obs, *out)


# ============================================================================ xarray-native kernels
def _xr():
    import xarray as xr
    return xr


def _safe_div(num, den):
    """num / den with NaN where den == 0 (den may be a Python number)."""
    if not hasattr(den, "where"):
        return num / den if den != 0 else num * np.nan
    return _xr().where(den != 0, num / den.where(den != 0, 1), np.nan)


def _nreduced(ref, dims) -> int:
    n = 1
    for d in dims:
        n *= int(ref.sizes[d])
    return n


def _valid(*arrays, mask=None, skipna=True):
    v = None
    if skipna:
        for a in arrays:
            if a is None or not hasattr(a, "notnull") or not np.issubdtype(a.dtype, np.floating):
                continue
            nn = a.notnull()
            v = nn if v is None else (v & nn)
    if mask is not None:
        m = mask.astype(bool)
        v = m if v is None else (v & m)
    return v


def _sum_w(w, x, dims):
    """Σ over *dims* of *w* broadcast to *x*'s dims (without materialising the broadcast)."""
    return w.sum([d for d in dims if d in w.dims], skipna=False) * _nreduced(x, [d for d in dims if d not in w.dims])


def _weighted_sums(x, dims, w=None, valid=None, weight_mode="mean"):
    """xarray-native Σw·x and denominator over *dims* (same rules as the numpy core)."""
    if valid is not None:
        x = x.where(valid, 0)
    if w is not None:
        we = w if valid is None else w.where(valid, 0)
        x = x.where(we != 0, 0)                            # 0 * inf -> NaN otherwise
        num = (x * we).sum(dims, skipna=False)
        if weight_mode == "mean":
            return num, _sum_w(we, x, dims)
    else:
        num = x.sum(dims, skipna=False)
    if valid is None:
        return num, _nreduced(x, dims)
    return num, _sum_w(valid, x, dims)


def _finish(out, name, ref, units_pow=1):
    out = out.rename(name)
    attrs = {"metric": name}
    units = ref.attrs.get("units")
    if units_pow == 0:
        attrs["units"] = "1"
    elif units:
        attrs["units"] = units if units_pow == 1 else f"({units})^2"
    out.attrs = attrs
    return out


# ============================================================================ Dataset dispatch
def _dataset_apply(fn, fcst_ds, obs_ds, *, aux: dict, reduce_dims, preserve_dims, join_vars: str, **kw):
    xr = _xr()
    if not isinstance(obs_ds, xr.Dataset):
        raise TypeError("fcst 是 Dataset 时 obs 也必须是 Dataset")
    if join_vars not in ("inner", "exact"):
        raise ValueError("join_vars 只能为 'inner' 或 'exact'")
    fv, ov = set(fcst_ds.data_vars), set(obs_ds.data_vars)
    common = fv & ov
    for a in aux.values():
        if isinstance(a, xr.Dataset):
            common &= set(a.data_vars)
    if fv != ov or len(common) < len(fv):
        msg = f"Dataset 变量不一致：fcst {sorted(fv)}，obs {sorted(ov)}；公共变量 {sorted(common)}"
        if join_vars == "exact":
            raise ValueError(msg)
        warnings.warn(msg + "（只评估公共变量）", UserWarning, stacklevel=3)
    if not common:
        raise ValueError(f"fcst 与 obs 没有公共变量：{sorted(fv)} vs {sorted(ov)}")
    union: list = []
    for v in sorted(common):
        union += [d for d in fcst_ds[v].dims if d not in union]
    _resolve_reduce_dims(union, reduce_dims, preserve_dims, strict=True, what="Dataset 各变量维的并集")
    pd_ = _to_tuple(preserve_dims)
    named = _to_tuple(reduce_dims) or (pd_ if pd_ != ("all",) else None)
    results = {}
    for v in sorted(common):
        dims = fcst_ds[v].dims
        if named:
            missing = [d for d in named if d not in dims]
            if missing:
                warnings.warn(f"变量 {v!r} 没有维 {missing}（dims={tuple(dims)}）", UserWarning, stacklevel=3)
        vaux = {}
        for name, a in aux.items():
            if isinstance(a, dict):
                a = a.get(v)
            elif isinstance(a, xr.Dataset):
                a = a[v]
            if isinstance(a, xr.DataArray) and not set(a.dims) <= set(dims):
                raise ValueError(f"变量 {v!r}: {name} 的维 {tuple(a.dims)} 不是变量维 {tuple(dims)} 的子集；"
                                 f"可按变量传入 dict")
            vaux[name] = a
        results[v] = fn(fcst_ds[v], obs_ds[v], reduce_dims=reduce_dims, preserve_dims=preserve_dims,
                        _lenient=True, **vaux, **kw)
    if all(isinstance(r, xr.Dataset) for r in results.values()):
        return results                                   # e.g. contingency tables: dict of Datasets
    return xr.Dataset(results)


def _dims_for(fcst, reduce_dims, preserve_dims, lenient):
    return _resolve_reduce_dims(fcst.dims, reduce_dims, preserve_dims, strict=not lenient)


# ============================================================================ continuous
def _mean_metric(kind, fcst, obs, *, reduce_dims, preserve_dims, weights, skipna, weight_mode, mask,
                 join, coord_tol, lenient):
    xr = _xr()
    check_weight_mode(weight_mode)
    fcst, obs, w, m = align_inputs(fcst, obs, ("weights", weights), ("mask", mask), join=join, coord_tol=coord_tol)
    dims = list(_dims_for(fcst, reduce_dims, preserve_dims, lenient))
    if weight_mode == "multiply" and m is not None:
        warnings.warn("weight_mode='multiply' 配合 mask：Σw·x/N 不再是加权平均，请用 weight_mode='mean'",
                      UserWarning, stacklevel=3)
    with xr.set_options(arithmetic_join="exact", keep_attrs=False):
        err = fcst - obs
        x = err * err if kind == "sq" else (abs(err) if kind == "abs" else err)
        valid = _valid(err, w, mask=m, skipna=skipna)
        num, den = _weighted_sums(x, dims, w=w, valid=valid, weight_mode=weight_mode)
        return fcst, _safe_div(num, den)


def _continuous(kind, name, pow_, post=None):
    def fn(fcst: Any, obs: Any, *, reduce_dims: FlexibleDims = None, preserve_dims: FlexibleDims = None,
           weights: Any = None, skipna: bool = True, weight_mode: str = "mean", mask: Any = None,
           join: str = "reorder", coord_tol: float = 1e-4, join_vars: str = "inner",
           _lenient: bool = False) -> Any:
        if isinstance(fcst, _xr().Dataset):
            return _dataset_apply(fn, fcst, obs, aux={"weights": weights, "mask": mask},
                                  reduce_dims=reduce_dims, preserve_dims=preserve_dims, join_vars=join_vars,
                                  skipna=skipna, weight_mode=weight_mode, join=join, coord_tol=coord_tol)
        ref, out = _mean_metric(kind, fcst, obs, reduce_dims=reduce_dims, preserve_dims=preserve_dims,
                                weights=weights, skipna=skipna, weight_mode=weight_mode, mask=mask, join=join,
                                coord_tol=coord_tol, lenient=_lenient)
        return _finish(out if post is None else post(out), name, ref, pow_)
    fn.__name__ = fn.__qualname__ = name
    return fn


_COMMON_DOC = """

    Parameters
    ----------
    fcst, obs : DataArray (or Dataset: evaluated per common variable)
    reduce_dims / preserve_dims : dims to reduce / keep (one of them; default reduce all;
        ``preserve_dims='all'`` keeps every dim).  Unknown names raise.
    weights : DataArray on a subset of fcst's dims (e.g. ``latitude_weights(fcst.latitude)``),
        a dict ``{var: DataArray}`` for Datasets, or an ndarray of exactly ``fcst.shape``
    skipna, weight_mode : as in :func:`metvane.rmse`
    mask : boolean DataArray (True = evaluate), aligned like weights
    join : coordinate alignment, ``'reorder'`` (default) / ``'exact'`` / ``'inner'``
    coord_tol : tolerance for matching numeric coordinates
    join_vars : Dataset variables, ``'inner'`` (warn, use common) / ``'exact'`` (raise)

    Returns a DataArray (lazy for dask input) named after the metric, or a Dataset.
    """

mse = _continuous("sq", "mse", 2)
mse.__doc__ = "Mean squared error over named dims." + _COMMON_DOC
rmse = _continuous("sq", "rmse", 1, post=np.sqrt)
rmse.__doc__ = "Root mean squared error = sqrt(weighted MSE) over named dims." + _COMMON_DOC
mae = _continuous("abs", "mae", 1)
mae.__doc__ = "Mean absolute error over named dims." + _COMMON_DOC
bias = _continuous("err", "bias", 1)
bias.__doc__ = "Mean error (fcst − obs) over named dims." + _COMMON_DOC


def _correlation(fcst, obs, clim, *, dims, w, m, skipna, centered, mean_over, name):
    xr = _xr()
    dims = list(dims)
    with xr.set_options(arithmetic_join="exact", keep_attrs=False):
        fa, oa = (fcst - clim, obs - clim) if clim is not None else (fcst, obs)
        valid = _valid(fa, oa, w, mask=m, skipna=skipna)
        W = w if w is None or valid is None else w.where(valid, 0)
        if W is None and valid is not None:
            W = valid.astype(fa.dtype)
        if valid is not None:
            fa, oa = fa.where(valid, 0), oa.where(valid, 0)
        if W is not None:
            fa, oa = fa.where(W != 0, 0), oa.where(W != 0, 0)

        def S(x):
            return (x if W is None else x * W).sum(dims, skipna=False)

        if centered:
            sw = _nreduced(fa, dims) if W is None else _sum_w(W, fa, dims)
            fa = fa - _safe_div(S(fa), sw)
            oa = oa - _safe_div(S(oa), sw)
            if valid is not None:
                fa, oa = fa.where(valid, 0), oa.where(valid, 0)
        fo, ff, oo = S(fa * oa), S(fa * fa), S(oa * oa)
        prod = ff * oo
        score = _safe_div(fo, np.sqrt(prod.where(prod > 0, 0)))
        if mean_over:
            score = score.mean(list(mean_over), skipna=True)
    return _finish(score, name, fcst, 0)


def _corr_dims(fcst, reduce_dims, preserve_dims, mean_over, lenient):
    mo = _to_tuple(mean_over) or ()
    unknown = [d for d in mo if d not in fcst.dims]
    if unknown:
        raise ValueError(f"mean_over 的维 {unknown} 不在 fcst 的维 {tuple(fcst.dims)} 中{_DIM_HINT}")
    dims = _dims_for(fcst, reduce_dims, preserve_dims, lenient)
    if mo and reduce_dims is None:
        dims = tuple(d for d in dims if d not in mo)      # sample dims are averaged, not pooled
    overlap = [d for d in mo if d in dims]
    if overlap:
        raise ValueError(f"mean_over={mo} 与归约维 {dims} 重叠：mean_over 为样本维（起报/时间），reduce_dims 为空间维")
    return dims, mo


def acc(fcst: Any, obs: Any, climatology: Any, *, reduce_dims: FlexibleDims = None,
        preserve_dims: FlexibleDims = None, weights: Any = None, skipna: bool = True, mask: Any = None,
        centered: bool = False, mean_over: FlexibleDims = None, join: str = "reorder", coord_tol: float = 1e-4,
        join_vars: str = "inner", _lenient: bool = False) -> Any:
    """Anomaly correlation over named dims (uncentred by default).

    *reduce_dims* are pooled into one correlation (typically the spatial dims); *mean_over*
    dims (sample dims such as ``time``) are averaged **after** computing the per-sample ACC —
    ``acc(f, o, c, reduce_dims=['latitude', 'longitude'], mean_over='time')`` is the
    WeatherBench-2 / ECMWF convention, while putting ``time`` into *reduce_dims* gives a
    pooled ACC.  *climatology*: DataArray (aligned like weights; must correspond to each
    verification time), scalar, or a Dataset/dict for Dataset inputs.  Other parameters as
    in :func:`mse`.
    """
    xr = _xr()
    if isinstance(fcst, xr.Dataset):
        return _dataset_apply(acc, fcst, obs, aux={"climatology": climatology, "weights": weights, "mask": mask},
                              reduce_dims=reduce_dims, preserve_dims=preserve_dims, join_vars=join_vars,
                              skipna=skipna, centered=centered, mean_over=mean_over, join=join,
                              coord_tol=coord_tol)
    if climatology is None:
        raise ValueError("acc 需要 climatology")
    fcst, obs, c, w, m = align_inputs(fcst, obs, ("climatology", climatology), ("weights", weights),
                                      ("mask", mask), join=join, coord_tol=coord_tol)
    dims, mo = _corr_dims(fcst, reduce_dims, preserve_dims, mean_over, _lenient)
    return _correlation(fcst, obs, c, dims=dims, w=w, m=m, skipna=skipna, centered=centered,
                        mean_over=mo, name="acc")


def pearson_correlation(fcst: Any, obs: Any, *, reduce_dims: FlexibleDims = None,
                        preserve_dims: FlexibleDims = None, weights: Any = None, skipna: bool = True,
                        mask: Any = None, mean_over: FlexibleDims = None, join: str = "reorder",
                        coord_tol: float = 1e-4, join_vars: str = "inner", _lenient: bool = False) -> Any:
    """Weighted Pearson correlation over named dims (``mean_over`` as in :func:`acc`)."""
    xr = _xr()
    if isinstance(fcst, xr.Dataset):
        return _dataset_apply(pearson_correlation, fcst, obs, aux={"weights": weights, "mask": mask},
                              reduce_dims=reduce_dims, preserve_dims=preserve_dims, join_vars=join_vars,
                              skipna=skipna, mean_over=mean_over, join=join, coord_tol=coord_tol)
    fcst, obs, w, m = align_inputs(fcst, obs, ("weights", weights), ("mask", mask), join=join, coord_tol=coord_tol)
    dims, mo = _corr_dims(fcst, reduce_dims, preserve_dims, mean_over, _lenient)
    return _correlation(fcst, obs, None, dims=dims, w=w, m=m, skipna=skipna, centered=True,
                        mean_over=mo, name="pearson_correlation")


def wind_vector_rmse(u_fcst: Any, v_fcst: Any, u_obs: Any, v_obs: Any, *, reduce_dims: FlexibleDims = None,
                     preserve_dims: FlexibleDims = None, weights: Any = None, skipna: bool = True,
                     weight_mode: str = "mean", mask: Any = None, join: str = "reorder",
                     coord_tol: float = 1e-4) -> Any:
    """Wind vector RMSE over named dims (all components aligned to *u_fcst*)."""
    xr = _xr()
    check_weight_mode(weight_mode)
    uf, uo, vf, vo, w, m = align_inputs(u_fcst, u_obs, ("v_fcst", v_fcst), ("v_obs", v_obs),
                                        ("weights", weights), ("mask", mask), join=join, coord_tol=coord_tol)
    for name, a in (("v_fcst", vf), ("v_obs", vo)):
        if set(a.dims) != set(uf.dims):
            raise ValueError(f"{name} 的维 {tuple(a.dims)} 与 u_fcst 的维 {tuple(uf.dims)} 不一致")
    dims = list(_resolve_reduce_dims(uf.dims, reduce_dims, preserve_dims))
    with xr.set_options(arithmetic_join="exact", keep_attrs=False):
        sq = (uf - uo) ** 2 + (vf - vo) ** 2
        valid = _valid(sq, w, mask=m, skipna=skipna)
        num, den = _weighted_sums(sq, dims, w=w, valid=valid, weight_mode=weight_mode)
        return _finish(np.sqrt(_safe_div(num, den)), "wind_vector_rmse", uf, 1)


# ============================================================================ categorical
def contingency_table(fcst: Any, obs: Any, thresholds: Any, *, op: str = ">=",
                      reduce_dims: FlexibleDims = None, preserve_dims: FlexibleDims = None,
                      skipna: bool = True, mask: Any = None, join: str = "reorder",
                      coord_tol: float = 1e-4, join_vars: str = "inner", _lenient: bool = False) -> Any:
    """Exact int64 TP/FP/FN/TN counts as an ``xr.Dataset`` with dims ``('threshold', *kept)``.

    NaN / ``mask == False`` points are excluded from all counts (``skipna=False``: slices
    containing NaN give NaN).  Counts are additive — sum the Datasets of several files, then
    call :func:`scores_from_counts`.  For Dataset inputs a dict ``{var: Dataset}`` is returned.
    """
    xr = _xr()
    if isinstance(fcst, xr.Dataset):
        return _dataset_apply(contingency_table, fcst, obs, aux={"mask": mask}, reduce_dims=reduce_dims,
                              preserve_dims=preserve_dims, join_vars=join_vars, thresholds=thresholds, op=op,
                              skipna=skipna, join=join, coord_tol=coord_tol)
    cmp = get_op(op)
    ths = normalize_thresholds(thresholds)
    fcst, obs, m = align_inputs(fcst, obs, ("mask", mask), join=join, coord_tol=coord_tol)
    dims = list(_dims_for(fcst, reduce_dims, preserve_dims, _lenient))
    with xr.set_options(arithmetic_join="exact", keep_attrs=False):
        valid = _valid(fcst, obs, mask=m, skipna=True)
        nv = _nreduced(fcst, dims) if valid is None else _sum_w(valid, fcst, dims)
        out: dict = {k: [] for k in ("tp", "fp", "fn", "tn")}
        for t in ths:
            pb, ob = cmp(fcst, t), cmp(obs, t)
            if valid is not None:
                pb, ob = pb & valid, ob & valid
            tp = (pb & ob).sum(dims)
            npred, nobs = pb.sum(dims), ob.sum(dims)
            out["tp"].append(tp)
            out["fp"].append(npred - tp)
            out["fn"].append(nobs - tp)
            out["tn"].append(nv - npred - nobs + tp)
        thr = xr.DataArray(ths, dims="threshold", name="threshold")
        ds = xr.Dataset({k: xr.concat(v, dim=thr).astype("int64") for k, v in out.items()})
        if not skipna:
            nan_any = None
            for a in (fcst, obs):
                if np.issubdtype(a.dtype, np.floating):
                    nn = a.isnull()
                    nan_any = nn if nan_any is None else (nan_any | nn)
            if nan_any is not None:
                ds = ds.astype("float64").where(~nan_any.any(dims))
    ds.attrs = {"op": op}
    return ds


_XR_SCORES = {
    "csi": lambda c: _safe_div(c.tp, c.tp + c.fp + c.fn),
    "pod": lambda c: _safe_div(c.tp, c.tp + c.fn),
    "far": lambda c: _safe_div(c.fp, c.tp + c.fp),
    "pofd": lambda c: _safe_div(c.fp, c.fp + c.tn),
    "hss": lambda c: _safe_div(2 * (c.tp * c.tn - c.fp * c.fn),
                               (c.tp + c.fn) * (c.fn + c.tn) + (c.tp + c.fp) * (c.fp + c.tn)),
    "ets": lambda c: _safe_div(c.tp * c.tn - c.fp * c.fn,
                               (c.tp * c.tn - c.fp * c.fn) + (c.tp + c.fp + c.fn + c.tn) * (c.fp + c.fn)),
    "bias_score": lambda c: _safe_div(c.tp + c.fp, c.tp + c.fn),
    "f1": lambda c: _safe_div(2 * c.tp, 2 * c.tp + c.fp + c.fn),
    "pc": lambda c: _safe_div(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn),
}


def scores_from_counts(counts: Any, metrics: Any = None) -> Any:
    """Scores (float64, NaN where undefined) from a :func:`contingency_table` Dataset.

    *metrics*: names / aliases as in :meth:`metvane.ContingencyTable.summary` (default all).
    """
    from .categorical import ContingencyTable
    xr = _xr()
    c = counts.astype("float64")
    names = list(ContingencyTable._DEFAULT) if metrics is None else (
        [metrics] if isinstance(metrics, str) else list(metrics))
    res = {}
    for n in names:
        canon = ContingencyTable._SCORES.get(n.lower())
        if canon is None:
            raise ValueError(f"未知指标 {n!r}，可选 {sorted(ContingencyTable._SCORES)}")
        res[n] = _XR_SCORES[canon](c)
    return xr.Dataset(res)


def categorical_scores(fcst: Any, obs: Any, thresholds: Any, *, metrics: Any = None, op: str = ">=",
                       reduce_dims: FlexibleDims = None, preserve_dims: FlexibleDims = None,
                       skipna: bool = True, mask: Any = None, join: str = "reorder",
                       coord_tol: float = 1e-4) -> Any:
    """``scores_from_counts(contingency_table(...), metrics)`` for a DataArray pair."""
    if isinstance(fcst, _xr().Dataset):
        raise TypeError("categorical_scores 需要 DataArray；Dataset 请用 contingency_table 后对各变量调用 "
                        "scores_from_counts")
    counts = contingency_table(fcst, obs, thresholds, op=op, reduce_dims=reduce_dims, preserve_dims=preserve_dims,
                               skipna=skipna, mask=mask, join=join, coord_tol=coord_tol)
    return scores_from_counts(counts, metrics)
