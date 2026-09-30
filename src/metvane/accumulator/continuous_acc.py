"""Continuous metric accumulator — chunked ``update`` / ``compute``.

Uses the same weighted sums as the functional API (:func:`metvane.continuous._impl.weighted_sums`),
so for chunks split along reduced axes the result equals the one-shot functional result,
including NaN handling (``skipna``), weights, ``weight_mode`` and zero-weight → NaN.
State is float64.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

from ..continuous._impl import (check_weight_mode, correlation_from_sums, correlation_sums,
                                multiply_mode_check, weighted_sums)
from ..core.array_ns import get_namespace
from ..core.prepare import (align_mask, align_to, check_same_shape, normalize_axis, normalize_metrics,
                            safe_divide, validity)
from ._base import AccumulatorBase


class ContinuousAccumulator(AccumulatorBase):
    """Accumulator for continuous metrics over multiple data chunks.

    Parameters
    ----------
    metrics : str or sequence of str
        Any of ``'rmse', 'mse', 'mae', 'bias', 'acc', 'acc_mean'``.
        ``'acc'`` is the pooled ACC over all reduced axes; ``'acc_mean'`` is the mean over
        samples of the per-sample ACC computed over *spatial_axes* (WeatherBench-2 / ECMWF
        convention).
    axis : int or tuple, optional
        Axes to reduce (as in the functional API).  Mutually exclusive with *preserve_axes*.
    preserve_axes : int or tuple, optional
        Axes to keep; all others are reduced.  Default (neither given): reduce everything.
        **Chunks may only be split along reduced axes** (checked).
    weights : array, optional
        Default weights, broadcastable to every chunk (e.g. ``latitude_weights(lat)[:, None]``);
        converted to each chunk's backend/device.  ``update(weights=...)`` overrides it.
    climatology : array or scalar, optional
        Static climatology for ACC (broadcast to every chunk).  For time-varying climatology
        pass ``update(climatology=...)`` with each chunk instead.
    spatial_axes : int or tuple, optional
        Axes pooled into one ACC per sample for ``'acc_mean'`` (must be reduced axes).
    weight_mode : {'mean', 'multiply'}
        As in :func:`metvane.rmse`.
    skipna : bool, default True
        Exclude NaN points (as the functional API).
    backend, device : optional
        Force a backend (``device`` only with ``backend='torch'``).

    Examples
    --------
    >>> acc = ContinuousAccumulator(['rmse', 'acc'], preserve_axes=[1],
    ...                             weights=latitude_weights(lat)[:, None])
    >>> for f, o, c in loader:            # chunks of (n_init, n_lead, lat, lon), split along init
    ...     acc.update(f, o, climatology=c)
    >>> acc.compute()                     # {'rmse': (n_lead,), 'acc': (n_lead,)}
    """

    _VALID = {"rmse", "mse", "mae", "bias", "acc", "acc_mean"}

    def __init__(
        self,
        metrics: Any = ("rmse",),
        *,
        axis: Any = None,
        preserve_axes: Optional[Sequence[int]] = None,
        weights: Optional[Any] = None,
        climatology: Optional[Any] = None,
        spatial_axes: Any = None,
        weight_mode: str = "mean",
        skipna: bool = True,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        check_weight_mode(weight_mode)
        self.metrics = normalize_metrics(metrics, self._VALID)
        self._weight_mode = weight_mode
        self._skipna = bool(skipna)
        self._weights = weights
        self._climatology = climatology
        self._spatial_arg = spatial_axes
        if "acc_mean" in self.metrics and spatial_axes is None:
            raise ValueError("'acc_mean' 需要 spatial_axes=（每个样本内做空间相关的轴，如 (-2, -1)）")
        self._need_sq = bool({"rmse", "mse"} & set(self.metrics))
        self._need_abs = "mae" in self.metrics
        self._need_err = "bias" in self.metrics
        self._need_acc = "acc" in self.metrics
        self._need_accm = "acc_mean" in self.metrics
        super().__init__(axis=axis, preserve_axes=preserve_axes, backend=backend, device=device)

    def _config(self) -> dict:
        c = super()._config()
        c.update(metrics=self.metrics, weight_mode=self._weight_mode, skipna=self._skipna,
                 spatial_axes=self._spatial_arg if self._spatial_arg is None or isinstance(self._spatial_arg, int)
                 else tuple(self._spatial_arg))
        return c

    def _extra_reset(self) -> None:
        self._clim_mode: Optional[str] = None
        self._warned_static = False

    def _merge_extra(self, other) -> None:
        if self._clim_mode is None:
            self._clim_mode = other._clim_mode

    def update(self, fcst: Any, obs: Any, *, climatology: Optional[Any] = None,
               weights: Optional[Any] = None, mask: Optional[Any] = None) -> None:
        """Accumulate one chunk.

        Parameters
        ----------
        fcst, obs : arrays of identical shape (numpy or torch)
        climatology : optional per-chunk climatology (takes precedence over the constructor's);
            must correspond to each verification time of the chunk.
        weights : optional per-chunk weights (take precedence over the constructor's).
        mask : optional boolean evaluation mask for this chunk.

        All validation happens before any state is modified (atomic update).
        """
        need_clim = self._need_acc or self._need_accm
        if need_clim:
            if climatology is None and self._climatology is None:
                raise ValueError("ACC 需要 climatology：请在构造时给出静态气候态，或 update(climatology=...) 逐块传入")
            mode = "chunk" if climatology is not None else "static"
            if self._clim_mode is not None and mode != self._clim_mode:
                raise ValueError(f"气候态用法不一致：之前为 {self._clim_mode!r}，本块为 {mode!r}")
        w_in = weights if weights is not None else self._weights
        c_in = climatology if climatology is not None else self._climatology

        fcst, obs, w_in, c_in, mask = self._inputs(fcst, obs, w_in, c_in, mask)
        xp = get_namespace(fcst, obs)
        f, o = xp.as_float(fcst), xp.as_float(obs)
        check_same_shape(f, ("obs", o))
        shape = tuple(f.shape)
        red = self._reduced_axes(f.ndim)
        kept = self._check_kept(shape, red)
        w = align_to(w_in, f, xp, name="weights")
        m = align_mask(mask, f, xp)
        f64 = xp.float64
        chunk: dict[str, Any] = {}

        err = f - o
        valid = validity(xp, err, w, mask=m, skipna=self._skipna)
        if self._weight_mode == "multiply" and (self._need_sq or self._need_abs or self._need_err):
            multiply_mode_check(xp, w, valid, shape, red)
        den = None
        for key, need, x in (("sum_sq", self._need_sq, lambda: err * err),
                             ("sum_abs", self._need_abs, lambda: xp.abs(err)),
                             ("sum_err", self._need_err, lambda: err)):
            if need:
                num, d = weighted_sums(xp, x(), axis=red, weights=w, valid=valid,
                                       weight_mode=self._weight_mode, dtype=f64)
                chunk[key] = num
                den = d if den is None else den
        if den is not None:
            chunk["count"] = den

        if need_clim:
            c = align_to(c_in, f, xp, name="climatology")
            if mode == "static" and not self._warned_static and len(c.shape) > 0:
                cshape = (1,) * (f.ndim - len(c.shape)) + tuple(c.shape)
                if any(cshape[a] == 1 and shape[a] > 1 for a in red):
                    warnings.warn("静态气候态沿被归约的轴（如起报/验证时间）广播：若块内包含不同验证时刻，"
                                  "季节循环会进入距平、ACC 被高估；时变气候态请用 update(climatology=...)",
                                  UserWarning, stacklevel=2)
                self._warned_static = True
            fa, oa = f - c, o - c
            va = validity(xp, fa, oa, w, mask=m, skipna=self._skipna)
            if self._need_acc:
                s = correlation_sums(xp, fa, oa, axis=red, weights=w, valid=va, centered=False, dtype=f64)
                chunk.update(acc_fo=s["fo"], acc_ff=s["ff"], acc_oo=s["oo"])
            if self._need_accm:
                sp = normalize_axis(self._spatial_arg, f.ndim, name="spatial_axes")
                if not set(sp) <= set(red):
                    raise ValueError(f"spatial_axes={sp} 必须是被归约的轴 {red} 的子集")
                s = correlation_sums(xp, fa, oa, axis=sp, weights=w, valid=va, centered=False,
                                     dtype=f64, keepdims=True)
                score = correlation_from_sums(xp, s, centered=False)
                rest = tuple(a for a in red if a not in sp)
                ok = ~xp.isnan(score)
                sc = xp.where(ok, score, xp.zeros_like(score))
                full = tuple(1 if a in sp else shape[a] for a in range(f.ndim))
                okb = ok if tuple(ok.shape) == full else xp.broadcast_to(ok, full)
                chunk["accm_sum"] = _squeeze(xp, xp.sum(sc, rest, keepdims=True, dtype=f64), sp + rest)
                chunk["accm_n"] = _squeeze(xp, xp.sum(okb, rest, keepdims=True, dtype=f64), sp + rest)
            self._clim_mode = mode

        self._commit(chunk, f.ndim, red, kept)

    def compute(self) -> dict[str, Any]:
        """Derive the metrics from the accumulated sums (NaN where undefined)."""
        st = self._require_state()
        xp = get_namespace(*st.values())
        out: dict[str, Any] = {}
        if self._need_sq:
            mean_sq = safe_divide(xp, st["sum_sq"], st["count"])
            if "mse" in self.metrics:
                out["mse"] = mean_sq
            if "rmse" in self.metrics:
                out["rmse"] = xp.sqrt(mean_sq)
        if self._need_abs:
            out["mae"] = safe_divide(xp, st["sum_abs"], st["count"])
        if self._need_err:
            out["bias"] = safe_divide(xp, st["sum_err"], st["count"])
        if self._need_acc:
            out["acc"] = correlation_from_sums(xp, {"fo": st["acc_fo"], "ff": st["acc_ff"], "oo": st["acc_oo"]},
                                               centered=False)
        if self._need_accm:
            out["acc_mean"] = safe_divide(xp, st["accm_sum"], st["accm_n"])
        return {k: out[k] for k in self.metrics}


def _squeeze(xp, x, axes):
    for a in sorted(axes, reverse=True):
        x = x.squeeze(a) if xp.name == "torch" else x.squeeze(axis=a)
    return x
