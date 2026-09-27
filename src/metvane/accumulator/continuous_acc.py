"""Continuous metric accumulator — chunked update / compute pattern.

Internally maintains running sums of squared errors, absolute errors, etc.
RMSE and MSE share the same squared-error state (statistic de-duplication).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..core.array_ns import get_namespace
from ..core.backend import Backend, convert, detect_backend, ensure_same_backend
from ..core.typing import normalize_axis
from ..continuous._impl import check_weight_mode


class ContinuousAccumulator:
    """Accumulator for continuous metrics over multiple data chunks.

    Parameters
    ----------
    metrics : sequence of str
        Metrics to compute, e.g. ``['rmse', 'mse', 'mae', 'bias']``.
    preserve_axes : sequence of int, optional
        Axes to **keep** in the output.  All other axes are reduced.
    weights : array-like, optional
        Spatial weights (e.g. latitude weights).  Must be broadcastable
        to the data shape.
    climatology : array-like, optional
        Climatology array — required when ``'acc'`` is in *metrics*.
    weight_mode : {'mean', 'multiply'}
        How *weights* enter rmse/mse/mae/bias (see :func:`metvane.rmse`):
        ``'mean'`` (default) normalizes by the summed weights over reduced axes;
        ``'multiply'`` divides by the number of points, keeping weights on
        preserved axes.  Preserving every axis (``preserve_axes`` = all) gives
        per-grid-point statistics accumulated over chunks.
    backend : {'numpy', 'torch'}, optional
        Force a specific backend.
    device : str, optional
        Torch device.

    Examples
    --------
    >>> acc = ContinuousAccumulator(['rmse', 'mae'], preserve_axes=[0])
    >>> for chunk_f, chunk_o in data_loader:
    ...     acc.update(chunk_f, chunk_o)
    >>> result = acc.compute()   # {'rmse': array, 'mae': array}
    """

    _VALID = {"rmse", "mse", "mae", "bias", "acc"}

    def __init__(
        self,
        metrics: Sequence[str] = ("rmse",),
        *,
        preserve_axes: Optional[Sequence[int]] = None,
        weights: Optional[Any] = None,
        climatology: Optional[Any] = None,
        weight_mode: str = "mean",
        backend: Optional[str] = None,
        device: Any = None,
    ):
        check_weight_mode(weight_mode)
        self._weight_mode = weight_mode
        unknown = set(metrics) - self._VALID
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Valid: {self._VALID}")

        self.metrics = tuple(metrics)
        self.preserve_axes = (
            tuple(preserve_axes) if preserve_axes is not None else None
        )
        self._weights = weights
        self._climatology = climatology
        self._forced_backend = backend
        self._device = device

        self._need_sq = "rmse" in metrics or "mse" in metrics
        self._need_abs = "mae" in metrics
        self._need_err = "bias" in metrics
        self._need_acc = "acc" in metrics

        self._sum_sq: Optional[Any] = None
        self._sum_abs: Optional[Any] = None
        self._sum_err: Optional[Any] = None
        self._count: Optional[Any] = None

        self._acc_fo: Optional[Any] = None
        self._acc_ff: Optional[Any] = None
        self._acc_oo: Optional[Any] = None
        self._acc_count: Optional[Any] = None

    def _reduce_axes(self, ndim: int) -> tuple[int, ...]:
        if self.preserve_axes is None:
            return tuple(range(ndim))
        keep = {a % ndim for a in self.preserve_axes}
        return tuple(i for i in range(ndim) if i not in keep)

    def update(self, fcst: Any, obs: Any) -> None:
        """Accumulate statistics from one chunk."""
        if self._forced_backend:
            target = Backend(self._forced_backend)
            fcst = convert(fcst, target, device=self._device)
            obs = convert(obs, target, device=self._device)

        fcst, obs = ensure_same_backend(fcst, obs)
        xp = get_namespace(fcst)
        fcst, obs = xp.as_float(fcst), xp.as_float(obs)
        axes = self._reduce_axes(fcst.ndim)

        weights = self._weights
        if weights is not None:
            weights = convert(
                weights,
                Backend.TORCH if xp.name == "torch" else Backend.NUMPY,
                device=getattr(fcst, "device", None),
            )
            weights = xp.as_float(weights)
            weights = xp.broadcast_to(weights, fcst.shape)

        err = fcst - obs

        if self._need_sq:
            se = err ** 2
            if weights is not None:
                se = se * weights
            chunk_sq = xp.sum(se, axis=axes)
            self._sum_sq = chunk_sq if self._sum_sq is None else self._sum_sq + chunk_sq

        if self._need_abs:
            ae = xp.abs(err)
            if weights is not None:
                ae = ae * weights
            chunk_abs = xp.sum(ae, axis=axes)
            self._sum_abs = chunk_abs if self._sum_abs is None else self._sum_abs + chunk_abs

        if self._need_err:
            be = err
            if weights is not None:
                be = be * weights
            chunk_err = xp.sum(be, axis=axes)
            self._sum_err = chunk_err if self._sum_err is None else self._sum_err + chunk_err

        if weights is not None and self._weight_mode == "mean":
            chunk_w = xp.sum(weights, axis=axes)
        else:
            ones = xp.ones_like(fcst)
            chunk_w = xp.sum(ones, axis=axes)
        self._count = chunk_w if self._count is None else self._count + chunk_w

        if self._need_acc:
            clim = self._climatology
            if clim is None:
                raise ValueError("climatology is required for ACC")
            clim = convert(
                clim,
                Backend.TORCH if xp.name == "torch" else Backend.NUMPY,
                device=getattr(fcst, "device", None),
            )
            clim = xp.as_float(clim)
            f_anom = fcst - clim
            o_anom = obs - clim

            w = weights if weights is not None else xp.ones_like(fcst)
            chunk_fo = xp.sum(w * f_anom * o_anom, axis=axes)
            chunk_ff = xp.sum(w * f_anom ** 2, axis=axes)
            chunk_oo = xp.sum(w * o_anom ** 2, axis=axes)

            self._acc_fo = chunk_fo if self._acc_fo is None else self._acc_fo + chunk_fo
            self._acc_ff = chunk_ff if self._acc_ff is None else self._acc_ff + chunk_ff
            self._acc_oo = chunk_oo if self._acc_oo is None else self._acc_oo + chunk_oo
            if self._acc_count is None:
                self._acc_count = chunk_w.copy() if hasattr(chunk_w, "copy") else chunk_w
            else:
                self._acc_count = self._acc_count + chunk_w

    def compute(self) -> dict[str, Any]:
        """Derive final metrics from accumulated states."""
        if self._count is None:
            raise RuntimeError("No data has been accumulated. Call update() first.")

        xp = get_namespace(self._count)
        results: dict[str, Any] = {}

        # 累计权重和（或点数）为 0 的位置没有定义，返回 NaN 而不是 0（与函数式 API 一致）
        valid = self._count > 0
        safe_count = xp.where(valid, self._count, xp.ones_like(self._count))

        def _mean(total):
            return xp.where(valid, total / safe_count, xp.zeros_like(total) + float("nan"))

        if self._need_sq:
            mean_sq = _mean(self._sum_sq)
            if "mse" in self.metrics:
                results["mse"] = mean_sq
            if "rmse" in self.metrics:
                results["rmse"] = xp.sqrt(mean_sq)

        if self._need_abs:
            results["mae"] = _mean(self._sum_abs)

        if self._need_err:
            results["bias"] = _mean(self._sum_err)

        if self._need_acc:
            den = xp.sqrt(self._acc_ff * self._acc_oo)
            safe_den = xp.where(den > 0, den, xp.ones_like(den))
            # 与 metvane.acc 相同：分母为 0（权重和为 0 或距平恒为 0）-> NaN
            results["acc"] = xp.where(den > 0, self._acc_fo / safe_den,
                                      xp.zeros_like(den) + float("nan"))

        return results

    def reset(self) -> None:
        """Clear all accumulated states."""
        self._sum_sq = None
        self._sum_abs = None
        self._sum_err = None
        self._count = None
        self._acc_fo = None
        self._acc_ff = None
        self._acc_oo = None
        self._acc_count = None
