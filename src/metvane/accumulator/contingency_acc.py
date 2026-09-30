"""Contingency-table and FSS accumulators for categorical / neighbourhood metrics.

Counts are exact int64 (float64 NaN only with ``skipna=False`` on NaN-containing slices);
scores are derived once from the pooled counts via :class:`metvane.ContingencyTable`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..categorical import ContingencyTable
from ..categorical._impl import compute_contingency
from ..core.array_ns import get_namespace
from ..core.prepare import get_op, normalize_thresholds, safe_divide
from ..spatial._impl import _check_windows, fss_components, fss_from_components
from ._base import AccumulatorBase


class ContingencyAccumulator(AccumulatorBase):
    """Accumulator for categorical metrics over multiple data chunks.

    Parameters
    ----------
    thresholds : float or sequence of float
        Event thresholds (e.g. ``[20, 35, 40]`` dBZ).
    op : {'>=', '>', '<=', '<', '=='}
    axis / preserve_axes : axes to reduce / keep (mutually exclusive; default reduce all).
        Scores have shape ``(n_thresholds, *kept)``.  Chunks may only be split along
        reduced axes (checked).
    skipna : bool, default True — NaN points are excluded from all counts.
    backend, device : optional

    Examples
    --------
    >>> ca = ContingencyAccumulator([20, 35, 40], preserve_axes=[1])
    >>> for pred, tgt in loader:           # (batch, lead, H, W), split along batch
    ...     ca.update(pred, tgt)
    >>> ca.compute()                       # {'csi': (3, n_lead), ...}
    >>> ca.as_contingency_table().hss()
    """

    def __init__(
        self,
        thresholds: Any,
        *,
        op: str = ">=",
        axis: Any = None,
        preserve_axes: Optional[Sequence[int]] = None,
        skipna: bool = True,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        get_op(op)
        self.thresholds = normalize_thresholds(thresholds)
        self.op = op
        self._skipna = bool(skipna)
        super().__init__(axis=axis, preserve_axes=preserve_axes, backend=backend, device=device)

    def _config(self) -> dict:
        c = super()._config()
        c.update(thresholds=tuple(self.thresholds), op=self.op, skipna=self._skipna)
        return c

    def update(self, fcst: Any, obs: Any, *, mask: Optional[Any] = None) -> None:
        """Accumulate TP/FP/FN/TN from one chunk (atomic)."""
        fcst, obs, mask = self._inputs(fcst, obs, mask)
        if tuple(fcst.shape) != tuple(obs.shape):
            raise ValueError(f"obs 形状 {tuple(obs.shape)} 与 fcst 形状 {tuple(fcst.shape)} 不一致")
        red = self._reduced_axes(fcst.ndim)
        kept = self._check_kept(tuple(fcst.shape), red)
        tp, fp, fn, tn = compute_contingency(fcst, obs, self.thresholds, op=self.op, axis=red,
                                             skipna=self._skipna, mask=mask)
        self._commit({"tp": tp, "fp": fp, "fn": fn, "tn": tn}, fcst.ndim, red, kept)

    def compute(self, metrics: Optional[Any] = None) -> dict[str, Any]:
        """Scores from the pooled counts (see :meth:`ContingencyTable.summary`)."""
        return self.as_contingency_table().summary(metrics)

    def as_contingency_table(self) -> ContingencyTable:
        """The accumulated counts as a :class:`ContingencyTable`."""
        st = self._require_state()
        return ContingencyTable.from_counts(st["tp"], st["fp"], st["fn"], st["tn"],
                                            thresholds=self.thresholds, op=self.op)


class FSSAccumulator(AccumulatorBase):
    """Pooled Fractions Skill Score over many fields / chunks.

    Accumulates ``Σ(F_f − F_o)²`` and ``ΣF_f² + ΣF_o²`` per threshold and window (float64)
    plus the observed-event and valid-point counts (int64), then
    ``FSS = 1 − Σnum / Σden`` (NaN when neither field has events) and
    ``FSS_useful = 0.5 + f_o / 2``.

    Parameters
    ----------
    thresholds, window_sizes : scalars or sequences (window sizes positive odd ints)
    op, skipna : as in :func:`metvane.fss`
    axis / preserve_axes : **leading** axes (the last two, H and W, are always reduced)
        to reduce / keep.  Results have shape ``(n_thr, n_win, *kept)``.

    Examples
    --------
    >>> fa = FSSAccumulator([1.0, 5.0], [1, 5, 11], preserve_axes=[1])   # keep lead
    >>> for f, o in loader:                 # (batch, lead, H, W)
    ...     fa.update(f, o)
    >>> fa.compute()["fss"].shape           # (2, 3, n_lead)
    """

    def __init__(
        self,
        thresholds: Any,
        window_sizes: Any,
        *,
        op: str = ">=",
        axis: Any = None,
        preserve_axes: Optional[Sequence[int]] = None,
        skipna: bool = True,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        get_op(op)
        self.thresholds = normalize_thresholds(thresholds)
        self.window_sizes = _check_windows(window_sizes)
        self.op = op
        self._skipna = bool(skipna)
        super().__init__(axis=axis, preserve_axes=preserve_axes, backend=backend, device=device)

    def _config(self) -> dict:
        c = super()._config()
        c.update(thresholds=tuple(self.thresholds), window_sizes=tuple(self.window_sizes), op=self.op,
                 skipna=self._skipna)
        return c

    def _reduced_axes(self, ndim: int) -> tuple[int, ...]:
        if ndim < 2:
            raise ValueError(f"FSS 需要 (..., H, W) 输入，得到 {ndim} 维")
        red = super()._reduced_axes(ndim)
        if not {ndim - 2, ndim - 1} <= set(red):
            raise ValueError("FSSAccumulator 的最后两维 (H, W) 总是被归约，不能出现在 preserve_axes 中")
        return red

    def update(self, fcst: Any, obs: Any, *, mask: Optional[Any] = None) -> None:
        """Accumulate one chunk of fields ``(..., H, W)`` (atomic)."""
        fcst, obs, mask = self._inputs(fcst, obs, mask)
        red = self._reduced_axes(fcst.ndim)
        kept = self._check_kept(tuple(fcst.shape), red)
        lead = tuple(a for a in red if a < fcst.ndim - 2)
        c = fss_components(fcst, obs, self.thresholds, self.window_sizes, op=self.op, axis=lead,
                           skipna=self._skipna, mask=mask)
        self._commit(c, fcst.ndim, red, kept)

    def compute(self, *, no_event_value: float = float("nan")) -> dict[str, Any]:
        """``{'fss': (n_thr, n_win, *kept), 'fss_useful': (n_thr, *kept), 'base_rate': (n_thr, *kept)}``."""
        st = self._require_state()
        xp = get_namespace(*st.values())
        base = safe_divide(xp, xp.astype(st["n_obs_events"], xp.float64), xp.astype(st["n_valid"], xp.float64))
        return {"fss": fss_from_components(st["num"], st["den"], no_event_value=no_event_value),
                "fss_useful": 0.5 + base / 2, "base_rate": base}
