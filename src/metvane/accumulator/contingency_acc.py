"""Contingency-table accumulator for categorical metrics.

Maintains running TP / FP / FN / TN counts across chunks, then derives
CSI, POD, FAR, HSS, ETS, etc. in one shot via :class:`ContingencyTable`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..core.array_ns import get_namespace
from ..core.backend import Backend, convert, ensure_same_backend
from ..categorical import ContingencyTable
from ..categorical._impl import compute_contingency


class ContingencyAccumulator:
    """Accumulator for categorical metrics over multiple data chunks.

    Parameters
    ----------
    thresholds : sequence of float
        Event thresholds (e.g. ``[20, 35, 40]`` dBZ).
    op : str
        Comparison operator.
    preserve_axes : sequence of int, optional
        Axes to keep (all others are reduced per chunk).
    backend, device : optional
        Force backend / device.

    Examples
    --------
    >>> ca = ContingencyAccumulator([20, 35, 40], preserve_axes=[1])
    >>> for pred, tgt in loader:
    ...     ca.update(pred, tgt)
    >>> result = ca.compute()         # {'csi': array, 'pod': array, ...}
    >>> table = ca.as_contingency_table()
    >>> table.hss()
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        op: str = ">=",
        preserve_axes: Optional[Sequence[int]] = None,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        self.thresholds = list(thresholds)
        self.op = op
        self.preserve_axes = (
            tuple(preserve_axes) if preserve_axes is not None else None
        )
        self._forced_backend = backend
        self._device = device

        self._tp: Optional[Any] = None
        self._fp: Optional[Any] = None
        self._fn: Optional[Any] = None
        self._tn: Optional[Any] = None

    def _reduce_axes(self, ndim: int) -> tuple[int, ...]:
        if self.preserve_axes is None:
            return tuple(range(ndim))
        keep = {a % ndim for a in self.preserve_axes}
        return tuple(i for i in range(ndim) if i not in keep)

    def update(self, fcst: Any, obs: Any) -> None:
        """Accumulate TP/FP/FN/TN from one chunk."""
        if self._forced_backend:
            target = Backend(self._forced_backend)
            fcst = convert(fcst, target, device=self._device)
            obs = convert(obs, target, device=self._device)

        fcst, obs = ensure_same_backend(fcst, obs)
        axes = self._reduce_axes(fcst.ndim)

        tp, fp, fn, tn = compute_contingency(
            fcst, obs, self.thresholds, op=self.op, axis=axes,
        )

        if self._tp is None:
            self._tp, self._fp, self._fn, self._tn = tp, fp, fn, tn
        else:
            self._tp = self._tp + tp
            self._fp = self._fp + fp
            self._fn = self._fn + fn
            self._tn = self._tn + tn

    def compute(self) -> dict[str, Any]:
        """Return all standard categorical scores."""
        table = self.as_contingency_table()
        return table.summary()

    def as_contingency_table(self) -> ContingencyTable:
        """Convert accumulated counts to a :class:`ContingencyTable`."""
        if self._tp is None:
            raise RuntimeError("No data accumulated. Call update() first.")
        table = ContingencyTable.__new__(ContingencyTable)
        table.thresholds = self.thresholds
        table.tp = self._tp
        table.fp = self._fp
        table.fn = self._fn
        table.tn = self._tn
        return table

    def reset(self) -> None:
        """Clear all accumulated states."""
        self._tp = None
        self._fp = None
        self._fn = None
        self._tn = None
