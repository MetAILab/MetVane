"""Categorical (dichotomous) evaluation metrics.

Classes
-------
ContingencyTable
    Build once from forecasts + observations (or from counts), then query any number of
    scores (CSI, POD, FAR, POFD, HSS, ETS, frequency bias, F1, PC).

Convenience functions
---------------------
csi, pod, far, pofd, hss, ets, frequency_bias, f1 — single-threshold shortcuts.

Conventions
-----------
* Counts are exact int64; scores are float64.
* A score whose denominator is 0 (e.g. CSI with no forecast and no observed events) is
  **NaN** (undefined), not 0 — ``nanmean`` over leads/cases then skips it.  Prefer pooling
  the counts (sum the tables, then compute the score) over averaging per-case scores.
* ``NaN`` / masked elements / ``mask == False`` points are excluded from all counts.
* ``op='>='`` includes the threshold; for precipitation use e.g. ``op='>='`` with 0.1 mm
  or ``op='>'`` with 0.
* Terminology: FAR = false alarm **ratio** FP/(TP+FP) (空报率);
  POFD = probability of false detection FP/(FP+TN) (误报率, a.k.a. false alarm *rate*).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np

from ..core.array_ns import get_namespace
from ..core.prepare import get_op, normalize_metrics, normalize_thresholds, resolve_backend, safe_divide
from . import _impl


class ContingencyTable:
    """Contingency table built from continuous forecasts and observations.

    Parameters
    ----------
    fcst, obs : numpy array or torch tensor of identical shape
    thresholds : float or sequence of float
        Event thresholds (e.g. ``[20, 35, 40]`` dBZ).
    op : {'>=', '>', '<=', '<', '=='}
        Event definition ``op(value, threshold)``.
    axis : int or tuple of int, optional
        Axes to reduce (``None`` = all, ``()`` = none).  Scores have shape
        ``(n_thresholds, *kept_dims)``.
    skipna : bool, default True
        Exclude NaN points from all counts; ``False`` makes NaN-containing slices NaN.
    mask : bool array, optional
        Evaluation mask (True = include), broadcastable to *fcst*.
    backend, device : optional
        Force a backend (``'numpy'`` / ``'torch'``) and torch device.

    Examples
    --------
    >>> table = ContingencyTable(fcst, obs, thresholds=[20, 35, 40], axis=(0, 2, 3))
    >>> table.csi()          # shape (3, n_lead)
    >>> table.summary(["csi", "pod", "far"])
    """

    def __init__(
        self,
        fcst: Any,
        obs: Any,
        thresholds: Any,
        *,
        op: str = ">=",
        axis: Any = None,
        skipna: bool = True,
        mask: Optional[Any] = None,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        get_op(op)
        fcst, obs, mask = resolve_backend((fcst, obs, mask), backend, device)
        self.thresholds = normalize_thresholds(thresholds)
        self.op = op
        self.tp, self.fp, self.fn, self.tn = _impl.compute_contingency(
            fcst, obs, self.thresholds, op=op, axis=axis, skipna=skipna, mask=mask,
        )

    @classmethod
    def from_counts(cls, tp: Any, fp: Any, fn: Any, tn: Any, *,
                    thresholds: Optional[Sequence[float]] = None, op: str = ">=") -> "ContingencyTable":
        """Build a table from (pooled) counts, e.g. summed over files or processes."""
        get_op(op)
        obj = cls.__new__(cls)
        xp = get_namespace(tp, fp, fn, tn)
        if xp.name == "numpy":
            tp, fp, fn, tn = (np.asarray(c) for c in (tp, fp, fn, tn))
        shapes = {tuple(c.shape) for c in (tp, fp, fn, tn)}
        if len(shapes) != 1:
            raise ValueError(f"tp/fp/fn/tn 形状不一致: {sorted(shapes)}")
        obj.tp, obj.fp, obj.fn, obj.tn = tp, fp, fn, tn
        obj.thresholds = None if thresholds is None else normalize_thresholds(thresholds)
        obj.op = op
        return obj

    # --- helpers -------------------------------------------------------------

    def _f64(self):
        xp = get_namespace(self.tp)
        return xp, tuple(xp.astype(c, xp.float64) for c in (self.tp, self.fp, self.fn, self.tn))

    @property
    def n(self) -> Any:
        """Number of valid points per table (TP+FP+FN+TN)."""
        return self.tp + self.fp + self.fn + self.tn

    # --- core scores ---------------------------------------------------------

    def csi(self) -> Any:
        """Critical Success Index (Threat Score): TP / (TP + FP + FN)."""
        xp, (tp, fp, fn, _) = self._f64()
        return safe_divide(xp, tp, tp + fp + fn)

    threat_score = csi

    def pod(self) -> Any:
        """Probability of Detection (hit rate): TP / (TP + FN)."""
        xp, (tp, _, fn, _) = self._f64()
        return safe_divide(xp, tp, tp + fn)

    hit_rate = pod

    def far(self) -> Any:
        """False Alarm **Ratio** (空报率): FP / (TP + FP)."""
        xp, (tp, fp, _, _) = self._f64()
        return safe_divide(xp, fp, tp + fp)

    false_alarm_ratio = far

    def pofd(self) -> Any:
        """Probability of False Detection (误报率, "false alarm rate"): FP / (FP + TN)."""
        xp, (_, fp, _, tn) = self._f64()
        return safe_divide(xp, fp, fp + tn)

    def false_alarm_rate(self) -> Any:
        """Deprecated alias of :meth:`pofd` (easily confused with FAR = false alarm ratio)."""
        warnings.warn("false_alarm_rate() 是 POFD = FP/(FP+TN)（误报率），不是 FAR = FP/(TP+FP)（空报率）；"
                      "请改用 pofd()，该别名将在后续版本移除", FutureWarning, stacklevel=2)
        return self.pofd()

    def hss(self) -> Any:
        """Heidke Skill Score: 2(TP·TN − FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))."""
        xp, (tp, fp, fn, tn) = self._f64()
        num = 2 * (tp * tn - fp * fn)
        den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        return safe_divide(xp, num, den)

    heidke_skill_score = hss

    def ets(self) -> Any:
        """Equitable Threat Score (Gilbert Skill Score).

        ETS = (TP − R) / (TP + FP + FN − R), R = (TP+FP)(TP+FN)/N, evaluated in the equivalent
        integer form  D / (D + N·(FP+FN)),  D = TP·TN − FP·FN.
        """
        xp, (tp, fp, fn, tn) = self._f64()
        d = tp * tn - fp * fn
        n = tp + fp + fn + tn
        return safe_divide(xp, d, d + n * (fp + fn))

    equitable_threat_score = gilbert_skill_score = ets

    def bias_score(self) -> Any:
        """Frequency bias: (TP + FP) / (TP + FN).  (Not the continuous mean error ``metvane.bias``.)"""
        xp, (tp, fp, fn, _) = self._f64()
        return safe_divide(xp, tp + fp, tp + fn)

    frequency_bias = bias_score

    def f1(self) -> Any:
        """F1 score: 2TP / (2TP + FP + FN)."""
        xp, (tp, fp, fn, _) = self._f64()
        return safe_divide(xp, 2 * tp, 2 * tp + fp + fn)

    def pc(self) -> Any:
        """Proportion Correct: (TP + TN) / N."""
        xp, (tp, fp, fn, tn) = self._f64()
        return safe_divide(xp, tp + tn, tp + fp + fn + tn)

    proportion_correct = accuracy = pc

    _SCORES = {
        "csi": "csi", "ts": "csi", "threat_score": "csi",
        "pod": "pod", "hit_rate": "pod",
        "far": "far", "false_alarm_ratio": "far",
        "pofd": "pofd",
        "hss": "hss", "heidke_skill_score": "hss",
        "ets": "ets", "gss": "ets", "equitable_threat_score": "ets", "gilbert_skill_score": "ets",
        "bias_score": "bias_score", "frequency_bias": "bias_score", "fbias": "bias_score",
        "f1": "f1",
        "pc": "pc", "accuracy": "pc", "proportion_correct": "pc",
    }
    _DEFAULT = ("csi", "pod", "far", "pofd", "hss", "ets", "bias_score", "f1", "pc")

    def summary(self, metrics: Optional[Any] = None) -> dict[str, Any]:
        """Dict of scores.  ``metrics=None`` → all canonical scores; names are case-insensitive
        and accept aliases (e.g. ``'TS'``, ``'frequency_bias'``); keys are returned as given."""
        if metrics is None:
            return {k: getattr(self, k)() for k in self._DEFAULT}
        names = (metrics,) if isinstance(metrics, str) else tuple(metrics)
        normalize_metrics(tuple(m.lower() for m in names), set(self._SCORES), what="ContingencyTable 指标")
        return {m: getattr(self, self._SCORES[m.lower()])() for m in names}

    def __repr__(self) -> str:
        return (f"ContingencyTable(thresholds={self.thresholds}, op='{self.op}', "
                f"shape={tuple(self.tp.shape)})")


# --- convenience functions -------------------------------------------------

def _scalar_threshold(threshold: Any) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        if getattr(threshold, "ndim", None) == 0:
            return float(threshold)
        raise TypeError(f"快捷函数只接受单个阈值，得到 {threshold!r}；多阈值请用 ContingencyTable")
    return float(threshold)


def _shortcut(name):
    def fn(fcst: Any, obs: Any, threshold: float, *, op: str = ">=", axis: Any = None,
           skipna: bool = True, mask: Optional[Any] = None, backend: Any = None, device: Any = None) -> Any:
        t = ContingencyTable(fcst, obs, [_scalar_threshold(threshold)], op=op, axis=axis,
                             skipna=skipna, mask=mask, backend=backend, device=device)
        return getattr(t, name)()[0]
    fn.__name__ = fn.__qualname__ = name if name != "bias_score" else "frequency_bias"
    fn.__doc__ = (f"Single-threshold :meth:`ContingencyTable.{name}` shortcut.\n\n"
                  f"Parameters as in :class:`ContingencyTable`; *threshold* must be a scalar.  "
                  f"Returns shape ``kept_dims`` (float64, NaN where undefined).")
    return fn


csi = _shortcut("csi")
pod = _shortcut("pod")
far = _shortcut("far")
pofd = _shortcut("pofd")
hss = _shortcut("hss")
ets = _shortcut("ets")
frequency_bias = _shortcut("bias_score")
f1 = _shortcut("f1")

__all__ = ["ContingencyTable", "csi", "pod", "far", "pofd", "hss", "ets", "frequency_bias", "f1"]
