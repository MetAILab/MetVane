"""Categorical evaluation metrics.

Classes
-------
ContingencyTable
    Build once from forecasts + observations, then query any number of
    binary classification scores (CSI, POD, FAR, HSS, ETS, …).

Convenience functions
---------------------
csi, pod, far, hss, ets — single-threshold shortcuts.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..core.backend import Backend, convert, detect_backend
from . import _impl


_EPS = 1e-8


class ContingencyTable:
    """Contingency table built from continuous forecasts and observations.

    Parameters
    ----------
    fcst, obs : array-like
        Forecast and observation arrays.
    thresholds : sequence of float
        Event thresholds (e.g. ``[20, 35, 40]`` for dBZ).
    op : str
        Comparison operator (``'>='``, ``'>'``, ``'<='``, ``'<'``).
    axis : int or tuple of ints, optional
        Axes to reduce (aggregate).  Un-mentioned axes are preserved.
    backend, device : optional
        Force a specific compute backend / device.

    Examples
    --------
    >>> table = ContingencyTable(fcst, obs, thresholds=[20, 35, 40],
    ...                         axis=(0, 2, 3))
    >>> table.csi()    # shape: (3_thresholds, n_leadtimes)
    >>> table.pod()
    >>> table.summary()
    """

    def __init__(
        self,
        fcst: Any,
        obs: Any,
        thresholds: Sequence[float],
        *,
        op: str = ">=",
        axis: Any = None,
        backend: Optional[str] = None,
        device: Any = None,
    ):
        if backend is not None:
            target = Backend(backend)
            fcst = convert(fcst, target, device=device)
            obs = convert(obs, target, device=device)

        self.thresholds = list(thresholds)
        self.tp, self.fp, self.fn, self.tn = _impl.compute_contingency(
            fcst, obs, thresholds, op=op, axis=axis,
        )

    # --- core scores -------------------------------------------------------

    def csi(self) -> Any:
        """Critical Success Index (Threat Score).

        CSI = TP / (TP + FP + FN)
        """
        return self.tp / (self.tp + self.fp + self.fn + _EPS)

    threat_score = csi

    def pod(self) -> Any:
        """Probability of Detection (Hit Rate / Recall).

        POD = TP / (TP + FN)
        """
        return self.tp / (self.tp + self.fn + _EPS)

    hit_rate = recall = pod

    def far(self) -> Any:
        """False Alarm Ratio.

        FAR = FP / (TP + FP)
        """
        return self.fp / (self.tp + self.fp + _EPS)

    false_alarm_ratio = far

    def pofd(self) -> Any:
        """Probability of False Detection (False Alarm Rate).

        POFD = FP / (FP + TN)
        """
        return self.fp / (self.fp + self.tn + _EPS)

    false_alarm_rate = pofd

    def hss(self) -> Any:
        """Heidke Skill Score.

        HSS = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
        """
        num = 2 * (self.tp * self.tn - self.fp * self.fn)
        den = (
            (self.tp + self.fn) * (self.fn + self.tn)
            + (self.tp + self.fp) * (self.fp + self.tn)
        )
        return num / (den + _EPS)

    heidke_skill_score = hss

    def ets(self) -> Any:
        """Equitable Threat Score (Gilbert Skill Score).

        ETS = (TP - hits_random) / (TP + FP + FN - hits_random)
        where hits_random = (TP+FP)(TP+FN) / N
        """
        n = self.tp + self.fp + self.fn + self.tn
        hits_random = (self.tp + self.fp) * (self.tp + self.fn) / (n + _EPS)
        return (self.tp - hits_random) / (
            self.tp + self.fp + self.fn - hits_random + _EPS
        )

    equitable_threat_score = gilbert_skill_score = ets

    def bias_score(self) -> Any:
        """Frequency Bias.

        BIAS = (TP + FP) / (TP + FN)
        """
        return (self.tp + self.fp) / (self.tp + self.fn + _EPS)

    frequency_bias = bias_score

    def f1(self) -> Any:
        """F1 Score.

        F1 = 2TP / (2TP + FP + FN)
        """
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + _EPS)

    def accuracy(self) -> Any:
        """Overall accuracy.

        ACC = (TP + TN) / N
        """
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / (n + _EPS)

    def summary(self, metrics: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """Return a dict of all (or selected) scores."""
        _all = {
            "csi": self.csi,
            "pod": self.pod,
            "far": self.far,
            "pofd": self.pofd,
            "hss": self.hss,
            "ets": self.ets,
            "bias_score": self.bias_score,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }
        keys = metrics or list(_all.keys())
        return {k: _all[k]() for k in keys}


# --- convenience functions -------------------------------------------------

def csi(fcst: Any, obs: Any, threshold: float, *, op: str = ">=",
        axis: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Single-threshold CSI (shortcut)."""
    t = ContingencyTable(fcst, obs, [threshold], op=op, axis=axis,
                         backend=backend, device=device)
    return t.csi()[0]


def pod(fcst: Any, obs: Any, threshold: float, *, op: str = ">=",
        axis: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Single-threshold POD (shortcut)."""
    t = ContingencyTable(fcst, obs, [threshold], op=op, axis=axis,
                         backend=backend, device=device)
    return t.pod()[0]


def far(fcst: Any, obs: Any, threshold: float, *, op: str = ">=",
        axis: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Single-threshold FAR (shortcut)."""
    t = ContingencyTable(fcst, obs, [threshold], op=op, axis=axis,
                         backend=backend, device=device)
    return t.far()[0]


def hss(fcst: Any, obs: Any, threshold: float, *, op: str = ">=",
        axis: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Single-threshold HSS (shortcut)."""
    t = ContingencyTable(fcst, obs, [threshold], op=op, axis=axis,
                         backend=backend, device=device)
    return t.hss()[0]


def ets(fcst: Any, obs: Any, threshold: float, *, op: str = ">=",
        axis: Any = None, backend: Any = None, device: Any = None) -> Any:
    """Single-threshold ETS (shortcut)."""
    t = ContingencyTable(fcst, obs, [threshold], op=op, axis=axis,
                         backend=backend, device=device)
    return t.ets()[0]
