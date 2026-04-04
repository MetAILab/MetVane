"""Categorical metrics — single implementation for numpy and torch.

Core routine: compute TP / FP / FN / TN counts from continuous forecasts
and observations using one or more thresholds.
"""

from __future__ import annotations

import operator as _op
from typing import Any, Optional, Sequence

from ..core.array_ns import get_namespace
from ..core.backend import ensure_same_backend

_OPS = {
    ">=": _op.ge,
    ">": _op.gt,
    "<=": _op.le,
    "<": _op.lt,
    "==": _op.eq,
}


def compute_contingency(
    fcst: Any,
    obs: Any,
    thresholds: Sequence[float],
    *,
    op: str = ">=",
    axis: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute TP, FP, FN, TN for each threshold.

    Returns four arrays each of shape ``(n_thresholds, ...preserved_dims)``.
    """
    fcst, obs = ensure_same_backend(fcst, obs)
    xp = get_namespace(fcst)
    fcst, obs = xp.as_float(fcst), xp.as_float(obs)
    cmp = _OPS[op]

    tps, fps, fns, tns = [], [], [], []
    for thresh in thresholds:
        p_bin = cmp(fcst, thresh)
        o_bin = cmp(obs, thresh)

        tps.append(xp.sum(p_bin & o_bin, axis=axis))
        fps.append(xp.sum(p_bin & ~o_bin, axis=axis))
        fns.append(xp.sum(~p_bin & o_bin, axis=axis))
        tns.append(xp.sum(~p_bin & ~o_bin, axis=axis))

    return (
        xp.as_float(xp.stack(tps)),
        xp.as_float(xp.stack(fps)),
        xp.as_float(xp.stack(fns)),
        xp.as_float(xp.stack(tns)),
    )
