"""Probabilistic forecast metrics — stub implementations.

These will be fully implemented in a future release.  The API signatures
are defined here so downstream code can start depending on them.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.backend import convert, Backend


def crps_ensemble(
    ensemble_fcst: Any,
    obs: Any,
    *,
    member_axis: int = 0,
    axis: Any = None,
) -> Any:
    """Continuous Ranked Probability Score for ensemble forecasts.

    Parameters
    ----------
    ensemble_fcst : array
        Ensemble forecast with shape ``(..., n_members, ...)``.
    obs : array
        Observation array (same shape as a single member).
    member_axis : int
        Axis of *ensemble_fcst* that indexes ensemble members.
    axis : int or tuple, optional
        Axes of the result to reduce.
    """
    ens = convert(ensemble_fcst, Backend.NUMPY).astype(np.float64)
    obs_np = convert(obs, Backend.NUMPY).astype(np.float64)

    n_members = ens.shape[member_axis]
    ens_sorted = np.sort(ens, axis=member_axis)

    obs_exp = np.expand_dims(obs_np, axis=member_axis)

    abs_diff = np.mean(np.abs(ens_sorted - obs_exp), axis=member_axis)

    ens_a = np.take(ens_sorted, range(n_members), axis=member_axis)
    spread = 0.0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            ai = np.take(ens_sorted, i, axis=member_axis)
            aj = np.take(ens_sorted, j, axis=member_axis)
            spread = spread + np.abs(ai - aj)

    spread = spread / (n_members * n_members)

    crps = abs_diff - spread

    if axis is not None:
        return np.mean(crps, axis=axis)
    return np.mean(crps)


def brier_score(
    prob_fcst: Any,
    obs: Any,
    *,
    axis: Any = None,
) -> Any:
    """Brier Score.

    Parameters
    ----------
    prob_fcst : array
        Probability forecasts in [0, 1].
    obs : array
        Binary observations (0 or 1).
    """
    prob = convert(prob_fcst, Backend.NUMPY).astype(np.float64)
    obs_np = convert(obs, Backend.NUMPY).astype(np.float64)
    bs = (prob - obs_np) ** 2
    if axis is not None:
        return np.mean(bs, axis=axis)
    return np.mean(bs)
