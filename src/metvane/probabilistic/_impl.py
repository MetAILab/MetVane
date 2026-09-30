"""Probabilistic forecast metrics (experimental) — numpy and torch.

* :func:`crps_ensemble` — ensemble CRPS, O(M log M) sorted closed form, optional *fair*
  (unbiased, as used by WeatherBench-2) estimator.
* :func:`brier_score` — Brier score with input validation and optional event threshold.

Reductions follow the continuous metrics: NaN = missing (``skipna``), optional weights with
``Σw·x / Σw``, zero denominators give NaN.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..continuous._impl import weighted_sums
from ..core.prepare import (align_mask, align_to, get_op, normalize_axis, prepare, safe_divide,
                            validity)


def _reduce(xp, field, *, axis, weights, mask, skipna):
    ax = normalize_axis(axis, field.ndim)
    w = align_to(weights, field, xp, name="weights")
    m = align_mask(mask, field, xp)
    valid = validity(xp, field, w, mask=m, skipna=skipna)
    num, den = weighted_sums(xp, field, axis=ax, weights=w, valid=valid)
    return safe_divide(xp, num, den)


def crps_ensemble(
    ensemble_fcst: Any,
    obs: Any,
    *,
    member_axis: int = 0,
    axis: Any = None,
    fair: bool = False,
    weights: Optional[Any] = None,
    skipna: bool = True,
    mask: Optional[Any] = None,
) -> Any:
    """Continuous Ranked Probability Score of an ensemble.

    .. math:: CRPS = \\frac{1}{M}\\sum_i |x_i - y| - \\frac{1}{2M^2}\\sum_{i,j}|x_i - x_j|

    (``fair=True`` uses ``1 / (2M(M-1))`` for the spread term — the unbiased estimator
    for finite ensembles, used by WeatherBench-2.)

    Parameters
    ----------
    ensemble_fcst : array ``(..., M, ...)`` with members on *member_axis*
    obs : array of the ensemble shape **without** the member axis (checked)
    axis : axes to average over, numbered **as in ensemble_fcst** (must not include
        *member_axis*); ``None`` = all, ``()`` = per-point CRPS
    fair : use the fair (unbiased) estimator (requires M ≥ 2)
    weights, skipna, mask : as in :func:`metvane.rmse` (shape of *obs*).  With ``skipna``,
        a point is missing if *obs* or any member is NaN.

    Returns an array on the input backend/device.
    """
    xp, ens, y = prepare(ensemble_fcst, obs)
    nd = ens.ndim
    (mem,) = normalize_axis(member_axis, nd, name="member_axis")
    exp_shape = tuple(s for i, s in enumerate(ens.shape) if i != mem)
    if tuple(y.shape) != exp_shape:
        raise ValueError(f"obs 形状 {tuple(y.shape)} 应等于集合去掉成员轴 {mem} 后的形状 {exp_shape}")
    M = int(ens.shape[mem])
    if M < 1 or (fair and M < 2):
        raise ValueError(f"成员数 M={M} 不足（fair=True 需要 M ≥ 2）")

    x = xp.sort(xp.moveaxis(ens, mem, -1), axis=-1)            # (..., M)
    yy = y[..., None]
    skill = xp.mean(xp.abs(x - yy), -1)
    k = xp.arange(M, like=x)                                    # 0..M-1
    half_pairs = xp.sum(x * (2 * k - (M - 1)), -1)              # Σ_{i<j} (x_(j) - x_(i))
    spread = half_pairs / (M * (M - 1) if fair else M * M)
    field = skill - spread

    if axis is None:
        red = None
    else:
        ax_in = normalize_axis(axis, nd, name="axis")
        if mem in ax_in:
            raise ValueError(f"axis={axis} 包含成员轴 member_axis={mem}")
        red = tuple(a - (a > mem) for a in ax_in)
    return _reduce(xp, field, axis=red, weights=weights, mask=mask, skipna=skipna)


def brier_score(
    prob_fcst: Any,
    obs: Any,
    *,
    threshold: Optional[float] = None,
    op: str = ">=",
    axis: Any = None,
    weights: Optional[Any] = None,
    skipna: bool = True,
    mask: Optional[Any] = None,
) -> Any:
    """Brier score ``mean_w[(p - o)²]``.

    Parameters
    ----------
    prob_fcst : probabilities in [0, 1] (percentages are rejected)
    obs : binary observations (0/1), or continuous values when *threshold* is given
        (event = ``op(obs, threshold)``, NaN kept as missing)
    axis, weights, skipna, mask : as in :func:`metvane.rmse`
    """
    xp, p, o = prepare(prob_fcst, obs)
    if tuple(p.shape) != tuple(o.shape):
        raise ValueError(f"obs 形状 {tuple(o.shape)} 与 prob_fcst 形状 {tuple(p.shape)} 不一致")
    pn = p[~xp.isnan(p)]
    if int(np.prod(tuple(pn.shape))) > 0:
        pmin, pmax = xp.to_scalar(pn.min()), xp.to_scalar(pn.max())
        if pmin < 0 or pmax > 1:
            hint = "（疑似百分比，请除以 100）" if pmax <= 100 and pmin >= 0 else ""
            raise ValueError(f"prob_fcst 取值 [{pmin}, {pmax}] 超出 [0, 1]{hint}")
    if threshold is not None:
        cmp = get_op(op)
        ev = cmp(o, float(threshold))
        o = xp.where(xp.isnan(o), o, xp.astype(ev, p.dtype))
    else:
        on = o[~xp.isnan(o)]
        if int(np.prod(tuple(on.shape))) > 0 and xp.any((on != 0) & (on != 1)):
            raise ValueError("obs 必须是 0/1 事件（忽略 NaN）；连续观测请给出 threshold=（及 op=）")
    bs = (p - o) ** 2
    return _reduce(xp, bs, axis=axis, weights=weights, mask=mask, skipna=skipna)
