"""Spatial verification metrics — Fractions Skill Score (numpy and torch).

Definition (Roberts & Lean 2008) with zero padding at the domain edges (outside the
domain = no event).  Missing data (NaN / masked / ``mask == False``) are handled by
normalising each neighbourhood fraction by its number of valid points, and only windows
containing at least one valid point enter the sums:

.. math::
    F = \\frac{\\mathrm{box}(e \\cdot v)}{\\mathrm{box}(v)},\\qquad
    FSS = 1 - \\frac{\\sum (F_f - F_o)^2}{\\sum F_f^2 + \\sum F_o^2}

(padding counts as valid, so inputs without missing values give the classic result).  When
neither field has an event the score is undefined → NaN (``no_event_value``).

Pooling: :func:`fss_components` returns the sums, so that the FSS of a set of cases is
``1 - Σnum / Σden`` (pooled), which differs from the mean of per-case FSS values.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.prepare import (align_mask, check_same_shape, get_op, normalize_axis,
                            normalize_thresholds, prepare, safe_divide, validity)


def _check_windows(window_sizes: Any) -> list[int]:
    if isinstance(window_sizes, (int, np.integer)) and not isinstance(window_sizes, (bool, np.bool_)):
        window_sizes = [window_sizes]
    out = []
    for s in window_sizes:
        if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 1 or s % 2 == 0:
            raise ValueError(f"window_size 必须是正奇数整数，得到 {s!r}")
        out.append(int(s))
    if not out:
        raise ValueError("window_sizes 不能为空")
    return out


def _integral(xp, x, pad: int):
    """Integral image of *x* (…, H, W) zero-padded by *pad*, with a leading zero row/column."""
    p = xp.pad_last2(x, pad + 1, pad)          # shape (…, H+2·pad+1, W+2·pad+1)
    return xp.cumsum(xp.cumsum(p, -2), -1)


def _box(S, size: int, pad: int, H: int, W: int):
    p = size // 2
    r0 = pad - p
    r1 = r0 + size
    return (S[..., r1:r1 + H, r1:r1 + W] - S[..., r0:r0 + H, r1:r1 + W]
            - S[..., r1:r1 + H, r0:r0 + W] + S[..., r0:r0 + H, r0:r0 + W])


def fss_components(
    fcst: Any,
    obs: Any,
    thresholds: Any,
    window_sizes: Any,
    *,
    op: str = ">=",
    axis: Any = None,
    skipna: bool = True,
    mask: Optional[Any] = None,
) -> dict[str, Any]:
    """Additive FSS sums for fields of shape ``(..., H, W)``.

    Parameters
    ----------
    thresholds : float or sequence; window_sizes : positive odd int or sequence
    axis : leading axes (not the last two) to sum over; ``None`` = all leading axes,
        ``()`` = keep every field separately.  The last two axes are always summed.

    Returns
    -------
    dict with
      ``num``  Σ(F_f − F_o)²,  ``den``  ΣF_f² + ΣF_o²  (float64, shape ``(n_thr, n_win, *kept)``),
      ``n_obs_events``, ``n_valid`` (int64, shape ``(n_thr, *kept)``) for the base rate
      ``f_o = n_obs_events / n_valid`` and ``FSS_useful = 0.5 + f_o / 2``.
    """
    cmp = get_op(op)
    ths = normalize_thresholds(thresholds)
    wins = _check_windows(window_sizes)
    xp, f, o = prepare(fcst, obs, as_float=False)
    check_same_shape(f, ("obs", o))
    if f.ndim < 2:
        raise ValueError(f"FSS 需要 (..., H, W) 的二维场，得到 {f.ndim} 维输入 {tuple(f.shape)}")
    lead_nd = f.ndim - 2
    if lead_nd == 0:
        if axis not in (None, ()):
            raise ValueError("2 维输入没有可归约的前导轴，axis 只能为 None 或 ()")
        red = ()
    else:
        lead_ax = normalize_axis(axis, lead_nd, name="axis（前导轴）")
        red = tuple(range(lead_nd)) if lead_ax is None else lead_ax
    red_full = red + (f.ndim - 2, f.ndim - 1)

    m = align_mask(mask, f, xp)
    valid = validity(xp, f, o, mask=m, skipna=True)
    H, W = int(f.shape[-2]), int(f.shape[-1])
    P = max(wins) // 2
    i64, f64 = xp.int64, xp.float64
    shape = tuple(f.shape)

    if valid is None:
        box_v = {s: None for s in wins}
        n_valid = xp.sum(f == f, red_full, dtype=i64)
    else:
        vfull = valid if tuple(valid.shape) == shape else xp.broadcast_to(valid, shape)
        inv = xp.astype(~vfull, i64)
        S_inv = _integral(xp, inv, P)
        # padding counts as valid: box(v) = s² - box(invalid)
        box_v = {s: s * s - _box(S_inv, s, P, H, W) for s in wins}
        n_valid = xp.sum(vfull, red_full, dtype=i64)

    nums, dens, nobs = [], [], []
    for t in ths:
        ef, eo = cmp(f, t), cmp(o, t)
        if valid is not None:
            ef, eo = ef & valid, eo & valid
        nobs.append(xp.sum(eo, red_full, dtype=i64))
        Sf = _integral(xp, xp.astype(ef, i64), P)
        So = _integral(xp, xp.astype(eo, i64), P)
        num_t, den_t = [], []
        for s in wins:
            bf = xp.astype(_box(Sf, s, P, H, W), f64)
            bo = xp.astype(_box(So, s, P, H, W), f64)
            if box_v[s] is None:
                ff, fo = bf / (s * s), bo / (s * s)
            else:
                bv = xp.astype(box_v[s], f64)
                ff, fo = safe_divide(xp, bf, bv), safe_divide(xp, bo, bv)
                ok = bv > 0
                ff = xp.where(ok, ff, xp.zeros_like(ff))
                fo = xp.where(ok, fo, xp.zeros_like(fo))
            d = ff - fo
            num_t.append(xp.sum(d * d, red_full))
            den_t.append(xp.sum(ff * ff + fo * fo, red_full))
        nums.append(xp.stack(num_t))
        dens.append(xp.stack(den_t))
    out = {"num": xp.stack(nums), "den": xp.stack(dens), "n_obs_events": xp.stack(nobs),
           "n_valid": xp.stack([n_valid] * len(ths))}

    if not skipna:
        nan_any = None
        for a in (f, o):
            if xp.is_float(a):
                nn = xp.isnan(a)
                nan_any = nn if nan_any is None else (nan_any | nn)
        if nan_any is not None and xp.any(nan_any):
            has = xp.sum(nan_any, red_full, dtype=i64) > 0
            for k in ("num", "den"):
                out[k] = xp.where(has, xp.nan_like(out[k]), out[k])
    return out


def fss_from_components(num: Any, den: Any, *, no_event_value: float = float("nan")) -> Any:
    """``1 - num / den``; *no_event_value* (default NaN) where ``den == 0`` (no events in either field)."""
    from ..core.array_ns import get_namespace
    xp = get_namespace(num, den)
    score = 1 - safe_divide(xp, num, den)
    if no_event_value != no_event_value:          # NaN: safe_divide already gives NaN
        return score
    return xp.where(den == 0, xp.full_like(score, float(no_event_value)), score)


def fss(
    fcst: Any,
    obs: Any,
    threshold: float,
    *,
    window_size: int,
    op: str = ">=",
    axis: Any = None,
    skipna: bool = True,
    mask: Optional[Any] = None,
    no_event_value: float = float("nan"),
) -> Any:
    """Fractions Skill Score.

    Parameters
    ----------
    fcst, obs : arrays of shape ``(..., H, W)`` (numpy or torch, identical shapes)
    threshold : float
        Event threshold (scalar).
    window_size : int
        Neighbourhood size in grid points (positive odd integer; 1 = grid-point scale).
    op : comparison operator (``'>='`` includes the threshold)
    axis : leading axes pooled into one score (``None`` = pool all fields,
        ``()`` = one FSS per field).  Pooling uses ``1 - Σnum/Σden``.
    skipna, mask : missing-data handling (see module docstring)
    no_event_value : value where neither field has an event (default NaN = undefined;
        the pre-0.2 behaviour returned 1.0)

    Returns
    -------
    0-d array (or array of kept leading dims) on the input backend/device, in [0, 1] or NaN.
    """
    if isinstance(threshold, (list, tuple)) or getattr(threshold, "ndim", 0) not in (0, None):
        raise TypeError("fss 只接受单个阈值；多阈值/多窗口请用 fss_components")
    c = fss_components(fcst, obs, [threshold], [window_size], op=op, axis=axis, skipna=skipna, mask=mask)
    return fss_from_components(c["num"][0, 0], c["den"][0, 0], no_event_value=no_event_value)
