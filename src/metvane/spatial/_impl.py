"""Spatial verification metrics."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.backend import convert, Backend


def _uniform_filter_2d(field: np.ndarray, size: int) -> np.ndarray:
    """Fast 2-D uniform (box) filter using integral image."""
    if size <= 1:
        return field.astype(np.float64)
    field = field.astype(np.float64)
    pad = size // 2
    padded = np.pad(field, ((pad, pad), (pad, pad)), mode="constant")
    ph, pw = padded.shape
    S = np.zeros((ph + 1, pw + 1), dtype=np.float64)
    S[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = field.shape
    result = (
        S[size: h + size, size: w + size]
        - S[:h, size: w + size]
        - S[size: h + size, :w]
        + S[:h, :w]
    )
    return result / (size * size)


def fss(
    fcst: Any,
    obs: Any,
    threshold: float,
    *,
    window_size: int,
    op: str = ">=",
) -> float:
    """Fractions Skill Score for a single 2-D field.

    Parameters
    ----------
    fcst, obs : 2-D arrays (H, W)
        Forecast and observation fields.
    threshold : float
        Binary threshold for the event.
    window_size : int
        Neighbourhood window size in pixels (must be odd).
    op : str
        Comparison operator.

    Returns
    -------
    float
        FSS value in [0, 1].  1 = perfect, 0 = no skill.
    """
    import operator as _op

    _ops = {">=": _op.ge, ">": _op.gt, "<=": _op.le, "<": _op.lt}
    cmp = _ops[op]

    fcst_np = convert(fcst, Backend.NUMPY).astype(np.float64)
    obs_np = convert(obs, Backend.NUMPY).astype(np.float64)

    f_bin = cmp(fcst_np, threshold).astype(np.float64)
    o_bin = cmp(obs_np, threshold).astype(np.float64)

    f_frac = _uniform_filter_2d(f_bin, window_size)
    o_frac = _uniform_filter_2d(o_bin, window_size)

    mse_val = np.mean((f_frac - o_frac) ** 2)
    mse_ref = np.mean(f_frac ** 2) + np.mean(o_frac ** 2)

    if mse_ref == 0:
        return 1.0
    return float(1.0 - mse_val / mse_ref)
