"""Public API hygiene: docstrings, signatures, lazy imports, one-pass scores, memory."""

from __future__ import annotations

import inspect
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

import metvane


def test_public_docstrings():
    missing = [n for n in metvane.__all__
               if callable(getattr(metvane, n)) and not (getattr(metvane, n).__doc__ or "").strip()]
    assert not missing, missing


def test_common_signature():
    for fn in (metvane.rmse, metvane.mse, metvane.mae, metvane.bias):
        params = inspect.signature(fn).parameters
        for p in ("axis", "weights", "skipna", "weight_mode", "mask", "backend", "device"):
            assert p in params, (fn.__name__, p)
    for fn in (metvane.acc, metvane.pearson_correlation):
        params = inspect.signature(fn).parameters
        for p in ("skipna", "mask", "mean_over"):
            assert p in params, (fn.__name__, p)


def test_version_single_source():
    assert metvane.__version__ == "0.2.0"


def test_import_does_not_load_torch_or_xarray():
    code = "import sys, metvane; print('torch' in sys.modules, 'xarray' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_continuous_scores_matches_individual(rng):
    f, o = rng.standard_normal((2, 3, 20, 30))
    f[0, 0, 0] = np.nan
    w = np.linspace(0.5, 1.5, 20)[:, None]
    out = metvane.continuous_scores(f, o, ["rmse", "mse", "mae", "bias"], axis=(1, 2), weights=w)
    for k, fn in (("rmse", metvane.rmse), ("mse", metvane.mse), ("mae", metvane.mae), ("bias", metvane.bias)):
        np.testing.assert_allclose(out[k], fn(f, o, axis=(1, 2), weights=w))


def test_weighted_rmse_peak_memory():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((4, 181, 360)).astype(np.float32)
    o = rng.standard_normal((4, 181, 360)).astype(np.float32)
    w = metvane.latitude_weights(np.linspace(90, -90, 181))[:, None]
    metvane.rmse(f, o, axis=(1, 2), weights=w)             # warm-up
    tracemalloc.start()
    metvane.rmse(f, o, axis=(1, 2), weights=w)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak / f.nbytes < 3.0, peak / f.nbytes
