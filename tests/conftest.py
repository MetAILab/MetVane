"""Shared pytest fixtures for MetVane tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def sample_2d(rng):
    """(20-leadtime, 100-spatial) random arrays."""
    fcst = rng.standard_normal((20, 100)).astype(np.float32)
    obs = rng.standard_normal((20, 100)).astype(np.float32)
    return fcst, obs


@pytest.fixture()
def sample_4d(rng):
    """(batch, lead_time, lat, lon) = (4, 10, 32, 64)."""
    shape = (4, 10, 32, 64)
    fcst = rng.standard_normal(shape).astype(np.float32)
    obs = rng.standard_normal(shape).astype(np.float32)
    return fcst, obs


@pytest.fixture()
def lat_32():
    """32-point latitude grid from -87.2 to 87.2."""
    return np.linspace(-87.2, 87.2, 32, dtype=np.float32)


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


torch_available = _try_import_torch() is not None
requires_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


def _try_import_xarray():
    try:
        import xarray
        return xarray
    except ImportError:
        return None


xarray_available = _try_import_xarray() is not None
requires_xarray = pytest.mark.skipif(not xarray_available, reason="xarray not installed")
