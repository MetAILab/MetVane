"""Tests for core.backend — detection, conversion, device management."""

from __future__ import annotations

import numpy as np
import pytest

from metvane.core.backend import Backend, detect_backend, convert, ensure_same_backend
from .conftest import requires_torch, requires_xarray


class TestDetectBackend:
    def test_numpy(self):
        assert detect_backend(np.array([1.0])) == Backend.NUMPY

    @requires_torch
    def test_torch(self):
        import torch
        assert detect_backend(torch.tensor([1.0])) == Backend.TORCH

    @requires_torch
    def test_mixed_prefers_torch(self):
        import torch
        assert detect_backend(np.array([1.0]), torch.tensor([1.0])) == Backend.TORCH

    @requires_xarray
    def test_xarray(self):
        import xarray as xr
        assert detect_backend(xr.DataArray([1.0])) == Backend.XARRAY


class TestConvert:
    @requires_torch
    def test_numpy_to_torch(self):
        import torch
        x = np.array([1.0, 2.0])
        t = convert(x, Backend.TORCH)
        assert isinstance(t, torch.Tensor)
        np.testing.assert_allclose(t.numpy(), x)

    @requires_torch
    def test_torch_to_numpy(self):
        import torch
        t = torch.tensor([1.0, 2.0])
        x = convert(t, Backend.NUMPY)
        assert isinstance(x, np.ndarray)
        np.testing.assert_allclose(x, t.numpy())

    @requires_xarray
    def test_xarray_to_numpy(self):
        import xarray as xr
        da = xr.DataArray([1.0, 2.0])
        x = convert(da, Backend.NUMPY)
        assert isinstance(x, np.ndarray)

    def test_same_backend_noop(self):
        x = np.array([1.0])
        assert convert(x, Backend.NUMPY) is x


class TestEnsureSameBackend:
    @requires_torch
    def test_promotes_to_torch(self):
        import torch
        a = np.array([1.0])
        b = torch.tensor([2.0])
        ra, rb = ensure_same_backend(a, b)
        assert isinstance(ra, torch.Tensor)
        assert isinstance(rb, torch.Tensor)

    def test_all_numpy(self):
        a = np.array([1.0])
        b = np.array([2.0])
        ra, rb = ensure_same_backend(a, b)
        assert isinstance(ra, np.ndarray)
        assert isinstance(rb, np.ndarray)
