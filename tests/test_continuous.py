"""Tests for continuous (regression) metrics."""

from __future__ import annotations

import numpy as np
import pytest

import metvane
from .conftest import requires_cuda, requires_torch


class TestContinuousNumpy:
    """Functional API with numpy arrays."""

    def test_rmse_perfect(self):
        x = np.array([1.0, 2.0, 3.0])
        result = metvane.rmse(x, x)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_rmse_known(self):
        fcst = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 5.0])
        expected = np.sqrt(np.mean((fcst - obs) ** 2))
        np.testing.assert_allclose(metvane.rmse(fcst, obs), expected)

    def test_mse_known(self):
        fcst = np.array([1.0, 2.0, 3.0])
        obs = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(metvane.mse(fcst, obs), (1 + 4 + 9) / 3)

    def test_mae_known(self):
        fcst = np.array([1.0, -2.0, 3.0])
        obs = np.zeros(3)
        np.testing.assert_allclose(metvane.mae(fcst, obs), 2.0)

    def test_bias_known(self):
        fcst = np.array([3.0, 3.0, 3.0])
        obs = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(metvane.bias(fcst, obs), 2.0)

    def test_rmse_axis(self, sample_2d):
        fcst, obs = sample_2d
        result = metvane.rmse(fcst, obs, axis=1)
        assert result.shape == (20,)
        manual = np.sqrt(np.mean((fcst - obs) ** 2, axis=1))
        np.testing.assert_allclose(result, manual, rtol=1e-5)

    def test_acc(self, rng):
        clim = np.zeros(100, dtype=np.float32)
        fcst = rng.standard_normal(100).astype(np.float32)
        obs = fcst + 0.1 * rng.standard_normal(100).astype(np.float32)
        result = metvane.acc(fcst, obs, clim)
        assert result > 0.9

    def test_pearson(self, rng):
        x = rng.standard_normal(1000).astype(np.float32)
        y = x + 0.01 * rng.standard_normal(1000).astype(np.float32)
        r = metvane.pearson_correlation(x, y)
        assert float(r) > 0.99

    def test_wind_vector_rmse(self, rng):
        u_f = rng.standard_normal(100).astype(np.float32)
        v_f = rng.standard_normal(100).astype(np.float32)
        result = metvane.wind_vector_rmse(u_f, v_f, u_f, v_f)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_weighted_rmse(self, sample_4d, lat_32):
        fcst, obs = sample_4d
        w = metvane.latitude_weights(lat_32)
        w_4d = metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)
        result = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w_4d)
        assert result.shape == (10,)

    def test_backend_override(self, sample_2d):
        fcst, obs = sample_2d
        result = metvane.rmse(fcst, obs, backend="numpy")
        assert isinstance(result, np.floating) or isinstance(result, np.ndarray)


@requires_torch
class TestContinuousTorch:
    """Same metrics with torch tensors — zero code duplication."""

    def test_rmse_perfect(self):
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        result = metvane.rmse(x, x)
        assert abs(float(result)) < 1e-6

    def test_rmse_known(self):
        import torch
        fcst = torch.tensor([1.0, 2.0, 3.0])
        obs = torch.tensor([1.0, 2.0, 5.0])
        expected = (((fcst - obs) ** 2).float().mean()).sqrt()
        np.testing.assert_allclose(float(metvane.rmse(fcst, obs)),
                                   float(expected), rtol=1e-5)

    def test_rmse_axis(self):
        import torch
        fcst = torch.randn(20, 100)
        obs = torch.randn(20, 100)
        result = metvane.rmse(fcst, obs, axis=1)
        assert result.shape == (20,)

    def test_bias(self):
        import torch
        fcst = torch.tensor([3.0, 3.0, 3.0])
        obs = torch.tensor([1.0, 1.0, 1.0])
        result = metvane.bias(fcst, obs)
        np.testing.assert_allclose(float(result), 2.0, rtol=1e-5)

    def test_backend_convert_numpy_to_torch(self):
        import torch
        fcst = np.array([1.0, 2.0, 3.0])
        obs = np.array([1.0, 2.0, 5.0])
        result = metvane.rmse(fcst, obs, backend="torch")
        assert isinstance(result, torch.Tensor)

    @requires_cuda
    def test_gpu_rmse(self):
        import torch
        fcst = torch.randn(100, device="cuda")
        obs = torch.randn(100, device="cuda")
        result = metvane.rmse(fcst, obs)
        assert result.device.type == "cuda"
