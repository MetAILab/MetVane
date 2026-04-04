"""Tests for accumulators."""

from __future__ import annotations

import numpy as np
import pytest

import metvane
from metvane.accumulator import ContinuousAccumulator, ContingencyAccumulator
from .conftest import requires_torch


class TestContinuousAccumulator:
    def test_rmse_single_chunk(self):
        rng = np.random.default_rng(42)
        fcst = rng.standard_normal((100,)).astype(np.float32)
        obs = rng.standard_normal((100,)).astype(np.float32)
        acc = ContinuousAccumulator(["rmse"])
        acc.update(fcst, obs)
        result = acc.compute()
        expected = float(metvane.rmse(fcst, obs))
        np.testing.assert_allclose(float(result["rmse"]), expected, rtol=1e-5)

    def test_rmse_multi_chunk(self):
        rng = np.random.default_rng(42)
        fcst = rng.standard_normal((200,)).astype(np.float32)
        obs = rng.standard_normal((200,)).astype(np.float32)

        acc = ContinuousAccumulator(["rmse", "mae", "bias"])
        acc.update(fcst[:100], obs[:100])
        acc.update(fcst[100:], obs[100:])
        result = acc.compute()

        np.testing.assert_allclose(
            float(result["rmse"]),
            float(metvane.rmse(fcst, obs)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(result["mae"]),
            float(metvane.mae(fcst, obs)),
            rtol=1e-5,
        )

    def test_preserve_axes(self):
        rng = np.random.default_rng(42)
        fcst = rng.standard_normal((10, 50)).astype(np.float32)
        obs = rng.standard_normal((10, 50)).astype(np.float32)

        acc = ContinuousAccumulator(["rmse"], preserve_axes=[0])
        acc.update(fcst, obs)
        result = acc.compute()
        assert result["rmse"].shape == (10,)

    def test_reset(self):
        rng = np.random.default_rng(42)
        fcst = rng.standard_normal((50,)).astype(np.float32)
        obs = rng.standard_normal((50,)).astype(np.float32)

        acc = ContinuousAccumulator(["rmse"])
        acc.update(fcst, obs)
        acc.reset()
        with pytest.raises(RuntimeError, match="No data"):
            acc.compute()

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown"):
            ContinuousAccumulator(["nonexistent"])


class TestContingencyAccumulator:
    def test_single_chunk(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, 500).astype(np.float32)
        obs = rng.uniform(0, 50, 500).astype(np.float32)

        ca = ContingencyAccumulator([20, 35])
        ca.update(fcst, obs)
        result = ca.compute()
        assert "csi" in result
        assert "pod" in result

    def test_multi_chunk_equivalence(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, 1000).astype(np.float32)
        obs = rng.uniform(0, 50, 1000).astype(np.float32)

        ca1 = ContingencyAccumulator([20])
        ca1.update(fcst, obs)
        ref = ca1.compute()["csi"]

        ca2 = ContingencyAccumulator([20])
        ca2.update(fcst[:500], obs[:500])
        ca2.update(fcst[500:], obs[500:])
        chunked = ca2.compute()["csi"]

        np.testing.assert_allclose(chunked, ref, rtol=1e-5)

    def test_preserve_axes(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, (10, 256)).astype(np.float32)
        obs = rng.uniform(0, 50, (10, 256)).astype(np.float32)

        ca = ContingencyAccumulator([20, 35], preserve_axes=[0])
        ca.update(fcst, obs)
        result = ca.compute()
        assert result["csi"].shape == (2, 10)


@requires_torch
class TestAccumulatorTorch:
    def test_continuous_torch(self):
        import torch
        rng = np.random.default_rng(42)
        fcst = torch.tensor(rng.standard_normal(100).astype(np.float32))
        obs = torch.tensor(rng.standard_normal(100).astype(np.float32))

        acc = ContinuousAccumulator(["rmse", "mae"])
        acc.update(fcst, obs)
        result = acc.compute()
        assert isinstance(result["rmse"], torch.Tensor)

    def test_contingency_torch(self):
        import torch
        rng = np.random.default_rng(42)
        fcst = torch.tensor(rng.uniform(0, 50, 500).astype(np.float32))
        obs = torch.tensor(rng.uniform(0, 50, 500).astype(np.float32))

        ca = ContingencyAccumulator([20])
        ca.update(fcst, obs)
        result = ca.compute()
        assert isinstance(result["csi"], torch.Tensor)
