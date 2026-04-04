"""Tests for categorical metrics."""

from __future__ import annotations

import numpy as np
import pytest

import metvane
from metvane.categorical import ContingencyTable
from .conftest import requires_torch


class TestContingencyTableNumpy:
    def test_perfect_forecast(self):
        obs = np.array([0, 10, 20, 30, 40, 50], dtype=np.float32)
        table = ContingencyTable(obs, obs, thresholds=[20])
        np.testing.assert_allclose(table.csi()[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(table.pod()[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(table.far()[0], 0.0, atol=1e-6)

    def test_all_miss(self):
        fcst = np.zeros(100, dtype=np.float32)
        obs = np.ones(100, dtype=np.float32) * 50.0
        table = ContingencyTable(fcst, obs, thresholds=[20])
        np.testing.assert_allclose(table.csi()[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(table.pod()[0], 0.0, atol=1e-6)

    def test_multiple_thresholds(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, 1000).astype(np.float32)
        obs = rng.uniform(0, 50, 1000).astype(np.float32)
        table = ContingencyTable(fcst, obs, thresholds=[10, 20, 30])
        assert table.csi().shape == (3,)
        assert table.pod().shape == (3,)
        assert table.far().shape == (3,)

    def test_per_leadtime(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, (10, 256, 256)).astype(np.float32)
        obs = rng.uniform(0, 50, (10, 256, 256)).astype(np.float32)
        table = ContingencyTable(fcst, obs, thresholds=[20, 35],
                                 axis=(1, 2))
        assert table.csi().shape == (2, 10)
        assert table.pod().shape == (2, 10)

    def test_summary(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, 500).astype(np.float32)
        obs = rng.uniform(0, 50, 500).astype(np.float32)
        table = ContingencyTable(fcst, obs, thresholds=[20])
        summary = table.summary()
        expected_keys = {"csi", "pod", "far", "pofd", "hss", "ets",
                         "bias_score", "f1", "accuracy"}
        assert set(summary.keys()) == expected_keys

    def test_hss_perfect(self):
        obs = np.array([0, 0, 50, 50], dtype=np.float32)
        table = ContingencyTable(obs, obs, thresholds=[25])
        np.testing.assert_allclose(table.hss()[0], 1.0, atol=1e-6)


class TestCategoricalShortcuts:
    def test_csi_func(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, 500).astype(np.float32)
        obs = rng.uniform(0, 50, 500).astype(np.float32)
        val = metvane.csi(fcst, obs, threshold=20)
        assert 0.0 <= float(val) <= 1.0

    def test_pod_func(self):
        obs = np.array([0, 10, 20, 30, 40, 50], dtype=np.float32)
        val = metvane.pod(obs, obs, threshold=20)
        np.testing.assert_allclose(float(val), 1.0, atol=1e-6)


@requires_torch
class TestContingencyTableTorch:
    def test_perfect_forecast(self):
        import torch
        obs = torch.tensor([0, 10, 20, 30, 40, 50], dtype=torch.float32)
        table = ContingencyTable(obs, obs, thresholds=[20])
        np.testing.assert_allclose(table.csi()[0].item(), 1.0, atol=1e-6)

    def test_backend_convert(self):
        import torch
        fcst_np = np.array([0, 10, 20, 30, 40, 50], dtype=np.float32)
        obs_np = fcst_np.copy()
        table = ContingencyTable(fcst_np, obs_np, thresholds=[20],
                                 backend="torch")
        assert isinstance(table.tp, torch.Tensor)
