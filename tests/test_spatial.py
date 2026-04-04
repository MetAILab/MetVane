"""Tests for spatial metrics (FSS)."""

from __future__ import annotations

import numpy as np
import pytest

import metvane


class TestFSS:
    def test_perfect(self):
        field = np.zeros((64, 64), dtype=np.float32)
        field[20:40, 20:40] = 50.0
        result = metvane.fss(field, field, threshold=20, window_size=5)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_no_skill(self):
        fcst = np.ones((64, 64), dtype=np.float32) * 50.0
        obs = np.zeros((64, 64), dtype=np.float32)
        result = metvane.fss(fcst, obs, threshold=20, window_size=3)
        assert result < 0.5

    def test_range(self):
        rng = np.random.default_rng(42)
        fcst = rng.uniform(0, 50, (64, 64)).astype(np.float32)
        obs = rng.uniform(0, 50, (64, 64)).astype(np.float32)
        result = metvane.fss(fcst, obs, threshold=25, window_size=5)
        assert 0.0 <= result <= 1.0

    def test_window_size_1(self):
        rng = np.random.default_rng(42)
        field = (rng.uniform(0, 50, (32, 32)) > 25).astype(np.float32) * 50
        result = metvane.fss(field, field, threshold=25, window_size=1)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)
