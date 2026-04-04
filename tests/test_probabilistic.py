"""Tests for probabilistic metrics."""

from __future__ import annotations

import numpy as np
import pytest

import metvane


class TestBrierScore:
    def test_perfect(self):
        prob = np.array([1.0, 0.0, 1.0, 0.0])
        obs = np.array([1.0, 0.0, 1.0, 0.0])
        result = metvane.brier_score(prob, obs)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_worst(self):
        prob = np.array([1.0, 1.0])
        obs = np.array([0.0, 0.0])
        result = metvane.brier_score(prob, obs)
        np.testing.assert_allclose(result, 1.0, atol=1e-7)

    def test_range(self):
        rng = np.random.default_rng(42)
        prob = rng.uniform(0, 1, 1000)
        obs = rng.integers(0, 2, 1000).astype(float)
        result = metvane.brier_score(prob, obs)
        assert 0.0 <= result <= 1.0


class TestCRPS:
    def test_perfect_ensemble(self):
        obs = np.array([5.0])
        ensemble = np.full((10, 1), 5.0)
        result = metvane.crps_ensemble(ensemble, obs, member_axis=0)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_positive(self):
        rng = np.random.default_rng(42)
        obs = rng.standard_normal(50)
        ensemble = rng.standard_normal((20, 50))
        result = metvane.crps_ensemble(ensemble, obs, member_axis=0)
        assert result >= 0.0
