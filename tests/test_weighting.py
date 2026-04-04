"""Tests for weighting utilities."""

from __future__ import annotations

import numpy as np
import pytest

from metvane.core.weighting import latitude_weights, broadcast_weights, region_mask
from .conftest import requires_torch


class TestLatitudeWeights:
    def test_shape(self):
        lat = np.linspace(-90, 90, 181, dtype=np.float32)
        w = latitude_weights(lat)
        assert w.shape == lat.shape

    def test_equator_gt_pole(self):
        lat = np.array([-90, -45, 0, 45, 90], dtype=np.float32)
        w = latitude_weights(lat, normalize=False)
        assert w[2] > w[0]
        assert w[2] > w[4]

    def test_normalized_mean(self):
        lat = np.linspace(-90, 90, 361, dtype=np.float32)
        w = latitude_weights(lat, normalize=True)
        np.testing.assert_allclose(w.mean(), 1.0, atol=1e-5)

    @requires_torch
    def test_torch_output(self):
        import torch
        lat = torch.linspace(-90, 90, 181)
        w = latitude_weights(lat)
        assert isinstance(w, torch.Tensor)


class TestBroadcastWeights:
    def test_4d(self, lat_32):
        w = latitude_weights(lat_32)
        target = (4, 10, 32, 64)
        result = broadcast_weights(w, target, lat_axis=-2)
        assert result.shape == target


class TestRegionMask:
    def test_global(self):
        lat = np.linspace(-90, 90, 181)
        lon = np.linspace(0, 360, 361)
        mask = region_mask(lat, lon, "global")
        assert mask.all()

    def test_tropics(self):
        lat = np.linspace(-90, 90, 181)
        lon = np.linspace(0, 360, 361)
        mask = region_mask(lat, lon, "tropics")
        assert mask.shape == (181, 361)
        assert not mask.all()
        assert mask.any()
