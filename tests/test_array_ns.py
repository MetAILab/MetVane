"""Tests for core.array_ns — namespace adapter."""

from __future__ import annotations

import numpy as np
import pytest

from metvane.core.array_ns import get_namespace, NumpyNS
from .conftest import requires_torch


class TestNumpyNS:
    def test_get_namespace_numpy(self):
        x = np.array([1.0])
        ns = get_namespace(x)
        assert isinstance(ns, NumpyNS)

    def test_sum_all(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        ns = get_namespace(x)
        np.testing.assert_allclose(ns.sum(x), 10.0)

    def test_sum_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        ns = get_namespace(x)
        np.testing.assert_allclose(ns.sum(x, axis=0), [4.0, 6.0])

    def test_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ns = get_namespace(x)
        np.testing.assert_allclose(ns.mean(x), 2.5)

    def test_sqrt(self):
        x = np.array([4.0, 9.0])
        ns = get_namespace(x)
        np.testing.assert_allclose(ns.sqrt(x), [2.0, 3.0])

    def test_where(self):
        x = np.array([1.0, -2.0, 3.0])
        ns = get_namespace(x)
        result = ns.where(x > 0, x, ns.zeros_like(x))
        np.testing.assert_allclose(result, [1.0, 0.0, 3.0])

    def test_stack(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        ns = get_namespace(a)
        stacked = ns.stack([a, b])
        assert stacked.shape == (2, 2)

    def test_as_float_int_input(self):
        x = np.array([1, 2, 3])
        ns = get_namespace(x)
        y = ns.as_float(x)
        assert np.issubdtype(y.dtype, np.floating)


@requires_torch
class TestTorchNS:
    def test_get_namespace_torch(self):
        import torch
        from metvane.core.array_ns import TorchNS
        x = torch.tensor([1.0])
        ns = get_namespace(x)
        assert isinstance(ns, TorchNS)

    def test_sum_all(self):
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ns = get_namespace(x)
        assert float(ns.sum(x)) == 10.0

    def test_sum_axis(self):
        import torch
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ns = get_namespace(x)
        result = ns.sum(x, axis=0)
        np.testing.assert_allclose(result.numpy(), [4.0, 6.0])

    def test_mean(self):
        import torch
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ns = get_namespace(x)
        assert abs(float(ns.mean(x)) - 2.5) < 1e-6

    def test_nanmean(self):
        import torch
        x = torch.tensor([1.0, float("nan"), 3.0])
        ns = get_namespace(x)
        assert abs(float(ns.nanmean(x)) - 2.0) < 1e-6

    def test_broadcast_to(self):
        import torch
        x = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1)
        ns = get_namespace(x)
        result = ns.broadcast_to(x, (3, 4))
        assert result.shape == (3, 4)
