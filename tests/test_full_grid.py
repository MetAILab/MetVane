"""Per-grid-point ("full grid") statistics and ``weight_mode``.

Covers: torch/numpy consistency for ``axis=()`` (no reduction), accumulators
preserving every axis, float64 latitude weights, ``weight_mode='multiply'``
(weights kept on preserved axes, e.g. eval_xrv4 ``multiply_mean1``), and the
xarray layer keeping dimension names / coordinates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import metvane
from metvane.accumulator import ContinuousAccumulator
from metvane.core.array_ns import get_namespace
from .conftest import requires_torch, requires_xarray, torch_available

cuda_available = torch_available and __import__("torch").cuda.is_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


def _fields(rng, shape=(4, 3, 5)):
    """(time, lat, lon) forecast / observation."""
    return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(shape).astype(np.float32)


def _w(lat):
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.mean()


class TestEmptyAxis:
    @requires_torch
    @pytest.mark.parametrize("fn", ["sum", "mean", "nanmean", "nansum"])
    def test_torch_matches_numpy(self, fn):
        import torch
        x = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = getattr(get_namespace(x), fn)(x, axis=())
        t = torch.from_numpy(x)
        got = getattr(get_namespace(t), fn)(t, axis=())
        assert tuple(got.shape) == ref.shape == (2, 2)
        np.testing.assert_array_equal(got.numpy(), ref)
        assert got.data_ptr() != t.data_ptr(), "应返回新张量（与 numpy 一致），不能与输入共享存储"

    @requires_torch
    def test_torch_bool_sum_is_integer_count(self):
        import torch
        b = torch.tensor([[True, False], [False, True]])
        got = get_namespace(b).sum(b, axis=())
        assert got.dtype == torch.int64 and got.tolist() == [[1, 0], [0, 1]]

    def test_mse_no_reduction_numpy(self, rng):
        f, o = _fields(rng)
        np.testing.assert_allclose(metvane.mse(f, o, axis=()), (f - o) ** 2, rtol=1e-6)

    @requires_torch
    def test_mse_no_reduction_torch(self, rng):
        import torch
        f, o = _fields(rng)
        got = metvane.mse(torch.from_numpy(f), torch.from_numpy(o), axis=())
        assert tuple(got.shape) == f.shape
        np.testing.assert_allclose(got.numpy(), (f - o) ** 2, rtol=1e-6)

    @requires_torch
    def test_contingency_no_reduction_torch(self, rng):
        import torch
        from metvane.categorical import ContingencyTable
        f, o = _fields(rng)
        tn = ContingencyTable(f, o, [0.0], axis=())
        tt = ContingencyTable(torch.from_numpy(f), torch.from_numpy(o), [0.0], axis=())
        assert tuple(tt.tp.shape) == (1,) + f.shape
        np.testing.assert_array_equal(tt.tp.numpy(), tn.tp)

    @pytest.mark.parametrize("use_torch", [False, pytest.param(True, marks=requires_torch)])
    def test_accumulator_preserve_all_axes(self, rng, use_torch):
        chunks = [_fields(rng, (3, 5)) for _ in range(4)]
        acc = ContinuousAccumulator(["mse", "mae"], preserve_axes=[0, 1])
        for f, o in chunks:
            if use_torch:
                import torch
                f, o = torch.from_numpy(f), torch.from_numpy(o)
            acc.update(f, o)
        res = acc.compute()
        mse = res["mse"].numpy() if use_torch else res["mse"]
        assert mse.shape == (3, 5)
        expect = np.mean([(f - o) ** 2 for f, o in chunks], axis=0)
        np.testing.assert_allclose(mse, expect, rtol=1e-6)


class TestLatitudeWeights:
    def test_float32_poles_positive(self):
        lat = np.linspace(90, -90, 721, dtype=np.float32)
        w = metvane.latitude_weights(lat)
        assert w.dtype == np.float32 and w[0] > 0 and w[-1] > 0
        np.testing.assert_allclose(w, _w(lat).astype(np.float32), rtol=1e-6)

    def test_float64_exact(self):
        lat = np.linspace(90, -90, 721)
        np.testing.assert_array_equal(metvane.latitude_weights(lat), _w(lat))

    @requires_torch
    def test_torch_dtype_and_values(self):
        import torch
        lat = np.linspace(90, -90, 181)
        for dt in (torch.float32, torch.float64):
            w = metvane.latitude_weights(torch.as_tensor(lat, dtype=dt))
            assert w.dtype == dt and float(w[0]) > 0
            np.testing.assert_allclose(w.numpy(), _w(lat), rtol=1e-6)


class TestWeightMode:
    def test_known_values_latitude_kept(self):
        f = np.arange(12, dtype=np.float32).reshape(2, 2, 3)    # (time, lat, lon)
        o = np.zeros_like(f)
        lat = np.array([60.0, 0.0])
        wb = metvane.broadcast_weights(_w(lat), f.shape, lat_axis=-2)
        plain = np.mean(f ** 2, axis=0)
        np.testing.assert_allclose(metvane.mse(f, o, axis=0, weights=wb), plain, rtol=1e-6)     # 'mean': 权重约掉
        np.testing.assert_allclose(metvane.mse(f, o, axis=0, weights=wb, weight_mode="multiply"),
                                   _w(lat)[:, None] * plain, rtol=1e-6)                        # 'multiply': 保留权重
        np.testing.assert_allclose(metvane.mse(f, o, axis=(0, 2), weights=wb, weight_mode="multiply"),
                                   _w(lat) * plain.mean(-1), rtol=1e-6)

    def test_multiply_equals_mean_when_latitude_reduced(self, rng):
        lat = np.linspace(90, -90, 19)
        f, o = _fields(rng, (4, 19, 8))
        wb = metvane.broadcast_weights(metvane.latitude_weights(lat), f.shape, lat_axis=-2)
        for axis in [(1, 2), (0, 1, 2), (0, 1)]:
            np.testing.assert_allclose(metvane.rmse(f, o, axis=axis, weights=wb, weight_mode="multiply"),
                                       metvane.rmse(f, o, axis=axis, weights=wb), rtol=1e-5)

    def test_sum_normalized_weights(self, rng):
        lat = np.linspace(90, -90, 7)
        f, o = _fields(rng, (4, 7, 6))
        w = _w(lat) / _w(lat).sum()
        wb = metvane.broadcast_weights(w, f.shape, lat_axis=-2)
        np.testing.assert_allclose(metvane.mae(f, o, axis=0, weights=wb, weight_mode="multiply"),
                                   w[:, None] * np.mean(np.abs(f - o), axis=0), rtol=1e-6)

    def test_multiply_skipna_counts_valid_points(self):
        f = np.array([[1.0, np.nan, 3.0], [2.0, 2.0, 2.0]], dtype=np.float32)   # (lat, lon)
        o = np.zeros_like(f)
        w = np.array([0.5, 1.5], dtype=np.float32)
        wb = metvane.broadcast_weights(w, f.shape, lat_axis=-2)
        got = metvane.mse(f, o, axis=1, weights=wb, weight_mode="multiply")
        np.testing.assert_allclose(got, [0.5 * (1 + 9) / 2, 1.5 * 4], rtol=1e-6)

    def test_default_unchanged(self, sample_4d, lat_32):
        f, o = sample_4d
        wb = metvane.broadcast_weights(metvane.latitude_weights(lat_32), f.shape, lat_axis=-2)
        np.testing.assert_array_equal(metvane.rmse(f, o, axis=(0, 2, 3), weights=wb),
                                      metvane.rmse(f, o, axis=(0, 2, 3), weights=wb, weight_mode="mean"))

    def test_invalid_mode(self, sample_2d):
        f, o = sample_2d
        with pytest.raises(ValueError, match="weight_mode"):
            metvane.rmse(f, o, weights=np.ones_like(f), weight_mode="bogus")
        with pytest.raises(ValueError, match="weight_mode"):
            ContinuousAccumulator(["rmse"], weight_mode="bogus")

    def test_accumulator_matches_functional(self, rng):
        lat = np.array([70.0, 30.0, 0.0])
        chunks = [_fields(rng, (2, 3, 4)) for _ in range(3)]     # (time, lat, lon) per chunk
        w = _w(lat)
        for mode in ("mean", "multiply"):
            acc = ContinuousAccumulator(["mse", "mae"], preserve_axes=[1, 2],
                                        weights=metvane.broadcast_weights(w, (2, 3, 4), lat_axis=-2), weight_mode=mode)
            for f, o in chunks:
                acc.update(f, o)
            res = acc.compute()
            F = np.concatenate([c[0] for c in chunks])
            O = np.concatenate([c[1] for c in chunks])
            wb = metvane.broadcast_weights(w, F.shape, lat_axis=-2)
            np.testing.assert_allclose(res["mse"], metvane.mse(F, O, axis=0, weights=wb, weight_mode=mode), rtol=1e-5)
            np.testing.assert_allclose(res["mae"], metvane.mae(F, O, axis=0, weights=wb, weight_mode=mode), rtol=1e-5)

    @requires_torch
    def test_torch_matches_numpy(self, rng):
        import torch
        lat = np.linspace(90, -90, 9)
        f, o = _fields(rng, (3, 2, 9, 10))                         # (time, level, lat, lon)
        wn = metvane.broadcast_weights(metvane.latitude_weights(lat), f.shape, lat_axis=-2)
        wt = metvane.broadcast_weights(metvane.latitude_weights(torch.as_tensor(lat)), f.shape, lat_axis=-2)
        for mode in ("mean", "multiply"):
            for axis in [(), (0,), (0, 3), (0, 2, 3)]:
                ref = metvane.mse(f, o, axis=axis, weights=wn, weight_mode=mode)
                got = metvane.mse(torch.from_numpy(f), torch.from_numpy(o), axis=axis, weights=wt, weight_mode=mode)
                np.testing.assert_allclose(got.numpy(), ref, rtol=1e-5, err_msg=f"{mode} {axis}")

    @requires_cuda
    def test_cuda_full_grid(self, rng):
        import torch
        lat = np.linspace(90, -90, 9)
        f, o = _fields(rng, (3, 9, 10))
        wn = metvane.broadcast_weights(metvane.latitude_weights(lat), f.shape, lat_axis=-2)
        wt = metvane.broadcast_weights(metvane.latitude_weights(torch.as_tensor(lat, device="cuda")), f.shape, lat_axis=-2)
        got = metvane.mse(torch.from_numpy(f).cuda(), torch.from_numpy(o).cuda(), axis=(0,), weights=wt,
                          weight_mode="multiply")
        assert got.device.type == "cuda" and tuple(got.shape) == (9, 10)
        np.testing.assert_allclose(got.cpu().numpy(), metvane.mse(f, o, axis=(0,), weights=wn, weight_mode="multiply"),
                                   rtol=1e-5)


@requires_xarray
class TestXrFullGrid:
    @staticmethod
    def _da(rng):
        import xarray as xr
        f, o = _fields(rng)
        coords = {"time": np.arange(4), "lat": [60.0, 30.0, 0.0], "lon": np.arange(5) * 1.5}
        return (xr.DataArray(f, dims=("time", "lat", "lon"), coords=coords),
                xr.DataArray(o, dims=("time", "lat", "lon"), coords=coords))

    def test_preserve_all(self, rng):
        import metvane.xr_api as mxr
        f, o = self._da(rng)
        for kw in ({"preserve_dims": "all"}, {"preserve_dims": ["time", "lat", "lon"]}, {"reduce_dims": []}):
            r = mxr.mse(f, o, **kw)
            assert r.dims == ("time", "lat", "lon"), kw
            np.testing.assert_allclose(r.values, (f.values - o.values) ** 2, rtol=1e-6)
            np.testing.assert_array_equal(r.lat.values, f.lat.values)

    def test_reduce_dims_keeps_names_and_coords(self, rng):
        import metvane.xr_api as mxr
        f, o = self._da(rng)
        r = mxr.rmse(f, o, reduce_dims="time")
        assert r.dims == ("lat", "lon") and "time" not in r.coords
        np.testing.assert_array_equal(r.lon.values, f.lon.values)

    def test_multiply_latitude_weights(self, rng):
        import xarray as xr
        import metvane.xr_api as mxr
        f, o = self._da(rng)
        w = xr.DataArray(_w(f.lat.values), dims="lat", coords={"lat": f.lat})
        r = mxr.rmse(f, o, preserve_dims=["lat", "lon"], weights=w, weight_mode="multiply")
        expect = np.sqrt(_w(f.lat.values)[:, None] * ((f.values - o.values) ** 2).mean(0))
        np.testing.assert_allclose(r.values, expect, rtol=1e-6)
        ds = mxr.rmse(xr.Dataset({"t": f}), xr.Dataset({"t": o}), preserve_dims=["lat", "lon"],
                      weights=w, weight_mode="multiply")
        np.testing.assert_allclose(ds["t"].values, expect, rtol=1e-6)
