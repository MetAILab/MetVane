"""Slices whose summed weight is zero (or with no valid point) return NaN, not 0."""

from __future__ import annotations

import numpy as np
import pytest

import metvane
from metvane.accumulator import ContinuousAccumulator
from .conftest import requires_torch, requires_xarray

BACKENDS = ["numpy", pytest.param("torch", marks=requires_torch)]


def _to(x, backend):
    if backend == "torch":
        import torch
        return torch.from_numpy(np.ascontiguousarray(x))
    return x


def _np(x):
    return np.asarray(x.numpy() if hasattr(x, "detach") else x, dtype=np.float64)


def _data():
    f = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)   # (lat, lon)
    return f, np.zeros_like(f)


ROW0_ZERO = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
class TestFunctional:
    def test_zero_weight_row_is_nan(self, backend):
        f, o = _data()
        for fn, row1 in ((metvane.mse, (16 + 25 + 36) / 3), (metvane.mae, 5.0), (metvane.bias, 5.0),
                         (metvane.rmse, np.sqrt((16 + 25 + 36) / 3))):
            got = _np(fn(_to(f, backend), _to(o, backend), axis=1, weights=_to(ROW0_ZERO, backend)))
            assert np.isnan(got[0]), fn.__name__
            np.testing.assert_allclose(got[1], row1, rtol=1e-6, err_msg=fn.__name__)

    def test_all_zero_weights_scalar(self, backend):
        f, o = _data()
        got = metvane.rmse(_to(f, backend), _to(o, backend), weights=_to(np.zeros_like(f), backend))
        assert np.isnan(float(got))

    def test_region_mask_outside_is_nan(self, backend):
        lat = np.linspace(-80, 80, 9)
        lon = np.arange(0, 360, 30.0)
        rng = np.random.default_rng(0)
        f = rng.standard_normal((9, 12)).astype(np.float32)
        o = np.zeros_like(f)
        mask = metvane.region_mask(lat, lon, (-20, 20, 60, 150))
        w = (metvane.latitude_weights(lat)[:, None] * mask).astype(np.float32)
        got = _np(metvane.mse(_to(f, backend), _to(o, backend), axis=0, weights=_to(w, backend)))   # 逐经度
        inside = mask.any(axis=0)
        assert np.all(np.isnan(got[~inside])) and np.all(np.isfinite(got[inside]))
        expect = (f[:, inside] ** 2 * w[:, inside]).sum(0) / w[:, inside].sum(0)
        np.testing.assert_allclose(got[inside], expect, rtol=1e-5)

    def test_all_nan_slice_with_weights(self, backend):
        f, o = _data()
        f[0] = np.nan
        got = _np(metvane.mse(_to(f, backend), _to(o, backend), axis=1, weights=_to(np.ones_like(f), backend)))
        assert np.isnan(got[0])
        np.testing.assert_allclose(got[1], (16 + 25 + 36) / 3, rtol=1e-6)

    def test_multiply_zero_weight_vs_no_valid_point(self, backend):
        f, o = _data()
        f[1] = np.nan                                                   # 第 1 行无有效点
        w = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)          # 第 0 行权重为 0（贡献为 0）
        got = _np(metvane.mse(_to(f, backend), _to(o, backend), axis=1, weights=_to(w, backend),
                              weight_mode="multiply"))
        assert got[0] == 0.0 and np.isnan(got[1])

    def test_nonzero_weights_unchanged(self, backend):
        f, o = _data()
        w = np.array([[1, 2, 3], [0.5, 0.5, 1]], dtype=np.float32)
        got = _np(metvane.mse(_to(f, backend), _to(o, backend), axis=1, weights=_to(w, backend)))
        np.testing.assert_allclose(got, (f ** 2 * w).sum(1) / w.sum(1), rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
class TestAccumulator:
    def test_zero_weight_row_is_nan(self, backend):
        f, o = _data()
        acc = ContinuousAccumulator(["rmse", "mse", "mae", "bias"], preserve_axes=[0], weights=_to(ROW0_ZERO, backend))
        acc.update(_to(f, backend), _to(o, backend))
        acc.update(_to(2 * f, backend), _to(o, backend))
        res = {k: _np(v) for k, v in acc.compute().items()}
        for k in ("rmse", "mse", "mae", "bias"):
            assert np.isnan(res[k][0]), k
        np.testing.assert_allclose(res["mse"][1], ((f[1] ** 2).sum() + ((2 * f[1]) ** 2).sum()) / 6, rtol=1e-6)
        np.testing.assert_allclose(res["mae"][1], (f[1].sum() + 2 * f[1].sum()) / 6, rtol=1e-6)

    def test_acc_metric_matches_functional(self, backend):
        rng = np.random.default_rng(1)
        f = rng.standard_normal((2, 50)).astype(np.float32)
        o = f + 0.3 * rng.standard_normal((2, 50)).astype(np.float32)
        clim = np.zeros_like(f)
        w = np.vstack([np.zeros(50), np.ones(50)]).astype(np.float32)
        acc = ContinuousAccumulator(["acc"], preserve_axes=[0], weights=_to(w, backend), climatology=_to(clim, backend))
        acc.update(_to(f, backend), _to(o, backend))
        got = _np(acc.compute()["acc"])
        ref = _np(metvane.acc(_to(f, backend), _to(o, backend), _to(clim, backend), axis=1, weights=_to(w, backend)))
        assert np.isnan(got[0]) and np.isnan(ref[0])
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-5)


@requires_xarray
def test_xr_zero_weight_latitude_is_nan():
    import xarray as xr
    import metvane.xr_api as mxr
    f, o = _data()
    da_f = xr.DataArray(f, dims=("lat", "lon"), coords={"lat": [80.0, 0.0]})
    da_o = xr.zeros_like(da_f)
    w = xr.DataArray([0.0, 1.0], dims="lat", coords={"lat": da_f.lat})
    r = mxr.rmse(da_f, da_o, preserve_dims="lat", weights=w)
    assert r.dims == ("lat",) and np.isnan(float(r.sel(lat=80.0)))
    np.testing.assert_allclose(float(r.sel(lat=0.0)), np.sqrt((16 + 25 + 36) / 3), rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_correlations_zero_weights_nan_without_warning(backend):
    import warnings
    rng = np.random.default_rng(2)
    f = rng.standard_normal((2, 40)).astype(np.float32)
    o = f + 0.5 * rng.standard_normal((2, 40)).astype(np.float32)
    w = np.vstack([np.zeros(40), np.ones(40)]).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # 0/0 不应再产生 invalid-value 警告
        a = _np(metvane.acc(_to(f, backend), _to(o, backend), _to(np.zeros_like(f), backend), axis=1,
                            weights=_to(w, backend)))
        r = _np(metvane.pearson_correlation(_to(f, backend), _to(o, backend), axis=1, weights=_to(w, backend)))
    assert np.isnan(a[0]) and np.isnan(r[0]) and np.isfinite(a[1]) and np.isfinite(r[1])
    np.testing.assert_allclose(r[1], np.corrcoef(f[1], o[1])[0, 1], rtol=1e-5)
