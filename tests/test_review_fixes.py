"""Regression tests for the 2026-09 review (METVANE_REVIEW_REPORT.md), one class per item."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import metvane
from metvane import ContingencyTable, ContinuousAccumulator, ContingencyAccumulator, FSSAccumulator
from .conftest import requires_torch, requires_xarray


def _np_csi(f, o, t):
    v = ~np.isnan(f) & ~np.isnan(o)
    p, q = (f >= t) & v, (o >= t) & v
    tp = (p & q).sum()
    return tp / (p | q).sum()


# ============================================================================ P0-1
class TestP0_1_CategoricalNaN:
    def _data(self):
        rng = np.random.default_rng(0)
        f = rng.gamma(0.5, 4, (4, 50, 50)).astype(np.float32)
        o = f + rng.normal(0, 0.5, f.shape).astype(np.float32)
        o[rng.random(o.shape) < 0.9] = np.nan          # 10 % station grid points
        return f, o

    def test_nan_excluded_from_all_counts(self):
        f, o = self._data()
        t = ContingencyTable(f, o, [0.1, 1, 5])
        v = ~np.isnan(o)
        for i, th in enumerate([0.1, 1, 5]):
            p, q = (f >= th) & v, (o >= th) & v
            assert int(t.tp[i]) == int((p & q).sum())
            assert int(t.fp[i]) == int((p & ~q).sum())
            assert int(t.fn[i]) == int((~p & q).sum())
            assert int(t.tn[i]) == int((~p & ~q & v).sum())
            np.testing.assert_allclose(t.csi()[i], _np_csi(f, o, th))
        assert t.tp.dtype == np.int64

    def test_nan_in_fcst_and_op_lt(self):
        f, o = self._data()
        f2 = f.copy()
        f2[0, :5] = np.nan
        t = ContingencyTable(f2, o, [1.0], op="<")
        v = ~np.isnan(f2) & ~np.isnan(o)
        assert int(t.n[0]) == int(v.sum())
        assert int(t.tp[0]) == int(((f2 < 1) & (o < 1) & v).sum())

    def test_all_nan_slice(self):
        f, o = self._data()
        o[1] = np.nan
        csi = ContingencyTable(f, o, [1.0], axis=(1, 2)).csi()
        assert np.isnan(csi[0, 1]) and np.isfinite(csi[0, 0])

    def test_skipna_false_gives_nan(self):
        f, o = self._data()
        t = ContingencyTable(f, o, [1.0], axis=(1, 2), skipna=False)
        assert np.isnan(t.tp).all()

    def test_mask(self):
        f, o = self._data()
        o = np.nan_to_num(o)
        m = np.zeros(f.shape[1:], bool)
        m[:10] = True
        t = ContingencyTable(f, o, [1.0], mask=m)
        t2 = ContingencyTable(f[:, :10], o[:, :10], [1.0])
        assert int(t.tp[0]) == int(t2.tp[0]) and int(t.tn[0]) == int(t2.tn[0])

    def test_shortcuts_and_accumulator(self):
        f, o = self._data()
        ref = ContingencyTable(f, o, [1.0])
        np.testing.assert_allclose(metvane.csi(f, o, 1.0), ref.csi()[0])
        ca = ContingencyAccumulator([1.0])
        for i in range(4):
            ca.update(f[i:i + 1], o[i:i + 1])
        tab = ca.as_contingency_table()
        for k in ("tp", "fp", "fn", "tn"):
            assert int(getattr(tab, k)[0]) == int(getattr(ref, k)[0])

    @requires_torch
    def test_torch(self):
        import torch
        f, o = self._data()
        tn = ContingencyTable(torch.from_numpy(f), torch.from_numpy(o), [0.1, 1])
        tr = ContingencyTable(f, o, [0.1, 1])
        assert tn.tp.dtype == torch.int64
        np.testing.assert_array_equal(tn.tp.numpy(), tr.tp)
        np.testing.assert_allclose(tn.csi().numpy(), tr.csi())

    def test_fss_nan(self):
        rng = np.random.default_rng(1)
        f = rng.gamma(0.5, 4, (40, 40))
        o = f.copy()
        o[:, 20:] = np.nan
        assert float(metvane.fss(f, o, 1.0, window_size=5)) == pytest.approx(1.0)


# ============================================================================ P0-2
@requires_xarray
class TestP0_2_XrAlignment:
    def _da(self, rng, shape=(3, 19, 36)):
        import xarray as xr
        lat = np.linspace(90, -90, shape[1])
        lon = np.arange(shape[2]) * 10.0
        return xr.DataArray(rng.standard_normal(shape), dims=("time", "lat", "lon"),
                            coords={"time": np.arange(shape[0]), "lat": lat, "lon": lon})

    def test_reversed_latitude(self, rng):
        import metvane.xr_api as mxr
        f = self._da(rng)
        assert float(mxr.rmse(f, f.sortby("lat"))) == pytest.approx(0.0, abs=1e-12)

    def test_transposed_square(self, rng):
        import xarray as xr
        import metvane.xr_api as mxr
        a = xr.DataArray(rng.standard_normal((8, 8)), dims=("y", "x"))
        assert float(mxr.rmse(a, a.transpose("x", "y"))) == pytest.approx(0.0, abs=1e-12)

    def test_climatology_reversed(self, rng):
        import metvane.xr_api as mxr
        f, o, c = self._da(rng), self._da(rng), self._da(rng)
        a1 = float(mxr.acc(f, o, c))
        a2 = float(mxr.acc(f, o, c.sortby("lat")))
        assert a1 == pytest.approx(a2, abs=1e-12)

    def test_obs_missing_dim_raises(self, rng):
        import metvane.xr_api as mxr
        f = self._da(rng)
        with pytest.raises(ValueError):
            mxr.rmse(f, f.isel(time=0))

    def test_mismatched_coords_raise(self, rng):
        import metvane.xr_api as mxr
        f = self._da(rng)
        o = f.assign_coords(lat=f.lat + 0.5)
        with pytest.raises(ValueError, match="坐标"):
            mxr.rmse(f, o)
        with pytest.raises(ValueError, match="exact"):
            mxr.rmse(f, f.sortby("lat"), join="exact")

    def test_functional_rejects_xarray(self, rng):
        f = self._da(rng)
        with pytest.raises(TypeError, match="xr_api"):
            metvane.rmse(f, f)
        with pytest.raises(TypeError):
            ContingencyTable(f, f, [0.0])


# ============================================================================ P0-3
class TestP0_3_AccumulatorPreservedAxis:
    def test_split_along_preserved_axis_raises(self, rng):
        x = rng.standard_normal((40, 8, 16))
        acc = ContinuousAccumulator(["rmse"], preserve_axes=[0])
        acc.update(x[:10], x[:10] + 1)
        with pytest.raises(ValueError, match="保留轴"):
            acc.update(x[10:17], x[10:17])

    def test_first_chunk_length_one_raises(self):
        acc = ContinuousAccumulator(["mse"], preserve_axes=[0])
        acc.update(np.ones(3), np.zeros(3))
        with pytest.raises(ValueError):
            acc.update(np.ones(1), np.zeros(1))

    def test_contingency_split_along_preserved_raises(self, rng):
        x = rng.random((20, 4, 4))
        ca = ContingencyAccumulator([0.5], preserve_axes=[0])
        ca.update(x[:5], x[:5])
        with pytest.raises(ValueError):
            ca.update(x[5:7], x[5:7])

    def test_split_along_reduced_equals_functional(self, rng):
        f = rng.standard_normal((10, 4, 8, 16))
        o = rng.standard_normal((10, 4, 8, 16))
        w = metvane.latitude_weights(np.linspace(-80, 80, 8))[:, None]
        acc = ContinuousAccumulator(["rmse", "mae", "bias"], preserve_axes=[1], weights=w)
        for s in (slice(0, 4), slice(4, 8), slice(8, 10)):
            acc.update(f[s], o[s])
        out = acc.compute()
        np.testing.assert_allclose(out["rmse"], metvane.rmse(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
        np.testing.assert_allclose(out["mae"], metvane.mae(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
        np.testing.assert_allclose(out["bias"], metvane.bias(f, o, axis=(0, 2, 3), weights=w), rtol=1e-10)


# ============================================================================ P0-4
@requires_torch
class TestP0_4_WeightsBackend:
    def test_torch_data_numpy_weights(self, rng):
        import torch
        f, o = rng.standard_normal((2, 3, 19, 36)).astype(np.float32)
        w = metvane.latitude_weights(np.linspace(-90, 90, 19))[:, None]
        ref = metvane.rmse(f, o, axis=(1, 2), weights=w)
        for fn in (metvane.rmse, metvane.mae, metvane.pearson_correlation):
            out = fn(torch.from_numpy(f), torch.from_numpy(o), axis=(1, 2), weights=w)
            assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
        out = metvane.rmse(torch.from_numpy(f), torch.from_numpy(o), axis=(1, 2), weights=w)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)
        a = metvane.acc(torch.from_numpy(f), torch.from_numpy(o), np.float64(0.0), axis=(1, 2), weights=w)
        assert isinstance(a, torch.Tensor)

    def test_float32_data_float64_weights_stays_float32(self, rng):
        f, o = rng.standard_normal((2, 4, 5)).astype(np.float32)
        out = metvane.rmse(f, o, axis=(0,), weights=np.ones((4, 5)))
        assert out.dtype == np.float32

    def test_device_without_backend_raises(self, rng):
        f = rng.standard_normal(5)
        with pytest.raises(ValueError, match="backend"):
            metvane.rmse(f, f, device="cpu")
        with pytest.warns(DeprecationWarning, match="xr_api"):
            out = metvane.rmse(f, f, backend="xarray")
        assert type(out).__module__ == "numpy"

    def test_forced_backend(self, rng):
        import torch
        f = rng.standard_normal(5)
        assert isinstance(metvane.rmse(f, f, backend="torch", device="cpu"), torch.Tensor)


# ============================================================================ P1-1
class TestP1_1_ZeroDenominatorNaN:
    def _table(self, tp, fp, fn, tn):
        return ContingencyTable.from_counts(*(np.array([v], np.int64) for v in (tp, fp, fn, tn)))

    def test_no_events(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            t = self._table(0, 0, 0, 100)
            for m in ("csi", "pod", "far", "ets", "bias_score", "f1"):
                assert np.isnan(getattr(t, m)()[0]), m
            assert t.pofd()[0] == 0.0 and t.pc()[0] == 1.0

    def test_single_false_alarm(self):
        t = self._table(0, 1, 0, 99)
        assert np.isnan(t.bias_score()[0])
        assert t.far()[0] == 1.0

    def test_all_events_perfect(self):
        t = self._table(100, 0, 0, 0)
        assert np.isnan(t.hss()[0]) and np.isnan(t.ets()[0])
        assert t.csi()[0] == 1.0

    def test_per_lead_zero_event(self):
        f = np.zeros((3, 10))
        o = np.zeros((3, 10))
        f[0, :5] = o[0, :5] = 1
        csi = ContingencyTable(f, o, [0.5], axis=1).csi()[0]
        assert csi[0] == 1.0 and np.isnan(csi[1:]).all()
        assert np.nanmean(csi) == 1.0

    @requires_torch
    def test_torch_nan(self):
        import torch
        t = ContingencyTable.from_counts(*(torch.tensor([v]) for v in (0, 0, 0, 10)))
        assert torch.isnan(t.csi()).all() and t.csi().dtype == torch.float64


# ============================================================================ P1-2
class TestP1_2_FSS:
    def test_no_events_nan(self):
        z = np.zeros((10, 10))
        assert np.isnan(metvane.fss(z, z, 1.0, window_size=3))
        assert float(metvane.fss(z, z, 1.0, window_size=3, no_event_value=1.0)) == 1.0

    @pytest.mark.parametrize("w", [4, 0, -3, 2.0, True])
    def test_window_validation(self, w):
        z = np.zeros((10, 10))
        with pytest.raises(ValueError, match="正奇数"):
            metvane.fss(z, z, 1.0, window_size=w)

    def test_ndim_validation(self):
        with pytest.raises(ValueError, match="H, W"):
            metvane.fss(np.zeros(5), np.zeros(5), 1.0, window_size=1)

    def test_pooled_vs_per_frame(self, rng):
        f = np.zeros((10, 20, 20))
        o = np.zeros((10, 20, 20))
        f[0, 5:10, 5:10] = 1
        o[0, 6:11, 6:11] = 1
        f[1, 2:4, 2:4] = 1
        per = metvane.fss(f, o, 0.5, window_size=3, axis=())
        c = metvane.fss_components(f, o, [0.5], [3], axis=())
        pooled = 1 - c["num"][0, 0].sum() / c["den"][0, 0].sum()
        assert float(metvane.fss(f, o, 0.5, window_size=3)) == pytest.approx(pooled)
        assert np.isnan(per[2:]).all()

    def test_against_scipy_uniform_filter(self, rng):
        ndimage = pytest.importorskip("scipy.ndimage")
        f = rng.gamma(0.6, 3, (37, 45))
        o = rng.gamma(0.6, 3, (37, 45))
        for s in (1, 3, 7):
            ff = ndimage.uniform_filter((f >= 2).astype(float), s, mode="constant", cval=0)
            fo = ndimage.uniform_filter((o >= 2).astype(float), s, mode="constant", cval=0)
            ref = 1 - np.mean((ff - fo) ** 2) / (np.mean(ff ** 2) + np.mean(fo ** 2))
            assert float(metvane.fss(f, o, 2.0, window_size=s)) == pytest.approx(ref, rel=1e-12)

    def test_fss_accumulator(self, rng):
        f = rng.gamma(0.6, 3, (6, 2, 20, 24))
        o = rng.gamma(0.6, 3, (6, 2, 20, 24))
        fa = FSSAccumulator([1.0, 3.0], [1, 5], preserve_axes=[1])
        fa.update(f[:4], o[:4])
        fa.update(f[4:], o[4:])
        out = fa.compute()
        assert out["fss"].shape == (2, 2, 2)
        for li in range(2):
            ref = metvane.fss(f[:, li], o[:, li], 3.0, window_size=5)
            assert out["fss"][1, 1, li] == pytest.approx(float(ref))
        base = (o >= 1.0).mean(axis=(0, 2, 3))
        np.testing.assert_allclose(out["fss_useful"][0], 0.5 + base / 2)


# ============================================================================ P1-3 / P1-4
class TestP1_3_4_ACC:
    def _data(self, rng):
        n, h, w = 10, 8, 16
        clim = rng.standard_normal((n, h, w)) * 5          # time-varying climatology
        o = clim + rng.standard_normal((n, h, w)) * np.linspace(0.2, 3, n)[:, None, None]
        f = o + rng.standard_normal((n, h, w))
        return f, o, clim

    def test_time_varying_climatology_chunks(self, rng):
        f, o, c = self._data(rng)
        ref = metvane.acc(f, o, c)
        acc = ContinuousAccumulator(["acc"])
        for s in (slice(0, 4), slice(4, 8), slice(8, 10)):
            acc.update(f[s], o[s], climatology=c[s])
        assert float(acc.compute()["acc"]) == pytest.approx(float(ref), abs=1e-10)

    def test_missing_clim_raises_at_update(self, rng):
        acc = ContinuousAccumulator(["acc"])
        with pytest.raises(ValueError, match="climatology"):
            acc.update(np.ones(3), np.ones(3))

    def test_scalar_climatology(self, rng):
        f, o, _ = self._data(rng)
        acc = ContinuousAccumulator(["acc"], climatology=0.0)
        acc.update(f, o)
        assert float(acc.compute()["acc"]) == pytest.approx(float(metvane.acc(f, o, 0.0)))

    def test_mixed_clim_modes_raise(self, rng):
        f, o, c = self._data(rng)
        acc = ContinuousAccumulator(["acc"], climatology=c[0])
        acc.update(f[:2], o[:2], climatology=c[:2])
        with pytest.raises(ValueError, match="不一致"):
            acc.update(f[2:4], o[2:4])

    def test_per_sample_mean_vs_pooled(self, rng):
        f, o, c = self._data(rng)
        per = np.array([np.sum((f[i] - c[i]) * (o[i] - c[i]))
                        / np.sqrt(np.sum((f[i] - c[i]) ** 2) * np.sum((o[i] - c[i]) ** 2)) for i in range(len(f))])
        np.testing.assert_allclose(metvane.acc(f, o, c, axis=(1, 2)), per)
        assert float(metvane.acc(f, o, c, axis=(1, 2), mean_over=0)) == pytest.approx(per.mean())
        pooled = float(metvane.acc(f, o, c))
        assert pooled != pytest.approx(per.mean(), abs=1e-3)
        acc = ContinuousAccumulator(["acc", "acc_mean"], spatial_axes=(1, 2))
        for s in (slice(0, 3), slice(3, 10)):
            acc.update(f[s], o[s], climatology=c[s])
        out = acc.compute()
        assert float(out["acc_mean"]) == pytest.approx(per.mean())
        assert float(out["acc"]) == pytest.approx(pooled)

    def test_mean_over_overlap_raises(self, rng):
        f, o, c = self._data(rng)
        with pytest.raises(ValueError, match="mean_over"):
            metvane.acc(f, o, c, axis=(0, 1, 2), mean_over=0)

    def test_centered(self, rng):
        f, o, c = self._data(rng)
        a = metvane.acc(f + 1.5, o, c, axis=(1, 2), centered=True)
        ref = [np.corrcoef((f[i] + 1.5 - c[i]).ravel(), (o[i] - c[i]).ravel())[0, 1] for i in range(len(f))]
        np.testing.assert_allclose(a, ref, rtol=1e-10)


# ============================================================================ P1-5
class TestP1_5_RegionMask:
    lat = np.linspace(90, -90, 181)
    lon360 = np.arange(0, 360, 1.0)
    lon180 = np.arange(-180, 180, 1.0)

    def test_all_regions_same_count_on_both_grids(self):
        for name in metvane.REGIONS:
            a = metvane.region_mask(self.lat, self.lon360, name).sum()
            b = metvane.region_mask(self.lat, self.lon180, name).sum()
            assert a == b and a > 0, name
        assert metvane.region_mask(self.lat, self.lon180, "global").all()

    def test_dateline_box(self):
        m = metvane.region_mask(self.lat, self.lon360, (0, 10, 160, -140))
        assert m.sum() == 11 * 61

    def test_2d_latlon(self):
        la, lo = np.meshgrid(self.lat[:5], self.lon360[:4], indexing="ij")
        assert metvane.region_mask(la, lo, "global").shape == (5, 4)

    def test_empty(self):
        lat = np.linspace(20, 40, 5)
        with pytest.raises(ValueError, match="没有选中"):
            metvane.region_mask(lat, self.lon360, "SH")
        with pytest.warns(UserWarning):
            metvane.region_mask(lat, self.lon360, (-50, -40, 0, 10))
        with pytest.raises(ValueError):
            metvane.region_mask(lat, self.lon360, (40, 20, 0, 10))

    @requires_torch
    def test_torch_output(self):
        import torch
        m = metvane.region_mask(torch.from_numpy(self.lat), torch.from_numpy(self.lon360), "NH")
        assert isinstance(m, torch.Tensor) and m.dtype == torch.bool


# ============================================================================ P1-6 / P2-18 / P2-19
class TestWeights:
    def test_1d_weights_rejected_for_nd(self, rng):
        f, o = rng.standard_normal((2, 64, 64))
        w = metvane.latitude_weights(np.linspace(-89, 89, 64))
        with pytest.raises(ValueError, match="broadcast_weights"):
            metvane.rmse(f, o, weights=w)
        a = metvane.rmse(f, o, weights=w[:, None])
        b = metvane.rmse(f, o, weights=metvane.broadcast_weights(w, f.shape, lat_axis=0))
        assert a == pytest.approx(b)
        ref = np.sqrt(np.sum(w[:, None] * (f - o) ** 2) / (np.sum(w) * 64))
        assert a == pytest.approx(ref)

    @requires_xarray
    def test_latitude_weights_dataarray(self):
        import xarray as xr
        lat = xr.DataArray(np.linspace(-80, 80, 9), dims="latitude")
        w = metvane.latitude_weights(lat)
        assert isinstance(w, xr.DataArray) and w.dims == ("latitude",)

    def test_validation(self):
        with pytest.raises(ValueError, match="90"):
            metvane.latitude_weights(np.linspace(0, 180, 10))
        with pytest.warns(UserWarning, match="弧度"):
            metvane.latitude_weights(np.linspace(-1.5, 1.5, 10))
        with pytest.raises(ValueError, match="lat_axis"):
            metvane.broadcast_weights(np.ones(4), (2, 4, 5), lat_axis=-4)
        with pytest.raises(ValueError, match="不一致"):
            metvane.broadcast_weights(np.ones(4), (2, 4, 5), lat_axis=-1)

    def test_area_weights(self):
        lat = np.linspace(90, -90, 181)
        wc = metvane.latitude_weights(lat)
        wa = metvane.latitude_weights(lat, method="area")
        assert wc[0] < 1e-10 and wa[0] > 1e-3
        np.testing.assert_allclose(wa.mean(), 1.0)
        np.testing.assert_allclose(wa[1:-1] / wc[1:-1], (wa[1:-1] / wc[1:-1])[90], rtol=1e-3)
        with pytest.warns(UserWarning, match="不均匀"):
            metvane.latitude_weights(np.array([0.0, 1, 3, 7, 15]))


# ============================================================================ P1-7
@requires_xarray
class TestP1_7_XrWeights:
    def test_weights_reversed_and_superset(self, rng):
        import xarray as xr
        import metvane.xr_api as mxr
        lat = np.linspace(90, -90, 19)
        f = xr.DataArray(rng.standard_normal((2, 19, 8)), dims=("t", "lat", "lon"), coords={"lat": lat})
        o = xr.DataArray(rng.standard_normal((2, 19, 8)), dims=("t", "lat", "lon"), coords={"lat": lat})
        w = xr.DataArray((lat > 31).astype(float), dims="lat", coords={"lat": lat}).sortby("lat")
        ref = float(np.sqrt(((f - o) ** 2).weighted(w).mean()))
        assert float(mxr.rmse(f, o, weights=w)) == pytest.approx(ref)
        glat = np.linspace(90, -90, 37)
        wg = metvane.latitude_weights(xr.DataArray(glat, dims="lat", coords={"lat": glat}))
        sub = f.sel(lat=slice(40, -40))
        out = mxr.rmse(sub, o.sel(lat=slice(40, -40)), weights=wg)
        ref = float(np.sqrt(((sub - o.sel(lat=slice(40, -40))) ** 2).weighted(wg.sel(lat=sub.lat)).mean()))
        assert float(out) == pytest.approx(ref)

    def test_ndarray_weights_must_match_shape(self, rng):
        import xarray as xr
        import metvane.xr_api as mxr
        f = xr.DataArray(rng.standard_normal((3, 4)), dims=("a", "b"))
        with pytest.raises(ValueError, match="形状"):
            mxr.rmse(f, f, weights=np.ones(4))


# ============================================================================ P1-8
class TestP1_8_NegativeAxes:
    @pytest.mark.parametrize("shape", [(3, 4, 5), (5, 5, 6), (8, 8, 8)])
    def test_pearson_negative_axes(self, rng, shape):
        f, o = rng.standard_normal((2,) + shape)
        a = metvane.pearson_correlation(f, o, axis=(-2, -1))
        b = metvane.pearson_correlation(f, o, axis=[1, 2])
        ref = [np.corrcoef(f[i].ravel(), o[i].ravel())[0, 1] for i in range(shape[0])]
        np.testing.assert_allclose(a, ref, rtol=1e-12)
        np.testing.assert_allclose(b, ref, rtol=1e-12)

    @requires_torch
    def test_torch(self, rng):
        import torch
        f, o = rng.standard_normal((2, 5, 5, 6))
        a = metvane.pearson_correlation(torch.from_numpy(f), torch.from_numpy(o), axis=(-2, -1))
        ref = [np.corrcoef(f[i].ravel(), o[i].ravel())[0, 1] for i in range(5)]
        np.testing.assert_allclose(a.numpy(), ref, rtol=1e-10)


# ============================================================================ P1-9 / P2-2
class TestDtype:
    def test_float16(self, rng):
        f = (rng.standard_normal((2, 181, 360)) * 300).astype(np.float16)
        o = np.zeros_like(f)
        ref = metvane.rmse(f.astype(np.float64), o.astype(np.float64))
        w = metvane.latitude_weights(np.linspace(90, -90, 181))[:, None]
        np.testing.assert_allclose(metvane.rmse(f, o), ref, rtol=1e-3)
        refw = metvane.rmse(f.astype(np.float64), o.astype(np.float64), weights=w)
        np.testing.assert_allclose(metvane.rmse(f, o, weights=w), refw, rtol=1e-3)

    @requires_torch
    def test_torch_float64_kept(self):
        import torch
        a = torch.tensor([1e8 + 1], dtype=torch.float64)
        b = torch.tensor([1e8], dtype=torch.float64)
        out = metvane.bias(a, b)
        assert out.dtype == torch.float64 and float(out) == 1.0

    @requires_torch
    def test_threshold_boundary_consistent(self):
        import torch
        x = np.array([0.1, 0.0999999], dtype=np.float32)
        tn = ContingencyTable(x, x, [0.1])
        tt = ContingencyTable(torch.from_numpy(x), torch.from_numpy(x), [0.1])
        assert int(tn.tp[0]) == int(tt.tp[0]) == 1
        c = metvane.fss_components(x.reshape(1, 2), x.reshape(1, 2), [0.1], [1])
        assert int(c["n_obs_events"][0]) == 1


# ============================================================================ P1-10 / P1-11 / P1-12 / P1-13
class TestMissingValues:
    def test_acc_pearson_skipna(self, rng):
        f, o, c = rng.standard_normal((3, 50))
        f[3] = np.nan
        v = ~np.isnan(f)
        assert float(metvane.acc(f, o, c)) == pytest.approx(float(metvane.acc(f[v], o[v], c[v])))
        assert float(metvane.pearson_correlation(f, o)) == pytest.approx(np.corrcoef(f[v], o[v])[0, 1])
        assert np.isnan(metvane.acc(f, o, c, skipna=False))
        c2 = c.copy()
        c2[7] = np.nan
        v2 = v & ~np.isnan(c2)
        assert float(metvane.acc(f, o, c2)) == pytest.approx(float(metvane.acc(f[v2], o[v2], c2[v2])))

    def test_accumulator_nan_equals_functional(self, rng):
        f, o = rng.standard_normal((2, 6, 3, 5, 7))
        f[0, 0, 0, 0] = np.nan
        o[3, 1, 2, :] = np.nan
        w = np.linspace(0.5, 1.5, 5)[:, None]
        acc = ContinuousAccumulator(["rmse", "mae", "acc"], preserve_axes=[1], weights=w, climatology=0.0)
        acc.update(f[:3], o[:3])
        acc.update(f[3:], o[3:])
        out = acc.compute()
        np.testing.assert_allclose(out["rmse"], metvane.rmse(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
        np.testing.assert_allclose(out["mae"], metvane.mae(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
        np.testing.assert_allclose(out["acc"], metvane.acc(f, o, 0.0, axis=(0, 2, 3), weights=w), rtol=1e-12)
        assert np.isfinite(out["rmse"]).all()

    def test_inf_is_a_value(self):
        x = np.array([1.0, 2.0, np.inf, 4.0])
        z = np.zeros(4)
        assert np.isinf(metvane.mae(x, z)) and np.isinf(metvane.mae(x, z, weights=np.ones(4)))
        assert metvane.mae(x, z, weights=np.array([1.0, 1, 0, 1])) == pytest.approx(7 / 3)

    def test_masked_array(self):
        f = np.ma.masked_array([1.0, 2.0, 9.97e36], mask=[0, 0, 1])
        o = np.array([1.0, 2.0, 0.0])
        assert metvane.rmse(f, o) == 0.0
        assert metvane.rmse(f, o, weights=np.ones(3)) == 0.0
        t = ContingencyTable(f, o, [0.5])
        assert int(t.n[0]) == 2
        wm = np.ma.masked_array([1.0, 1.0, 1.0], mask=[0, 1, 0])
        assert metvane.rmse(np.array([1.0, 5.0, 1.0]), np.ones(3), weights=wm) == 0.0


# ============================================================================ P1-14
class TestP1_14_Shapes:
    def test_no_implicit_broadcast(self, rng):
        a = rng.random(100)
        with pytest.raises(ValueError, match="形状"):
            ContingencyTable(a, a[:, None], [0.5])
        with pytest.raises(ValueError, match="形状"):
            metvane.rmse(a, a[:, None])
        with pytest.raises(ValueError, match="形状"):
            metvane.crps_ensemble(rng.random((10, 5)), rng.random((10, 1)))
        with pytest.raises(ValueError):
            metvane.rmse(a, 0.0)


# ============================================================================ P1-15 / P1-16 / P2-8 / P2-9
@requires_xarray
class TestXrApi:
    def _da(self, rng, name="t2m"):
        import xarray as xr
        return xr.DataArray(rng.standard_normal((4, 6, 8)), dims=("lead_time", "latitude", "longitude"),
                            coords={"latitude": np.linspace(-50, 50, 6)}, name=name, attrs={"units": "K"})

    def test_unknown_dims(self, rng):
        import metvane.xr_api as mxr
        f = self._da(rng)
        with pytest.raises(ValueError, match="latitude"):
            mxr.rmse(f, f, reduce_dims=["lat", "lon"])
        with pytest.raises(ValueError, match="lead"):
            mxr.rmse(f, f, preserve_dims="step")
        assert mxr.rmse(f, f, preserve_dims="all").shape == f.shape

    def test_dask_lazy(self, rng):
        pytest.importorskip("dask")
        import metvane.xr_api as mxr
        f, o = self._da(rng), self._da(rng)
        fl, ol = f.chunk({"lead_time": 1}), o.chunk({"lead_time": 1})
        w = metvane.latitude_weights(f.latitude)
        out = mxr.rmse(fl, ol, preserve_dims="lead_time", weights=w)
        assert out.chunks is not None
        ref = metvane.rmse(f.values, o.values, axis=(1, 2), weights=w.values[:, None])
        np.testing.assert_allclose(out.compute().values, ref, rtol=1e-12)
        a = mxr.acc(fl, ol, 0.0, reduce_dims=["latitude", "longitude"], mean_over="lead_time")
        ref = metvane.acc(f.values, o.values, 0.0, axis=(1, 2), mean_over=0)
        assert float(a.compute()) == pytest.approx(float(ref), abs=1e-12)

    def test_name_units_and_all(self, rng):
        import metvane.xr_api as mxr
        f = self._da(rng)
        r = mxr.rmse(f, f + 1, preserve_dims="lead_time")
        assert r.name == "rmse" and r.attrs["units"] == "K"
        assert {"rmse", "acc", "pearson_correlation", "wind_vector_rmse", "contingency_table"} <= set(mxr.__all__)

    def test_dataset_paths(self, rng):
        import xarray as xr
        import metvane.xr_api as mxr
        fd = xr.Dataset({"t2m": self._da(rng), "z": self._da(rng)})
        od = xr.Dataset({"t2m": self._da(rng), "u": self._da(rng)})
        with pytest.warns(UserWarning, match="变量不一致"):
            out = mxr.rmse(fd, od, preserve_dims="lead_time")
        assert list(out.data_vars) == ["t2m"]
        with pytest.raises(ValueError):
            mxr.rmse(fd, od, join_vars="exact")
        a = mxr.acc(fd[["t2m"]], od[["t2m"]], 0.0, preserve_dims="lead_time")
        assert a["t2m"].shape == (4,)
        wbad = xr.DataArray(np.ones((2, 6)), dims=("level", "latitude"))
        with pytest.raises(ValueError, match="t2m"):
            mxr.rmse(fd[["t2m"]], od[["t2m"]], weights=wbad)

    def test_pearson_wind_categorical(self, rng):
        import metvane.xr_api as mxr
        f, o = self._da(rng), self._da(rng)
        p = mxr.pearson_correlation(f, o, preserve_dims="lead_time")
        np.testing.assert_allclose(p.values, metvane.pearson_correlation(f.values, o.values, axis=(1, 2)))
        wv = mxr.wind_vector_rmse(f, f, o, o)
        assert float(wv) == pytest.approx(float(metvane.wind_vector_rmse(f.values, f.values, o.values, o.values)))
        s = mxr.categorical_scores(f, o, [0.0, 1.0], op=">", preserve_dims="lead_time")
        ref = ContingencyTable(f.values, o.values, [0.0, 1.0], op=">", axis=(1, 2))
        np.testing.assert_allclose(s["csi"].values, ref.csi())
        np.testing.assert_allclose(s["hss"].values, ref.hss())
        c = mxr.contingency_table(f, o, [1.0], preserve_dims="lead_time")
        assert c.tp.dtype == np.int64 and c.tp.dims == ("threshold", "lead_time")


# ============================================================================ P1-17
@requires_xarray
def test_p1_17_accumulator_rejects_xarray(rng):
    import xarray as xr
    a = xr.DataArray(rng.standard_normal((3, 4)))
    with pytest.raises(TypeError, match="xr"):
        ContinuousAccumulator(["rmse"]).update(a, a)
    with pytest.raises(TypeError):
        ContingencyAccumulator([0.0]).update(a, a)


# ============================================================================ P2-1
class TestP2_1_ExactCounts:
    def test_int64_counts_beyond_2_24(self):
        n = 2 ** 24 + 10
        x = np.ones(n, dtype=np.float32)
        t = ContingencyTable(x, x, [0.5])
        assert int(t.tp[0]) == n

    @requires_torch
    def test_torch_counts_beyond_2_24(self):
        import torch
        n = 2 ** 24 + 10
        x = torch.ones(n)
        assert int(ContingencyTable(x, x, [0.5]).tp[0]) == n
        ca = ContingencyAccumulator([0.5])
        ca.update(x, x)
        ca.update(x[:1], x[:1])
        assert int(ca.as_contingency_table().tp[0]) == n + 1

    @requires_torch
    def test_continuous_state_float64(self):
        import torch
        acc = ContinuousAccumulator(["mse"])
        acc.update(torch.ones(10), torch.zeros(10))
        assert acc.stats["count"].dtype == torch.float64


# ============================================================================ P2-3 / P2-4 / P3-2
class TestCategoricalInputs:
    def test_threshold_zero_warns(self):
        x = np.array([0.0, 1.0])
        with pytest.warns(UserWarning, match="op='>'"):
            ContingencyTable(x, x, [0.0])

    def test_threshold_and_op_validation(self, rng):
        f = rng.random(20)
        with pytest.raises(TypeError, match="单个阈值"):
            metvane.csi(f, f, [0.1, 1.0])
        assert ContingencyTable(f, f, 0.1).csi().shape == (1,)
        with pytest.raises(ValueError, match="op"):
            ContingencyTable(f, f, [0.1], op="ge")
        acc = ContinuousAccumulator("rmse")
        assert acc.metrics == ("rmse",)

    def test_summary_aliases(self, rng):
        f = rng.random(20)
        t = ContingencyTable(f, f, [0.5])
        assert set(t.summary(["CSI", "TS", "frequency_bias"])) == {"CSI", "TS", "frequency_bias"}
        assert list(t.summary("csi")) == ["csi"]
        assert t.summary([]) == {}
        with pytest.raises(ValueError, match="Unknown"):
            t.summary(["rmse"])

    def test_naming(self, rng):
        f, o = rng.random((2, 50))
        t = ContingencyTable(f, o, [0.5])
        with pytest.warns(FutureWarning, match="POFD"):
            np.testing.assert_allclose(t.false_alarm_rate(), t.pofd())
        np.testing.assert_allclose(metvane.frequency_bias(f, o, 0.5), t.bias_score()[0])
        np.testing.assert_allclose(metvane.pofd(f, o, 0.5), t.pofd()[0])
        np.testing.assert_allclose(metvane.f1(f, o, 0.5), t.f1()[0])
        np.testing.assert_allclose(t.accuracy(), t.pc())


# ============================================================================ P2-5 / P2-6
class TestAxes:
    def test_axis_forms(self, rng):
        f, o = rng.standard_normal((2, 3, 4, 5))
        ref = metvane.rmse(f, o, axis=(1, 2))
        np.testing.assert_allclose(metvane.rmse(f, o, axis=[1, 2]), ref)
        np.testing.assert_allclose(metvane.rmse(f, o, axis=(-1, 1)), ref)
        with pytest.raises(Exception, match="超出"):
            metvane.rmse(f, o, axis=3)
        with pytest.raises(ValueError, match="重复"):
            metvane.rmse(f, o, axis=(1, -2))
        with pytest.raises(Exception, match="超出"):
            ContinuousAccumulator(["rmse"], preserve_axes=[3]).update(f, o)

    @requires_torch
    def test_numpy_int_axis_torch(self, rng):
        import torch
        f = torch.from_numpy(rng.standard_normal((3, 4)))
        np.testing.assert_allclose(metvane.rmse(f, f * 0, axis=np.int64(0)).numpy(),
                                   metvane.rmse(f.numpy(), f.numpy() * 0, axis=0))

    def test_accumulator_axis_equals_functional(self, rng):
        f, o = rng.standard_normal((2, 6, 7))
        acc = ContinuousAccumulator(["rmse"], axis=[0])
        acc.update(f, o)
        np.testing.assert_allclose(acc.compute()["rmse"], metvane.rmse(f, o, axis=[0]))
        with pytest.raises(ValueError, match="互斥"):
            ContinuousAccumulator(["rmse"], axis=0, preserve_axes=[1])


# ============================================================================ P2-7
def test_p2_7_multiply_mode_warnings(rng):
    f, o = rng.standard_normal((2, 10, 4))
    w = np.linspace(0.1, 2, 10)[:, None]
    m = np.zeros((10, 4), bool)
    m[:5] = True
    with pytest.warns(UserWarning, match="weight_mode='mean'"):
        metvane.rmse(f, o, weights=w, weight_mode="multiply", mask=m)
    with pytest.warns(UserWarning, match="均值不为 1"):
        metvane.rmse(f, o, weights=w, weight_mode="multiply")


# ============================================================================ P2-10 / P2-11 / P2-12
class TestProbabilistic:
    def test_crps_closed_forms(self):
        y = np.array([0.3])
        np.testing.assert_allclose(metvane.crps_ensemble(np.array([[1.0]]), y), 0.7)
        ens = np.array([[1.0], [-1.0]])
        mean_abs = (0.7 + 1.3) / 2
        np.testing.assert_allclose(metvane.crps_ensemble(ens, y), mean_abs - 2 / 4)
        np.testing.assert_allclose(metvane.crps_ensemble(ens, y, fair=True), mean_abs - 2 / 2)

    def test_crps_vs_naive(self, rng):
        ens = rng.standard_normal((7, 3, 4, 5))
        y = rng.standard_normal((3, 4, 5))
        pair = np.abs(ens[:, None] - ens[None]).sum((0, 1))
        naive = np.abs(ens - y).mean(0) - pair / (2 * 49)
        np.testing.assert_allclose(metvane.crps_ensemble(ens, y, axis=()), naive, rtol=1e-12)
        np.testing.assert_allclose(metvane.crps_ensemble(ens, y, axis=(2, 3)), naive.mean((1, 2)), rtol=1e-12)
        with pytest.raises(ValueError, match="成员轴"):
            metvane.crps_ensemble(ens, y, axis=(0,))
        fair = np.abs(ens - y).mean(0) - pair / (2 * 7 * 6)
        np.testing.assert_allclose(metvane.crps_ensemble(ens, y, fair=True), fair.mean(), rtol=1e-12)

    @requires_torch
    def test_crps_fss_torch_return_type(self, rng):
        import torch
        ens = torch.from_numpy(rng.standard_normal((5, 10)))
        y = torch.from_numpy(rng.standard_normal(10))
        out = metvane.crps_ensemble(ens, y)
        assert isinstance(out, torch.Tensor)
        np.testing.assert_allclose(out.numpy(), metvane.crps_ensemble(ens.numpy(), y.numpy()))
        f = torch.from_numpy(rng.gamma(0.5, 3, (20, 20)))
        o = torch.from_numpy(rng.gamma(0.5, 3, (20, 20)))
        s = metvane.fss(f, o, 1.0, window_size=5)
        assert isinstance(s, torch.Tensor)
        assert float(s) == pytest.approx(float(metvane.fss(f.numpy(), o.numpy(), 1.0, window_size=5)))

    def test_brier_validation(self, rng):
        p = rng.random(20)
        o = (rng.random(20) > 0.5).astype(float)
        np.testing.assert_allclose(metvane.brier_score(p, o), np.mean((p - o) ** 2))
        with pytest.raises(ValueError, match="百分比"):
            metvane.brier_score(p * 100, o)
        with pytest.raises(ValueError, match="threshold"):
            metvane.brier_score(p, o * 43.5)
        mm = o * 10
        np.testing.assert_allclose(metvane.brier_score(p, mm, threshold=5), np.mean((p - o) ** 2))
        o2 = o.copy()
        o2[0] = np.nan
        np.testing.assert_allclose(metvane.brier_score(p, o2), np.mean((p[1:] - o[1:]) ** 2))


# ============================================================================ P2-13 / P2-14
class TestAccumulatorState:
    def test_atomic_update(self, rng):
        f, o = rng.standard_normal((2, 4, 5))
        acc = ContinuousAccumulator(["rmse", "acc"], climatology=0.0)
        acc.update(f, o)
        before = acc.stats
        with pytest.raises(ValueError):
            acc.update(f, o, weights=np.ones((3, 5)))
        after = acc.stats
        for k in before:
            np.testing.assert_array_equal(before[k], after[k])

    def test_merge_and_state_dict(self, rng):
        f, o = rng.standard_normal((2, 8, 3, 4, 5))
        a, b = ContinuousAccumulator(["rmse"], preserve_axes=[1]), ContinuousAccumulator(["rmse"], preserve_axes=[1])
        a.update(f[:4], o[:4])
        b.update(f[4:], o[4:])
        a.merge(b)
        np.testing.assert_allclose(a.compute()["rmse"], metvane.rmse(f, o, axis=(0, 2, 3)))
        c = ContinuousAccumulator(["rmse"], preserve_axes=[1])
        c.load_state_dict(a.state_dict())
        np.testing.assert_allclose(c.compute()["rmse"], a.compute()["rmse"])
        with pytest.raises(ValueError, match="配置"):
            a.merge(ContinuousAccumulator(["mae"], preserve_axes=[1]))
        ca, cb = ContingencyAccumulator([0.0]), ContingencyAccumulator([0.0])
        ca.update(f[:4], o[:4])
        cb.update(f[4:], o[4:])
        ca.merge(cb)
        ref = ContingencyTable(f, o, [0.0])
        assert int(ca.as_contingency_table().tp[0]) == int(ref.tp[0])
        with pytest.raises(ValueError):
            ca.merge(ContingencyAccumulator([1.0]))


# ============================================================================ P2-20 / P2-21 / P2-22
@requires_torch
class TestBackendMisc:
    def test_requires_grad_and_negative_strides(self, rng):
        import torch
        f = rng.standard_normal((4, 5))
        t = torch.from_numpy(f.copy()).requires_grad_(True)
        out = metvane.rmse(f, t)
        assert isinstance(out, torch.Tensor)
        np.testing.assert_allclose(float(metvane.rmse(f[:, ::-1], torch.from_numpy(f))),
                                   float(metvane.rmse(f[:, ::-1], f)))

    def test_detect_backend_order(self):
        import torch
        import xarray as xr
        t, x = torch.zeros(1), xr.DataArray([0.0])
        assert metvane.detect_backend(t, x) == metvane.detect_backend(x, t) == metvane.Backend.TORCH

    def test_mismatched_devices_raise(self):
        import torch
        a = torch.zeros(3)
        b = torch.zeros(3, device="meta")
        with pytest.raises(ValueError, match="设备"):
            metvane.rmse(a, b)


# ============================================================================ P2-24 golden values
class TestGolden:
    def test_finley(self):
        t = ContingencyTable.from_counts(*(np.array([v]) for v in (28, 72, 23, 2680)))
        np.testing.assert_allclose(t.csi(), 28 / 123)
        np.testing.assert_allclose(t.pod(), 28 / 51)
        np.testing.assert_allclose(t.far(), 0.72)
        np.testing.assert_allclose(t.pofd(), 72 / 2752)
        np.testing.assert_allclose(t.bias_score(), 100 / 51)
        np.testing.assert_allclose(t.pc(), 2708 / 2803)
        np.testing.assert_allclose(t.hss(), 146768 / 413053)
        np.testing.assert_allclose(t.ets(), 73384 / 339669)
        np.testing.assert_allclose(t.hss(), 0.3553, atol=1e-4)
        np.testing.assert_allclose(t.ets(), 0.2161, atol=1e-4)

    def test_fss_3x3_hand(self):
        f = np.zeros((3, 3))
        o = np.zeros((3, 3))
        f[1, 1] = 1
        o[0, 0] = 1
        assert float(metvane.fss(f, o, 0.5, window_size=1)) == pytest.approx(0.0)
        # window 3: F_f = 1/9 everywhere; F_o = 1/9 on the 4 cells around (0,0), 0 elsewhere
        ff = np.full((3, 3), 1 / 9)
        fo = np.zeros((3, 3))
        fo[:2, :2] = 1 / 9
        ref = 1 - np.sum((ff - fo) ** 2) / (np.sum(ff ** 2) + np.sum(fo ** 2))
        assert float(metvane.fss(f, o, 0.5, window_size=3)) == pytest.approx(ref)


# ============================================================================ P2-16 conversion cache
@requires_torch
def test_weight_conversion_cache(rng):
    import torch
    from metvane.core import prepare as P
    P.clear_conversion_cache()
    f = torch.from_numpy(rng.standard_normal((3, 64, 32)))
    w = np.linspace(0.5, 1.5, 64)[:, None] * np.ones((1, 32))
    a = metvane.rmse(f, f * 0, axis=(1, 2), weights=w)
    assert len(P._TORCH_CACHE) == 1
    t1 = next(iter(P._TORCH_CACHE.values()))[2]
    b = metvane.rmse(f, f * 0, axis=(1, 2), weights=w)
    assert next(iter(P._TORCH_CACHE.values()))[2] is t1
    torch.testing.assert_close(a, b)
    w2 = w * 2                                     # new array -> new entry, correct result
    np.testing.assert_allclose(metvane.rmse(f, f * 0, axis=(1, 2), weights=w2).numpy(), a.numpy())
    assert len(P._TORCH_CACHE) == 2
    P.clear_conversion_cache()
    assert not P._TORCH_CACHE
