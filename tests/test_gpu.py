"""CUDA consistency tests (numpy reference == torch CUDA).  Run on a GPU node:
``sbatch scripts/run_gpu_tests.sh`` (skipped when CUDA is unavailable)."""

from __future__ import annotations

import numpy as np
import pytest

import metvane

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")]


def _cuda(*xs):
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).cuda() for x in xs)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    f = rng.gamma(0.6, 4, (6, 5, 40, 48))
    o = f + rng.normal(0, 1, f.shape)
    o[:, :, :5, :5] = np.nan
    w = metvane.latitude_weights(np.linspace(80, -80, 40))[:, None]
    return f, o, w


def test_continuous(data):
    f, o, w = data
    ft, ot = _cuda(f, o)
    for fn in (metvane.rmse, metvane.mae, metvane.bias):
        out = fn(ft, ot, axis=(0, 2, 3), weights=w)
        assert out.is_cuda
        np.testing.assert_allclose(out.cpu().numpy(), fn(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
    a = metvane.acc(ft, ot, 1.0, axis=(2, 3), mean_over=0, weights=w)
    np.testing.assert_allclose(a.cpu().numpy(), metvane.acc(f, o, 1.0, axis=(2, 3), mean_over=0, weights=w),
                               rtol=1e-10)
    full = metvane.rmse(ft, ot, axis=(), weights=w)
    assert tuple(full.shape) == f.shape


def test_categorical_and_fss(data):
    f, o, _ = data
    ft, ot = _cuda(f, o)
    t = metvane.ContingencyTable(ft, ot, [0.1, 1, 5], axis=(0, 2, 3))
    r = metvane.ContingencyTable(f, o, [0.1, 1, 5], axis=(0, 2, 3))
    assert t.tp.is_cuda and t.tp.dtype == torch.int64
    np.testing.assert_array_equal(t.tp.cpu().numpy(), r.tp)
    np.testing.assert_allclose(t.hss().cpu().numpy(), r.hss())
    c = metvane.fss_components(ft, ot, [1.0, 5.0], [1, 5, 9], axis=(0,))
    cr = metvane.fss_components(f, o, [1.0, 5.0], [1, 5, 9], axis=(0,))
    np.testing.assert_allclose(c["num"].cpu().numpy(), cr["num"], rtol=1e-12)
    np.testing.assert_allclose(c["den"].cpu().numpy(), cr["den"], rtol=1e-12)


def test_accumulators(data):
    f, o, w = data
    acc = metvane.ContinuousAccumulator(["rmse", "acc"], preserve_axes=[1], weights=w, climatology=0.0,
                                        backend="torch", device="cuda")
    ca = metvane.ContingencyAccumulator([1.0], preserve_axes=[1], backend="torch", device="cuda")
    for s in (slice(0, 4), slice(4, 6)):
        acc.update(f[s], o[s])
        ca.update(f[s], o[s])
    out = acc.compute()
    assert out["rmse"].is_cuda and acc.stats["count"].dtype == torch.float64
    np.testing.assert_allclose(out["rmse"].cpu().numpy(), metvane.rmse(f, o, axis=(0, 2, 3), weights=w), rtol=1e-12)
    np.testing.assert_allclose(ca.compute()["csi"].cpu().numpy(),
                               metvane.ContingencyTable(f, o, [1.0], axis=(0, 2, 3)).csi())


def test_crps_region(data):
    rng = np.random.default_rng(1)
    ens = rng.standard_normal((10, 3, 20))
    y = rng.standard_normal((3, 20))
    et, yt = _cuda(ens, y)
    np.testing.assert_allclose(metvane.crps_ensemble(et, yt, fair=True).cpu().numpy(),
                               metvane.crps_ensemble(ens, y, fair=True), rtol=1e-12)
    lat, lon = _cuda(np.linspace(90, -90, 19), np.arange(0, 360, 10.0))
    m = metvane.region_mask(lat, lon, "europe")
    assert m.is_cuda and m.dtype == torch.bool
