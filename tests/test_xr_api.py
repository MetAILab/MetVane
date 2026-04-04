"""Tests for xr_api — xarray convenience layer."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import requires_xarray


@requires_xarray
class TestXrApi:
    def test_rmse(self):
        import xarray as xr
        import metvane.xr_api as mxr

        rng = np.random.default_rng(42)
        fcst = xr.DataArray(
            rng.standard_normal((10, 32, 64)).astype(np.float32),
            dims=["lead_time", "lat", "lon"],
        )
        obs = xr.DataArray(
            rng.standard_normal((10, 32, 64)).astype(np.float32),
            dims=["lead_time", "lat", "lon"],
        )
        result = mxr.rmse(fcst, obs, preserve_dims="lead_time")
        assert result.dims == ("lead_time",)
        assert result.shape == (10,)

    def test_rmse_reduce_all(self):
        import xarray as xr
        import metvane.xr_api as mxr

        rng = np.random.default_rng(42)
        fcst = xr.DataArray(rng.standard_normal((10, 32)).astype(np.float32),
                            dims=["time", "space"])
        obs = xr.DataArray(rng.standard_normal((10, 32)).astype(np.float32),
                           dims=["time", "space"])
        result = mxr.rmse(fcst, obs)
        assert result.ndim == 0

    def test_mae(self):
        import xarray as xr
        import metvane.xr_api as mxr

        fcst = xr.DataArray([1.0, -2.0, 3.0], dims=["x"])
        obs = xr.DataArray([0.0, 0.0, 0.0], dims=["x"])
        result = mxr.mae(fcst, obs)
        np.testing.assert_allclose(float(result), 2.0)

    def test_dataset(self):
        import xarray as xr
        import metvane.xr_api as mxr

        rng = np.random.default_rng(42)
        shape = (10, 32)
        fcst_ds = xr.Dataset({
            "t2m": xr.DataArray(rng.standard_normal(shape).astype(np.float32),
                                dims=["time", "space"]),
            "u10": xr.DataArray(rng.standard_normal(shape).astype(np.float32),
                                dims=["time", "space"]),
        })
        obs_ds = xr.Dataset({
            "t2m": xr.DataArray(rng.standard_normal(shape).astype(np.float32),
                                dims=["time", "space"]),
            "u10": xr.DataArray(rng.standard_normal(shape).astype(np.float32),
                                dims=["time", "space"]),
        })
        result = mxr.rmse(fcst_ds, obs_ds, preserve_dims="time")
        assert isinstance(result, xr.Dataset)
        assert "t2m" in result.data_vars
        assert "u10" in result.data_vars

    def test_acc(self):
        import xarray as xr
        import metvane.xr_api as mxr

        rng = np.random.default_rng(42)
        clim = xr.DataArray(np.zeros(100, dtype=np.float32), dims=["space"])
        fcst_arr = rng.standard_normal(100).astype(np.float32)
        obs_arr = fcst_arr + 0.1 * rng.standard_normal(100).astype(np.float32)
        fcst = xr.DataArray(fcst_arr, dims=["space"])
        obs = xr.DataArray(obs_arr, dims=["space"])
        result = mxr.acc(fcst, obs, clim)
        assert float(result) > 0.9

    def test_error_both_dims(self):
        import xarray as xr
        import metvane.xr_api as mxr

        fcst = xr.DataArray([1.0], dims=["x"])
        obs = xr.DataArray([2.0], dims=["x"])
        with pytest.raises(ValueError, match="only one"):
            mxr.rmse(fcst, obs, reduce_dims="x", preserve_dims="x")
