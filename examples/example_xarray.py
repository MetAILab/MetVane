"""Example: xarray layer (named dims, coordinate alignment, dask).

Observations use the opposite latitude order to the forecast — the xarray layer aligns them by
coordinate.  Latitude weights are a DataArray on the latitude dimension.
"""

import numpy as np


def main():
    try:
        import xarray as xr
    except ImportError:
        print("xarray not available — example skipped (pip install 'metvane[xarray]').")
        return None
    import metvane
    import metvane.xr_api as mxr

    rng = np.random.default_rng(42)
    time = np.arange(6)
    lead = np.arange(1, 11)
    lat = np.linspace(-90, 90, 73)
    lon = np.arange(0, 360, 5.0)
    shape = (6, 10, 73, 72)
    dims = ("time", "lead_time", "latitude", "longitude")
    coords = {"time": time, "lead_time": lead, "latitude": lat, "longitude": lon}

    clim = xr.DataArray(rng.standard_normal(shape[2:]) * 5, dims=dims[2:], coords={"latitude": lat, "longitude": lon})
    obs = xr.DataArray(rng.standard_normal(shape), dims=dims, coords=coords, attrs={"units": "K"}) + clim
    fcst = obs + xr.DataArray(rng.standard_normal(shape) * np.linspace(0.2, 2, 10)[None, :, None, None],
                              dims=dims, coords=coords)
    obs = obs.sortby("latitude", ascending=False)   # different latitude order: aligned by coordinate

    w = metvane.latitude_weights(fcst.latitude)     # DataArray on 'latitude'

    rmse = mxr.rmse(fcst, obs, preserve_dims="lead_time", weights=w)
    print(rmse.name, dict(rmse.attrs), np.round(rmse.values[:3], 4))
    ref = metvane.rmse(fcst.values, obs.sortby("latitude").values, axis=(0, 2, 3), weights=w.values[:, None])
    np.testing.assert_allclose(rmse.values, ref, rtol=1e-12)

    acc = mxr.acc(fcst, obs, clim, reduce_dims=["latitude", "longitude"], mean_over="time", weights=w)
    print("ACC (per-time spatial ACC averaged over time):", np.round(acc.values[:3], 4))

    ds_f = xr.Dataset({"t2m": fcst, "z500": fcst * 100})
    ds_o = xr.Dataset({"t2m": obs, "z500": obs * 100})
    print("Dataset RMSE:", list(mxr.rmse(ds_f, ds_o, preserve_dims="lead_time", weights=w).data_vars))

    scores = mxr.categorical_scores(fcst, obs, [1.0, 5.0], preserve_dims="lead_time", metrics=["csi", "hss"])
    print("CSI dims:", scores["csi"].dims)

    try:
        import dask  # noqa: F401
        lazy = mxr.rmse(fcst.chunk({"time": 2}), obs.chunk({"time": 2}), preserve_dims="lead_time", weights=w)
        assert lazy.chunks is not None
        np.testing.assert_allclose(lazy.compute().values, rmse.values, rtol=1e-12)
        print("dask: lazy result == eager ✓")
    except ImportError:
        pass
    return rmse


if __name__ == "__main__":
    main()
