"""Example: xarray convenience API.

Shows dimension-aware evaluation using named dimensions
(reduce_dims / preserve_dims) with xr.DataArray and xr.Dataset.
"""

import numpy as np

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False


def main():
    if not HAS_XR:
        print("xarray not available. Install: pip install xarray netCDF4")
        return

    import metvane.xr_api as mxr
    import metvane

    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------
    # 1. Create labeled data
    # ---------------------------------------------------------------
    lead_time = np.arange(1, 11)
    lat = np.linspace(-90, 90, 64, dtype=np.float32)
    lon = np.linspace(0, 359, 128, dtype=np.float32)

    fcst = xr.DataArray(
        rng.standard_normal((10, 64, 128)).astype(np.float32),
        dims=["lead_time", "lat", "lon"],
        coords={"lead_time": lead_time, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.standard_normal((10, 64, 128)).astype(np.float32),
        dims=["lead_time", "lat", "lon"],
        coords={"lead_time": lead_time, "lat": lat, "lon": lon},
    )

    # ---------------------------------------------------------------
    # 2. RMSE per lead time (preserve_dims)
    # ---------------------------------------------------------------
    rmse_lt = mxr.rmse(fcst, obs, preserve_dims="lead_time")
    print(f"RMSE per lead_time:  dims={rmse_lt.dims}, shape={rmse_lt.shape}")
    print(rmse_lt.values)

    # ---------------------------------------------------------------
    # 3. RMSE overall (reduce all)
    # ---------------------------------------------------------------
    rmse_all = mxr.rmse(fcst, obs)
    print(f"\nOverall RMSE: {float(rmse_all):.4f}")

    # ---------------------------------------------------------------
    # 4. Weighted by latitude
    # ---------------------------------------------------------------
    w = metvane.latitude_weights(lat)
    w_da = xr.DataArray(w, dims=["lat"], coords={"lat": lat})
    rmse_weighted = mxr.rmse(fcst, obs, preserve_dims="lead_time", weights=w_da)
    print(f"\nWeighted RMSE per lead_time:\n{rmse_weighted.values}")

    # ---------------------------------------------------------------
    # 5. Dataset (multi-variable)
    # ---------------------------------------------------------------
    fcst_ds = xr.Dataset({
        "t2m": fcst,
        "z500": fcst * 1000,
    })
    obs_ds = xr.Dataset({
        "t2m": obs,
        "z500": obs * 1000,
    })
    rmse_ds = mxr.rmse(fcst_ds, obs_ds, preserve_dims="lead_time")
    print(f"\nDataset RMSE vars: {list(rmse_ds.data_vars)}")
    for var in rmse_ds.data_vars:
        print(f"  {var}: shape={rmse_ds[var].shape}")

    # ---------------------------------------------------------------
    # 6. ACC
    # ---------------------------------------------------------------
    clim = xr.DataArray(
        np.zeros((64, 128), dtype=np.float32),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )
    acc_lt = mxr.acc(fcst, obs, clim, preserve_dims="lead_time")
    print(f"\nACC per lead_time:\n{acc_lt.values}")

    print("\n✓ xarray evaluation example complete.")


if __name__ == "__main__":
    main()
