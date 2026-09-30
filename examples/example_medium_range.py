"""Example: medium-range forecast evaluation.

Data layout ``(n_init, n_lead, lat, lon)``: 8 initialisations × 20 lead times (6-hourly) on a
1° global grid.  Shows latitude-weighted RMSE / ACC per lead time (WeatherBench-2 style),
regional scores via a weighted mask, wind-vector RMSE, and chunked accumulation along the
init axis that reproduces the one-shot functional result.
"""

import numpy as np

import metvane


def main():
    rng = np.random.default_rng(42)
    n_init, n_lead, n_lat, n_lon = 8, 20, 181, 360
    lat = np.linspace(90, -90, n_lat)              # ERA5 order (north -> south)
    lon = np.arange(n_lon, dtype=np.float64)      # 0..359

    clim = rng.standard_normal((n_lat, n_lon)).astype(np.float32) * 5
    growth = np.linspace(0.2, 2.0, n_lead, dtype=np.float32)[None, :, None, None]
    obs = clim + rng.standard_normal((n_init, n_lead, n_lat, n_lon)).astype(np.float32)
    fcst = obs + growth * rng.standard_normal(obs.shape).astype(np.float32)

    # 1. latitude weights: (nlat, 1) broadcasts along latitude (a bare 1-D array is rejected)
    w = metvane.latitude_weights(lat)[:, None]

    # 2. RMSE per lead: sqrt of the init-averaged weighted MSE (WeatherBench convention)
    rmse = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w)
    print("RMSE per lead:", np.round(rmse[::5], 4))

    # 3. ACC per lead: spatial ACC per init, then averaged over inits (WB2 / ECMWF convention)
    acc = metvane.acc(fcst, obs, clim, axis=(2, 3), mean_over=0, weights=w)
    acc_pooled = metvane.acc(fcst, obs, clim, axis=(0, 2, 3), weights=w)
    print("ACC per lead (per-init mean):", np.round(acc[::5], 4))
    print("ACC per lead (pooled)       :", np.round(acc_pooled[::5], 4))

    # 4. regional score: pass the region as mask= together with the latitude weights
    nh = metvane.region_mask(lat, lon, "NH")
    rmse_nh = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w, mask=nh)
    print("NH RMSE lead 0:", round(float(rmse_nh[0]), 4))

    # 5. wind vector RMSE
    u_o, v_o = rng.standard_normal((2, n_init, n_lead, n_lat, n_lon)).astype(np.float32)
    u_f = u_o + 0.3 * rng.standard_normal(u_o.shape).astype(np.float32)
    v_f = v_o + 0.3 * rng.standard_normal(v_o.shape).astype(np.float32)
    wv = metvane.wind_vector_rmse(u_f, v_f, u_o, v_o, axis=(0, 2, 3), weights=w)
    print("Wind vector RMSE lead 0:", round(float(wv[0]), 4))

    # 6. accumulator: split chunks ONLY along a reduced axis (init), keep lead (axis 1)
    accum = metvane.ContinuousAccumulator(["rmse", "mae", "bias", "acc", "acc_mean"], preserve_axes=[1],
                                          weights=w, climatology=clim, spatial_axes=(2, 3))
    for start in range(0, n_init, 3):              # uneven chunks 3/3/2
        accum.update(fcst[start:start + 3], obs[start:start + 3])
    res = accum.compute()
    assert res["rmse"].shape == (n_lead,)
    np.testing.assert_allclose(res["rmse"], rmse, rtol=1e-5)
    np.testing.assert_allclose(res["mae"], metvane.mae(fcst, obs, axis=(0, 2, 3), weights=w), rtol=1e-5)
    np.testing.assert_allclose(res["acc"], acc_pooled, rtol=1e-5)
    np.testing.assert_allclose(res["acc_mean"], acc, rtol=1e-5)
    print("Accumulator == functional ✓")
    return {"rmse": rmse, "acc": acc, "accumulator": res}


if __name__ == "__main__":
    main()
