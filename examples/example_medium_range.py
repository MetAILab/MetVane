"""Example: Medium-range weather forecast evaluation.

Demonstrates how to evaluate a 10-day (40-step, 6h interval) global forecast
for variables like t2m, z500, u10, v10 using MetVane.
"""

import numpy as np

import metvane


def main():
    # ---------------------------------------------------------------
    # 1. Simulate data: (lead_time=40, lat=181, lon=360)
    # ---------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_lead, n_lat, n_lon = 40, 181, 360
    lat = np.linspace(-90, 90, n_lat, dtype=np.float32)
    lon = np.linspace(0, 359, n_lon, dtype=np.float32)

    obs = rng.standard_normal((n_lead, n_lat, n_lon)).astype(np.float32)
    fcst = obs + 0.5 * rng.standard_normal((n_lead, n_lat, n_lon)).astype(np.float32)
    clim = np.zeros((n_lat, n_lon), dtype=np.float32)

    # ---------------------------------------------------------------
    # 2. Latitude weights
    # ---------------------------------------------------------------
    w = metvane.latitude_weights(lat)
    w_3d = metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)

    # ---------------------------------------------------------------
    # 3. Per-lead-time RMSE (weighted)
    # ---------------------------------------------------------------
    rmse_per_lt = metvane.rmse(fcst, obs, axis=(1, 2), weights=w_3d)
    print("RMSE per lead time:")
    for i in range(0, n_lead, 8):
        print(f"  step {i:3d} ({(i + 1) * 6:3d}h): {rmse_per_lt[i]:.4f}")

    # ---------------------------------------------------------------
    # 4. Overall RMSE
    # ---------------------------------------------------------------
    rmse_all = metvane.rmse(fcst, obs, weights=w_3d)
    print(f"\nOverall weighted RMSE: {float(rmse_all):.4f}")

    # ---------------------------------------------------------------
    # 5. ACC per lead time
    # ---------------------------------------------------------------
    clim_3d = np.broadcast_to(clim, fcst.shape)
    w_acc = np.broadcast_to(w.reshape(1, -1, 1), fcst.shape)
    acc_per_lt = metvane.acc(fcst, obs, clim_3d, axis=(1, 2), weights=w_acc)
    print("\nACC per lead time:")
    for i in range(0, n_lead, 8):
        print(f"  step {i:3d} ({(i + 1) * 6:3d}h): {acc_per_lt[i]:.4f}")

    # ---------------------------------------------------------------
    # 6. Wind vector RMSE
    # ---------------------------------------------------------------
    u_fcst = rng.standard_normal((n_lead, n_lat, n_lon)).astype(np.float32)
    v_fcst = rng.standard_normal((n_lead, n_lat, n_lon)).astype(np.float32)
    u_obs = u_fcst + 0.3 * rng.standard_normal(u_fcst.shape).astype(np.float32)
    v_obs = v_fcst + 0.3 * rng.standard_normal(v_fcst.shape).astype(np.float32)

    wv_rmse = metvane.wind_vector_rmse(
        u_fcst, v_fcst, u_obs, v_obs, axis=(1, 2), weights=w_3d,
    )
    print("\nWind vector RMSE per lead time:")
    for i in range(0, n_lead, 8):
        print(f"  step {i:3d} ({(i + 1) * 6:3d}h): {wv_rmse[i]:.4f}")

    # ---------------------------------------------------------------
    # 7. Region-specific evaluation
    # ---------------------------------------------------------------
    mask_nh = metvane.region_mask(lat, lon, "NH")
    fcst_nh = fcst[:, mask_nh]
    obs_nh = obs[:, mask_nh]
    rmse_nh = metvane.rmse(fcst_nh, obs_nh, axis=1)
    print(f"\nNH RMSE (step 0): {rmse_nh[0]:.4f}")

    # ---------------------------------------------------------------
    # 8. Accumulator mode (chunked evaluation)
    # ---------------------------------------------------------------
    acc_obj = metvane.ContinuousAccumulator(
        ["rmse", "mae", "bias"], preserve_axes=[0],
    )
    chunk_size = 10
    for start in range(0, n_lead, chunk_size):
        end = min(start + chunk_size, n_lead)
        acc_obj.update(fcst[start:end], obs[start:end])

    result = acc_obj.compute()
    print(f"\nAccumulator RMSE shape: {result['rmse'].shape}")
    print(f"Accumulator MAE (step 0): {result['mae'][0]:.4f}")
    print(f"Accumulator Bias (step 0): {result['bias'][0]:.4f}")

    print("\n✓ Medium-range evaluation complete.")


if __name__ == "__main__":
    main()
