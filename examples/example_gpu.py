"""Example: GPU evaluation with PyTorch tensors.

The same functions accept torch tensors; results stay on the input device.  numpy weights /
climatology are converted automatically.  Accumulator state is float64 / int64 on the device.
"""

import numpy as np

import metvane


def main(device=None):
    try:
        import torch
    except ImportError:
        print("PyTorch not available — example skipped.")
        return None
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)
    n_init, n_lead, n_lat, n_lon = 4, 10, 91, 180
    lat = np.linspace(90, -90, n_lat)
    w = metvane.latitude_weights(lat)[:, None]      # numpy weights, converted to the device

    f_np = rng.standard_normal((n_init, n_lead, n_lat, n_lon)).astype(np.float32)
    o_np = rng.standard_normal(f_np.shape).astype(np.float32)
    f, o = torch.from_numpy(f_np).to(device), torch.from_numpy(o_np).to(device)

    rmse = metvane.rmse(f, o, axis=(0, 2, 3), weights=w)
    assert rmse.device.type == torch.device(device).type
    np.testing.assert_allclose(rmse.cpu().numpy(), metvane.rmse(f_np, o_np, axis=(0, 2, 3), weights=w), rtol=1e-5)
    print(f"RMSE on {rmse.device}: {rmse[:3].cpu().numpy()}")

    radar_f = torch.rand(n_init, n_lead, 64, 64, device=device) * 60
    radar_o = torch.rand(n_init, n_lead, 64, 64, device=device) * 60
    table = metvane.ContingencyTable(radar_f, radar_o, [20, 35], axis=(0, 2, 3))
    assert table.tp.dtype == torch.int64
    print("CSI (2 thresholds × lead) on", table.csi().device)

    acc = metvane.ContinuousAccumulator(["rmse", "mae"], preserve_axes=[1], weights=w,
                                        backend="torch", device=device)
    acc.update(f_np[:3], o_np[:3])                  # numpy chunks converted to the device
    acc.update(f_np[3:], o_np[3:])
    res = acc.compute()
    np.testing.assert_allclose(res["rmse"].cpu().numpy(), rmse.cpu().numpy(), rtol=1e-5)
    print("GPU accumulator == functional ✓")
    return res


if __name__ == "__main__":
    main()
