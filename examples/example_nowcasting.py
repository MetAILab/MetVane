"""Example: Radar nowcasting evaluation.

Demonstrates categorical metrics (CSI, POD, FAR) for radar reflectivity
nowcasting with multiple dBZ thresholds and per-lead-time analysis.
"""

import numpy as np

import metvane
from metvane.categorical import ContingencyTable


def main():
    # ---------------------------------------------------------------
    # 1. Simulate radar data: (n_frames=20, H=256, W=256) dBZ
    # ---------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_frames, h, w = 20, 256, 256

    obs = rng.uniform(0, 50, (n_frames, h, w)).astype(np.float32)
    obs[obs < 10] = 0
    fcst = obs + rng.uniform(-5, 5, (n_frames, h, w)).astype(np.float32)
    fcst = np.clip(fcst, 0, 65)

    thresholds = [20.0, 35.0, 40.0]

    # ---------------------------------------------------------------
    # 2. Overall evaluation (all frames together)
    # ---------------------------------------------------------------
    table_all = ContingencyTable(fcst, obs, thresholds=thresholds)
    print("=== Overall evaluation ===")
    summary = table_all.summary(["csi", "pod", "far", "hss"])
    for name, vals in summary.items():
        val_str = ", ".join(f"{float(v):.4f}" for v in vals)
        print(f"  {name:4s}: [{val_str}]")
    print(f"  thresholds: {thresholds}")

    # ---------------------------------------------------------------
    # 3. Per-lead-time evaluation (preserve frame dimension)
    # ---------------------------------------------------------------
    table_lt = ContingencyTable(fcst, obs, thresholds=thresholds, axis=(1, 2))
    csi_per_lt = table_lt.csi()
    print(f"\n=== Per-lead-time CSI ===  shape: {csi_per_lt.shape}")
    print("  (rows=thresholds, cols=lead_time)")
    for i, thresh in enumerate(thresholds):
        vals = ", ".join(f"{float(csi_per_lt[i, t]):.3f}" for t in [0, 4, 9, 14, 19])
        print(f"  dBZ>={thresh:2.0f}: [{vals}]  (t=0,4,9,14,19)")

    # ---------------------------------------------------------------
    # 4. Accumulator mode (simulating streaming data)
    # ---------------------------------------------------------------
    ca = metvane.ContingencyAccumulator(thresholds, preserve_axes=[0])
    batch_size = 5
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        ca.update(fcst[start:end], obs[start:end])

    result = ca.compute()
    print(f"\n=== Accumulated CSI ===  shape: {result['csi'].shape}")

    # ---------------------------------------------------------------
    # 5. Shortcut functions for single-threshold evaluation
    # ---------------------------------------------------------------
    csi_20 = metvane.csi(fcst, obs, threshold=20)
    pod_20 = metvane.pod(fcst, obs, threshold=20)
    far_20 = metvane.far(fcst, obs, threshold=20)
    print(f"\nSingle-threshold (≥20 dBZ):")
    print(f"  CSI = {float(csi_20):.4f}")
    print(f"  POD = {float(pod_20):.4f}")
    print(f"  FAR = {float(far_20):.4f}")

    # ---------------------------------------------------------------
    # 6. FSS (spatial verification)
    # ---------------------------------------------------------------
    for ws in [3, 5, 11, 21]:
        fss_val = metvane.fss(fcst[0], obs[0], threshold=20, window_size=ws)
        print(f"  FSS (window={ws:2d}, ≥20 dBZ): {fss_val:.4f}")

    print("\n✓ Nowcasting evaluation complete.")


if __name__ == "__main__":
    main()
