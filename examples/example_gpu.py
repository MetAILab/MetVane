"""Example: GPU-accelerated evaluation with PyTorch.

Shows how to run metrics on CUDA tensors and how to use the backend
conversion to switch between numpy and torch.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import metvane


def main():
    if not HAS_TORCH:
        print("PyTorch not available. Install it to run GPU examples.")
        return

    rng = np.random.default_rng(42)
    shape = (10, 128, 256)

    # ---------------------------------------------------------------
    # 1. Direct torch tensor input
    # ---------------------------------------------------------------
    fcst_t = torch.tensor(rng.standard_normal(shape).astype(np.float32))
    obs_t = torch.tensor(rng.standard_normal(shape).astype(np.float32))

    rmse_cpu = metvane.rmse(fcst_t, obs_t, axis=(1, 2))
    print(f"CPU torch RMSE shape: {rmse_cpu.shape}, dtype: {rmse_cpu.dtype}")

    # ---------------------------------------------------------------
    # 2. backend="torch" converts numpy arrays to torch
    # ---------------------------------------------------------------
    fcst_np = rng.standard_normal(shape).astype(np.float32)
    obs_np = rng.standard_normal(shape).astype(np.float32)
    rmse_converted = metvane.rmse(fcst_np, obs_np, backend="torch")
    print(f"Converted to torch: type={type(rmse_converted)}")

    # ---------------------------------------------------------------
    # 3. GPU acceleration (if available)
    # ---------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        fcst_gpu = fcst_t.to(device)
        obs_gpu = obs_t.to(device)

        rmse_gpu = metvane.rmse(fcst_gpu, obs_gpu, axis=(1, 2))
        print(f"GPU RMSE device: {rmse_gpu.device}, shape: {rmse_gpu.shape}")

        bias_gpu = metvane.bias(fcst_gpu, obs_gpu, axis=(1, 2))
        print(f"GPU Bias device: {bias_gpu.device}")

        # Categorical on GPU
        fcst_radar = torch.rand(20, 256, 256, device=device) * 50
        obs_radar = torch.rand(20, 256, 256, device=device) * 50
        table = metvane.ContingencyTable(
            fcst_radar, obs_radar, thresholds=[20, 35],
        )
        print(f"GPU CSI: {table.csi()}")

        # Accumulator on GPU
        acc = metvane.ContinuousAccumulator(
            ["rmse", "mae"], preserve_axes=[0], backend="torch", device=device,
        )
        acc.update(fcst_np[:5], obs_np[:5])
        acc.update(fcst_np[5:], obs_np[5:])
        result = acc.compute()
        print(f"GPU Accumulator RMSE device: {result['rmse'].device}")
    else:
        print("CUDA not available — GPU examples skipped.")

    print("\n✓ GPU evaluation example complete.")


if __name__ == "__main__":
    main()
