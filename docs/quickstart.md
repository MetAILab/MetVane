# 快速开始指南

## 安装

```bash
cd metvane
pip install -e ".[all]"
```

## 场景 1：中短期天气预报评估

假设你有 10 个预报时刻、全球 0.25° 分辨率的预报和观测数据：

```python
import numpy as np
import metvane

# 数据形状: (lead_time, lat, lon)
fcst = np.load("fcst.npy")     # (40, 721, 1440)
obs = np.load("obs.npy")       # (40, 721, 1440)
lat = np.linspace(90, -90, 721, dtype=np.float32)

# 纬度权重
w = metvane.latitude_weights(lat)
w = metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)

# 按预报时刻计算加权 RMSE
rmse = metvane.rmse(fcst, obs, axis=(1, 2), weights=w)
# rmse.shape = (40,)

# 整体评估
rmse_all = metvane.rmse(fcst, obs, weights=w)
# 标量

# 偏差
bias = metvane.bias(fcst, obs, axis=(1, 2), weights=w)
```

## 场景 2：短临预报评估

对于雷达回波临近预报（dBZ 数据）：

```python
import metvane
from metvane.categorical import ContingencyTable

# 数据形状: (n_frames, H, W)
fcst = np.load("radar_pred.npy")   # (20, 256, 256)
obs = np.load("radar_obs.npy")     # (20, 256, 256)

# 多阈值列联表，按帧保留
table = ContingencyTable(
    fcst, obs,
    thresholds=[20, 35, 40],    # dBZ 阈值
    axis=(1, 2),                # 沿空间维度聚合
)

# 查询指标
csi = table.csi()       # shape: (3_thresholds, 20_frames)
pod = table.pod()
far = table.far()

# FSS 空间评估
for frame in range(0, 20, 5):
    score = metvane.fss(fcst[frame], obs[frame], threshold=20, window_size=11)
    print(f"Frame {frame}: FSS = {score:.4f}")
```

## 场景 3：GPU 加速

```python
import torch
import metvane

fcst = torch.randn(10, 181, 360, device="cuda")
obs = torch.randn(10, 181, 360, device="cuda")

# 自动在 GPU 上计算
rmse = metvane.rmse(fcst, obs, axis=(1, 2))
print(rmse.device)  # cuda:0

# 或者将 numpy 数据发送到 GPU
fcst_np = np.load("fcst.npy")
obs_np = np.load("obs.npy")
rmse = metvane.rmse(fcst_np, obs_np, backend="torch", device="cuda")
```

## 场景 4：大数据集分块处理

```python
import metvane

acc = metvane.ContinuousAccumulator(
    ["rmse", "mae", "bias"],
    preserve_axes=[0],   # 保留第 0 维（如预报时刻）
)

for i in range(0, 100, 10):
    fcst_chunk = np.load(f"fcst_{i}.npy")
    obs_chunk = np.load(f"obs_{i}.npy")
    acc.update(fcst_chunk, obs_chunk)

result = acc.compute()
print(result["rmse"])  # 按预报时刻的 RMSE
```

## 场景 5：xarray 工作流

```python
import xarray as xr
import metvane.xr_api as mxr

fcst = xr.open_dataset("fcst.nc")
obs = xr.open_dataset("obs.nc")

# 自动处理多变量 Dataset
rmse = mxr.rmse(fcst, obs, preserve_dims="lead_time")
# 返回 xr.Dataset，每个变量一条 RMSE 曲线
```
