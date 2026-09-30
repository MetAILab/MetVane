# 快速开始指南

## 安装

```bash
cd MetVane
pip install -e ".[all]"
```

下面的代码块可以按顺序直接运行（`tests/test_docs_snippets.py` 会执行它们）；实际使用时把模拟数据换成 `np.load` / `xr.open_dataset` 读入的数据即可。

## 场景 1：中短期天气预报

数据形状 `(init, lead, lat, lon)`，纬度按 ERA5 顺序（北→南）：

```python
import numpy as np
import metvane

rng = np.random.default_rng(0)
n_init, n_lead = 4, 12
lat = np.linspace(90, -90, 181)
lon = np.arange(0, 360, 1.0)
clim = rng.standard_normal((181, 360)) * 5
obs = clim + rng.standard_normal((n_init, n_lead, 181, 360))
fcst = obs + rng.standard_normal(obs.shape) * np.linspace(0.2, 2, n_lead)[None, :, None, None]

w = metvane.latitude_weights(lat)[:, None]                     # (181, 1)

rmse = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w)      # (12,)
bias = metvane.bias(fcst, obs, axis=(0, 2, 3), weights=w)      # 平均误差 ME
acc = metvane.acc(fcst, obs, clim, axis=(2, 3), mean_over=0, weights=w)   # WB2 口径 ACC

# 区域评估：把区域作为 mask 与纬度权重一起传入（不要用全局归一化权重的子集 + multiply）
nh = metvane.region_mask(lat, lon, "NH")
rmse_nh = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w, mask=nh)

# 一次算多个指标
scores = metvane.continuous_scores(fcst, obs, ["rmse", "mae", "bias"], axis=(0, 2, 3), weights=w)
```

## 场景 2：短临预报（雷达回波 / 降水）

```python
from metvane import ContingencyTable

radar_obs = rng.uniform(0, 60, (6, 20, 128, 128))              # (case, lead, H, W) dBZ
radar_fcst = np.clip(radar_obs + rng.normal(0, 5, radar_obs.shape), 0, 65)
radar_obs[:, :, :16, :16] = np.nan                             # 无观测区域：NaN，不计入任何计数

table = ContingencyTable(radar_fcst, radar_obs, thresholds=[20, 35, 40], axis=(0, 2, 3))
csi = table.csi()                        # (3, 20)，无事件的 lead 为 NaN（而不是 0）
summary = table.summary(["csi", "pod", "far", "hss", "frequency_bias"])

# 降水阈值：'>=' 包含等号；0 mm 请用 op='>'，或使用 0.1 mm
precip_f = rng.gamma(0.3, 2.0, (6, 12, 64, 64))
precip_o = rng.gamma(0.3, 2.0, (6, 12, 64, 64))
ts = ContingencyTable(precip_f, precip_o, [0.1, 1.0, 5.0], axis=(0, 2, 3)).csi()

# FSS：多帧池化（1 - Σnum/Σden），逐 lead
fss = metvane.fss(radar_fcst[:, 0], radar_obs[:, 0], 35.0, window_size=11)
fa = metvane.FSSAccumulator([35.0], [1, 5, 11], preserve_axes=[1])
fa.update(radar_fcst, radar_obs)
fss_by_lead = fa.compute()["fss"]        # (1 阈值, 3 窗口, 20 lead)
```

## 场景 3：GPU

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
f_t = torch.from_numpy(fcst).to(device)
o_t = torch.from_numpy(obs).to(device)
rmse_t = metvane.rmse(f_t, o_t, axis=(0, 2, 3), weights=w)     # numpy 权重自动转到 device
np.testing.assert_allclose(rmse_t.cpu().numpy(), rmse, rtol=1e-10)

# numpy 数据也可以强制在 GPU 上计算
metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w, backend="torch", device=device)
```

## 场景 4：大数据集分块

分块只能沿**被归约**的轴（如起报时间）切分；保留轴的长度在各块之间必须一致（会被检查）。

```python
accum = metvane.ContinuousAccumulator(["rmse", "acc", "acc_mean"], preserve_axes=[1],
                                      weights=w, spatial_axes=(2, 3))
for s in range(0, n_init, 3):
    accum.update(fcst[s:s + 3], obs[s:s + 3], climatology=clim)   # 时变气候态可逐块传入
res = accum.compute()
np.testing.assert_allclose(res["rmse"], rmse, rtol=1e-10)
np.testing.assert_allclose(res["acc_mean"], acc, rtol=1e-10)

# 多进程：各自累加后 merge，或保存 state_dict
other = metvane.ContinuousAccumulator(["rmse", "acc", "acc_mean"], preserve_axes=[1],
                                      weights=w, spatial_axes=(2, 3))
state = accum.state_dict()
other.load_state_dict(state)
```

## 场景 5：xarray / dask

```python
import xarray as xr
import metvane.xr_api as mxr

dims = ("time", "lead_time", "latitude", "longitude")
coords = {"time": np.arange(n_init), "lead_time": np.arange(n_lead), "latitude": lat, "longitude": lon}
ds_f = xr.Dataset({"t2m": (dims, fcst)}, coords=coords)
ds_o = xr.Dataset({"t2m": (dims, obs)}, coords=coords).sortby("latitude")   # 纬度顺序不同也没关系
w_da = metvane.latitude_weights(ds_f.latitude)

rmse_ds = mxr.rmse(ds_f, ds_o, preserve_dims="lead_time", weights=w_da)     # Dataset，每个变量一条曲线
clim_da = xr.DataArray(clim, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
acc_da = mxr.acc(ds_f.t2m, ds_o.t2m, clim_da, reduce_dims=["latitude", "longitude"],
                 mean_over="time", weights=w_da)
np.testing.assert_allclose(acc_da.values, acc, rtol=1e-10)
```

对 `xr.open_mfdataset(..., chunks=...)` 打开的 dask 数据，结果是惰性的，调用 `.compute()` 时按块计算。
