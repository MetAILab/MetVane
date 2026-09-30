# MetVane

**Meteorological Verification and Numerical forecast Evaluation**

面向气象领域的统一评估框架 — 支持中短期天气预报和短临预报

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**语言 / Language**: [English](README_en.md) | **中文**

## 特性

- **多后端**：NumPy / PyTorch（CPU、CUDA）使用同一套实现；结果留在输入的后端与设备上
- **xarray 层**：按维名和坐标对齐（纬度反序、维顺序不同都能正确配对），dask 输入保持惰性
- **四类指标**：回归（连续）、分类、空间（FSS）、概率（实验性）
- **统一的缺测规则**：NaN（以及 MaskedArray 的掩码）= 缺测，所有指标一致跳过；`inf` 视为数值
- **累加器**：`update/compute` 分块计算，int64/float64 状态，支持 `merge`、`state_dict`
- **气象工具**：纬度权重（cos / 单元面积）、区域掩码（0..360 与 -180..180 经度通用）

## 安装

```bash
cd MetVane
pip install -e .              # 仅 numpy
pip install -e ".[torch]"     # + PyTorch
pip install -e ".[xarray]"    # + xarray
pip install -e ".[dev]"       # 测试依赖
```

## 快速开始

### 函数式 API（numpy / torch）

```python
import numpy as np
import metvane

rng = np.random.default_rng(0)
lat = np.linspace(90, -90, 181)
fcst = rng.standard_normal((4, 10, 181, 360)).astype(np.float32)   # (init, lead, lat, lon)
obs = rng.standard_normal((4, 10, 181, 360)).astype(np.float32)

w = metvane.latitude_weights(lat)[:, None]         # (nlat, 1)：沿纬度广播
rmse = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w)          # (10,) 每个 lead 一个值
acc = metvane.acc(fcst, obs, 0.0, axis=(2, 3), mean_over=0, weights=w)  # 逐起报空间 ACC 再平均
```

> 1-D 纬度权重直接传给 N 维数据会被拒绝（numpy 会把它对齐到最后一维——经度）。请用
> `w[:, None]` 或 `metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)`。
> fcst 与 obs 形状必须完全相同，不做隐式广播。

### PyTorch / GPU

```python
import torch
import metvane

device = "cuda" if torch.cuda.is_available() else "cpu"
f = torch.randn(10, 181, 360, device=device)
o = torch.randn(10, 181, 360, device=device)
w = metvane.latitude_weights(np.linspace(90, -90, 181))[:, None]   # numpy 权重自动转到数据设备
metvane.rmse(f, o, axis=(1, 2), weights=w)          # 结果在同一设备上，float64 输入不降精度
```

### xarray（按维名与坐标对齐）

```python
import xarray as xr
import metvane.xr_api as mxr

coords = {"lead_time": np.arange(10), "latitude": lat, "longitude": np.arange(360.0)}
f_da = xr.DataArray(fcst[0], dims=("lead_time", "latitude", "longitude"), coords=coords)
o_da = xr.DataArray(obs[0], dims=("lead_time", "latitude", "longitude"), coords=coords)
o_da = o_da.sortby("latitude")                      # 纬度顺序不同：按坐标自动对齐
w_da = metvane.latitude_weights(f_da.latitude)      # DataArray 权重
mxr.rmse(f_da, o_da, preserve_dims="lead_time", weights=w_da)
```

写错维名（如 `lat` vs `latitude`）会报错；dask 数据返回惰性结果（`.compute()` 求值）。

### 分类指标（短临）

```python
from metvane import ContingencyTable

radar_f = rng.uniform(0, 60, (6, 20, 64, 64))       # (case, lead, H, W) dBZ
radar_o = rng.uniform(0, 60, (6, 20, 64, 64))
radar_o[:, :, :8, :8] = np.nan                      # 雷达盲区 = 缺测，不计入任何计数

table = ContingencyTable(radar_f, radar_o, thresholds=[20, 35, 40], axis=(0, 2, 3))
table.csi().shape                                   # (3, 20)：(阈值, lead)
scores = table.summary(["csi", "pod", "far", "frequency_bias"])
```

### 累加器（只能沿被归约的轴分块）

```python
accum = metvane.ContinuousAccumulator(["rmse", "mae", "bias"], preserve_axes=[1], weights=w)
for s in range(0, 4, 3):                            # 沿 init（轴 0，被归约）分块
    accum.update(fcst[s:s + 3], obs[s:s + 3])
res = accum.compute()                               # 与 metvane.rmse(..., axis=(0, 2, 3)) 相同
np.testing.assert_allclose(res["rmse"], rmse, rtol=1e-5)
```

沿保留轴切分会被检测并报错（各块的保留轴长度必须一致）。

## 评分口径

| 主题 | MetVane 的约定 |
|---|---|
| 缺测 | NaN / MaskedArray 掩码 / `mask=False` 的点从所有统计量中排除；`skipna=False` 时含 NaN 的切片为 NaN；`inf` 是数值（模式发散不会被掩盖） |
| 零分母 | 无定义 → **NaN**（如无事件时的 CSI、两场都无事件时的 FSS、权重和为 0 的切片）；用 `nanmean` 汇总时会被跳过 |
| RMSE 跨样本 | `sqrt(样本平均 MSE)`（WeatherBench 口径），不是逐样本 RMSE 的平均 |
| ACC | 默认未中心化；`axis` 为空间轴、`mean_over` 为样本轴 → 逐样本空间 ACC 再平均（WB2/ECMWF 口径）；把样本轴放进 `axis` 则为池化 ACC；`centered=True` 为中心化 ACC |
| 分类 | 计数为精确 int64，分数为 float64；推荐先合并列联表（累加器 / 求和）再计算分数 |
| FSS | 边界补零；缺测点按邻域有效点数归一化；多场池化 `1 − Σnum/Σden`（`fss_components` / `FSSAccumulator`），与逐帧平均不同 |
| 纬度权重 | `latitude_weights` 归一化为均值 1；自行加权时用 `(w*e).sum()/w.sum()`，区域子集用子集自己的权重和；`method='area'` 为单元面积权重 |
| `weight_mode` | `"mean"`（默认）Σw·x/Σw；`"multiply"` Σw·x/N，保留纬度时得到面积贡献场；区域、掩码、含缺测时请用 `"mean"`（否则会警告） |
| 术语 | FAR = 空报率 FP/(TP+FP)；POFD = 误报率 FP/(FP+TN)（`false_alarm_rate` 为其别名，已弃用）；`metvane.bias` 是平均误差 ME，频率偏差是 `frequency_bias` |

### 全网格（逐格点）评估与 `weight_mode`

```python
rmse_map = metvane.rmse(fcst, obs, axis=(0,))                        # (lead, lat, lon) 逐格点 RMSE
contrib = metvane.rmse(fcst, obs, axis=(0,), weights=w, weight_mode="multiply")
# contrib**2 的格点简单平均 == 纬度加权 MSE（全球网格、权重均值 1、无缺测时）
np.testing.assert_allclose((contrib ** 2).mean(axis=(1, 2)),
                           metvane.mse(fcst, obs, axis=(0, 2, 3), weights=w), rtol=1e-4)
```

## 函数式 axis 与累加器对照

| 需求 | 函数式 | 累加器 |
|---|---|---|
| 保留 lead（轴 1） | `axis=(0, 2, 3)` | `preserve_axes=[1]` 或 `axis=(0, 2, 3)` |
| 全部归约 | `axis=None` | 默认 |
| 不归约（逐点） | `axis=()` | `preserve_axes` 覆盖全部轴 |

分类结果形状为 `(n_thresholds, *保留轴)`。

## 指标列表

### 顶层函数

| 函数 | 说明 |
|------|------|
| `rmse` / `mse` / `mae` / `bias` | 连续指标（`bias` = 平均误差 ME） |
| `continuous_scores` | 一次计算多个连续指标 |
| `acc` | 距平相关系数（`mean_over`、`centered`） |
| `pearson_correlation` | 加权皮尔逊相关 |
| `wind_vector_rmse` | 风矢量 RMSE |
| `csi` / `pod` / `far` / `pofd` / `hss` / `ets` / `frequency_bias` / `f1` | 单阈值分类快捷函数 |
| `fss` / `fss_components` / `fss_from_components` | 分数技巧评分及其可池化分量 |
| `crps_ensemble`（`fair=`）/ `brier_score`（`threshold=`） | 概率指标（实验性） |

### `ContingencyTable` 方法

`csi`（`threat_score`）、`pod`（`hit_rate`）、`far`（`false_alarm_ratio`）、`pofd`、`hss`、`ets`（`gilbert_skill_score`）、
`bias_score`（`frequency_bias`）、`f1`、`pc`（`proportion_correct`，旧名 `accuracy`）、`summary(metrics)`（大小写不敏感，接受别名）、
`from_counts(tp, fp, fn, tn)`。

### 累加器

`ContinuousAccumulator`（rmse/mse/mae/bias/acc/acc_mean）、`ContingencyAccumulator`、`FSSAccumulator`。

## 运行测试与示例

```bash
cd MetVane
PYTHONPATH=src python -m pytest tests            # CPU（CUDA 用例自动跳过）
sbatch scripts/run_gpu_tests.sh                   # 在 GPU 节点上运行全部测试（含 CUDA 用例）
PYTHONPATH=src python scripts/coverage.py --fail-under 90   # 行覆盖率（仅标准库）
python examples/example_medium_range.py
python examples/example_nowcasting.py
python examples/example_gpu.py
python examples/example_xarray.py
```

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## License

Apache License 2.0
