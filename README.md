# MetVane

**Meteorological Verification and Numerical forecast Evaluation**

面向气象领域的统一评估框架 — 支持中短期天气预报和短临预报

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**语言 / Language**: [English](README_en.md) | **中文**

## 特性

- **多后端支持**：NumPy / PyTorch / xarray，自动检测数据格式，零代码切换
- **GPU 加速**：PyTorch 张量自动在 GPU 上计算，无需额外配置
- **命名空间适配器**：一套代码同时支持 NumPy 和 PyTorch，避免代码重复
- **四类指标**：回归（连续）、分类、空间、概率
- **灵活聚合**：整体评估、按预报时刻评估、区域评估
- **累加器模式**：`update/compute` 模式支持超大数据集分块计算
- **xarray 便捷层**：通过命名维度 (`preserve_dims` / `reduce_dims`) 轻松聚合
- **气象工具**：纬度权重、区域掩码、广播权重

## 安装

```bash
# 基础安装（仅 numpy）
pip install -e .

# 安装 PyTorch 支持
pip install -e ".[torch]"

# 安装 xarray 支持
pip install -e ".[xarray]"

# 安装全部依赖
pip install -e ".[all]"

# 开发模式（含测试依赖）
pip install -e ".[dev]"
```

## 快速开始

### 函数式 API（最简用法）

```python
import numpy as np
import metvane

fcst = np.random.randn(10, 181, 360).astype(np.float32)
obs  = np.random.randn(10, 181, 360).astype(np.float32)
lat  = np.linspace(-90, 90, 181, dtype=np.float32)

# 纬度加权 RMSE，按预报时刻输出
w = metvane.latitude_weights(lat)
w = metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)
rmse = metvane.rmse(fcst, obs, axis=(1, 2), weights=w)
# rmse.shape == (10,)
```

### PyTorch + GPU

```python
import torch
import metvane

fcst = torch.randn(10, 181, 360, device="cuda")
obs  = torch.randn(10, 181, 360, device="cuda")
rmse = metvane.rmse(fcst, obs, axis=(1, 2))
# 结果自动保留在 GPU 上
```

### xarray 命名维度

```python
import xarray as xr
import metvane.xr_api as mxr

fcst = xr.DataArray(data, dims=["lead_time", "lat", "lon"])
obs  = xr.DataArray(data, dims=["lead_time", "lat", "lon"])

rmse = mxr.rmse(fcst, obs, preserve_dims="lead_time")
# 自动沿 lat, lon 聚合，保留 lead_time 维度
```

### 分类指标（短临预报）

```python
import metvane
from metvane.categorical import ContingencyTable

table = ContingencyTable(fcst, obs, thresholds=[20, 35, 40], axis=(1, 2))
print(table.csi())     # shape: (3, n_leadtimes)
print(table.pod())
print(table.summary())  # 所有指标的字典
```

### 累加器模式（大数据集）

```python
acc = metvane.ContinuousAccumulator(
    ["rmse", "mae", "bias"],
    preserve_axes=[0],
)
for chunk_f, chunk_o in data_loader:
    acc.update(chunk_f, chunk_o)

result = acc.compute()  # {"rmse": array, "mae": array, "bias": array}
```

## 指标列表

### 回归指标 (`metvane.continuous`)

| 函数 | 说明 |
|------|------|
| `rmse` | 均方根误差 |
| `mse` | 均方误差 |
| `mae` | 平均绝对误差 |
| `bias` | 偏差 (fcst - obs) |
| `acc` | 距平相关系数 |
| `pearson_correlation` | 加权皮尔逊相关系数 |
| `wind_vector_rmse` | 风矢量均方根误差 |

### 分类指标 (`metvane.categorical`)

| 函数/方法 | 说明 |
|-----------|------|
| `ContingencyTable` | 列联表（支持多阈值） |
| `csi` / `threat_score` | 临界成功指数 |
| `pod` / `hit_rate` | 命中率 |
| `far` | 虚假警报率 |
| `hss` | Heidke 技巧评分 |
| `ets` | 公正威胁评分 |
| `bias_score` | 频率偏差 |
| `f1` | F1 分数 |
| `accuracy` | 总体准确率 |

### 空间指标 (`metvane.spatial`)

| 函数 | 说明 |
|------|------|
| `fss` | 分数技巧评分 (Fractions Skill Score) |

### 概率指标 (`metvane.probabilistic`)

| 函数 | 说明 |
|------|------|
| `crps_ensemble` | 连续排序概率评分 |
| `brier_score` | Brier 评分 |


## 运行测试

```bash
cd metvane
pip install -e ".[dev]"
pytest tests/ -v
```

## 运行示例

```bash
cd metvane
python examples/example_medium_range.py
python examples/example_nowcasting.py
python examples/example_gpu.py
python examples/example_xarray.py
```

## License

Apache License 2.0
