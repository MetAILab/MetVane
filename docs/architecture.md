# MetVane 架构设计文档

## 1. 设计目标

MetVane 旨在提供一个简洁、统一的气象评估框架，满足以下需求：

- **中短期天气预报**：RMSE、MAE、ACC、风矢量 RMSE 等回归指标
- **短临预报（临近预报）**：CSI、POD、FAR 等分类指标，FSS 空间指标
- **多后端支持**：NumPy、PyTorch（含 GPU）、xarray
- **灵活聚合**：整体 / 按预报时刻 / 按区域
- **大数据集支持**：累加器模式，分块处理

## 2. 核心架构

```
                 用户 API
          ┌──────────────────┐
          │   metvane.rmse() │  函数式 API
          │   metvane.csi()  │
          └────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
continuous/   categorical/   spatial/      指标子包
_impl.py      _impl.py      _impl.py
    │              │              │
    └──────────────┼──────────────┘
                   │
          ┌────────┴─────────┐
          │  core/array_ns   │  命名空间适配器
          │  get_namespace() │
          └────────┬─────────┘
                   │
          ┌────────┴─────────┐
          │  NumpyNS  或     │
          │  TorchNS         │
          └──────────────────┘
```

## 3. 命名空间适配器 (array_ns)

这是 MetVane 最核心的设计创新。

### 问题

NumPy 和 PyTorch 的 API 存在差异：

| 操作 | NumPy | PyTorch |
|------|-------|---------|
| 求和 | `np.sum(x, axis=0)` | `torch.sum(x, dim=0)` |
| 广播 | `np.broadcast_to(x, shape)` | `x.expand(shape)` |
| NaN 均值 | `np.nanmean(x)` | `torch.nanmean(x)` |

### 解决方案

```python
class NumpyNS:
    @staticmethod
    def sum(x, axis=None):
        return np.sum(x, axis=axis)

class TorchNS:
    @staticmethod
    def sum(x, axis=None):
        return torch.sum(x, dim=axis)

def get_namespace(*arrays):
    # 自动检测输入类型，返回对应命名空间
```

### 效果

- 所有指标只需写 **一套代码**
- 使用 `xp = get_namespace(fcst)` 获取适配器
- 然后用 `xp.sum()`, `xp.mean()` 等统一接口
- NumPy 和 PyTorch 自动切换

## 4. 后端检测与转换 (backend)

```python
# 自动检测
metvane.detect_backend(numpy_array)    # → Backend.NUMPY
metvane.detect_backend(torch_tensor)   # → Backend.TORCH
metvane.detect_backend(xr_dataarray)   # → Backend.XARRAY

# 显式转换
metvane.convert(numpy_array, Backend.TORCH, device="cuda")

# 确保同后端
fcst, obs = ensure_same_backend(fcst, obs)
# 如果有一个是 torch，全部提升为 torch
```

## 5. 指标分类

### 5.1 回归指标 (continuous/)

```
_impl.py → rmse(), mse(), mae(), bias(), acc(), pearson_correlation(), wind_vector_rmse()
```

所有函数共享 `_weighted_mean()` 内部工具，支持加权和 NaN 跳过。

### 5.2 分类指标 (categorical/)

```
_impl.py → compute_contingency()  → TP, FP, FN, TN
__init__.py → ContingencyTable 类  → csi(), pod(), far(), hss(), ets(), ...
```

ContingencyTable 一次性计算列联表，然后按需查询任意分数。

### 5.3 空间指标 (spatial/)

```
_impl.py → fss()  → Fractions Skill Score
```

使用累积和实现高效的 2D 均匀滤波器。

### 5.4 概率指标 (probabilistic/)

```
_impl.py → crps_ensemble(), brier_score()
```

## 6. 累加器模式

对于无法一次性加载到内存的大数据集：

```python
# 连续指标累加器
acc = ContinuousAccumulator(["rmse", "mae"], preserve_axes=[0])
for chunk in loader:
    acc.update(chunk_fcst, chunk_obs)
result = acc.compute()

# 分类指标累加器
ca = ContingencyAccumulator([20, 35, 40], preserve_axes=[0])
for chunk in loader:
    ca.update(chunk_fcst, chunk_obs)
result = ca.compute()
```

**关键设计**：累加器只维护中间统计量（平方误差和、绝对误差和、计数等），而非原始数据。

## 7. xarray 便捷层

```python
import metvane.xr_api as mxr

# 通过命名维度控制聚合
mxr.rmse(fcst, obs, preserve_dims="lead_time")
mxr.rmse(fcst, obs, reduce_dims=["lat", "lon"])

# 自动支持 xr.Dataset（多变量）
mxr.rmse(fcst_ds, obs_ds, preserve_dims="lead_time")
```

## 8. 数据流

```
用户输入 (numpy/torch/xarray)
    │
    ▼
detect_backend() → 检测后端
    │
    ▼
ensure_same_backend() → 对齐后端
    │
    ▼
get_namespace() → 获取适配器 (NumpyNS/TorchNS)
    │
    ▼
_impl.py → 使用适配器执行计算
    │
    ▼
返回结果 (同后端类型)
```

## 9. 与参考框架的对比

| 特性 | WeatherBenchX | nci/scores | torchmetrics | **MetVane** |
|------|:---:|:---:|:---:|:---:|
| NumPy 支持 | ✗ (xarray) | ✓ (xarray) | ✗ (torch) | **✓** |
| PyTorch 支持 | ✗ | ✗ | ✓ | **✓** |
| xarray 支持 | ✓ | ✓ | ✗ | **✓** |
| GPU 加速 | ✗ | ✗ | ✓ | **✓** |
| 累加器模式 | ✗ | ✗ | ✓ | **✓** |
| 气象专用权重 | ✓ | 部分 | ✗ | **✓** |
| 分类指标 | 有限 | ✓ | ✓ | **✓** |
| 空间指标 | ✗ | 有限 | ✗ | **✓** (FSS) |
| 单套代码多后端 | N/A | N/A | N/A | **✓** (array_ns) |
