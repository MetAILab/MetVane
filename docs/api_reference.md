# MetVane API 参考

## 回归指标 — `metvane.continuous`

所有函数共享统一签名：

```python
metric(fcst, obs, *, axis=None, weights=None, skipna=True, weight_mode="mean", backend=None, device=None)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `fcst` | array-like | 预报数据（numpy/torch/xarray） |
| `obs` | array-like | 观测数据 |
| `axis` | int or tuple | 聚合维度，`None` 表示全部，`()` 表示不归约（逐格点） |
| `weights` | array-like | 权重数组（如纬度权重）；权重和为 0（如区域掩码外）的切片返回 NaN |
| `skipna` | bool | 是否跳过 NaN |
| `weight_mode` | str | `"mean"`（默认）：Σ(w·x)/Σw；`"multiply"`：Σ(w·x)/N，在被保留的维上保留权重（见 README“全网格评估”） |
| `backend` | str | 强制使用 `'numpy'` 或 `'torch'` |
| `device` | str | Torch 设备（仅 backend='torch' 时有效） |

### metvane.rmse

均方根误差。

```python
>>> metvane.rmse(fcst, obs)
>>> metvane.rmse(fcst, obs, axis=(1, 2), weights=w)
```

### metvane.mse

均方误差。

### metvane.mae

平均绝对误差。

### metvane.bias

偏差：`mean(fcst - obs)`。

### metvane.acc

距平相关系数。需额外提供 `climatology` 参数。

```python
>>> metvane.acc(fcst, obs, clim, axis=(1, 2), weights=w)
```

### metvane.pearson_correlation

加权皮尔逊相关系数。

### metvane.wind_vector_rmse

风矢量 RMSE。需提供 u/v 两个分量。

```python
>>> metvane.wind_vector_rmse(u_fcst, v_fcst, u_obs, v_obs, axis=(1, 2))
```

---

## 分类指标 — `metvane.categorical`

### ContingencyTable

```python
table = metvane.ContingencyTable(fcst, obs, thresholds=[20, 35, 40],
                                  op='>=', axis=(1, 2))
```

| 方法 | 说明 |
|------|------|
| `table.csi()` | 临界成功指数 |
| `table.pod()` | 命中率 |
| `table.far()` | 虚假警报率 |
| `table.pofd()` | 误报率 |
| `table.hss()` | Heidke 技巧评分 |
| `table.ets()` | 公正威胁评分 |
| `table.bias_score()` | 频率偏差 |
| `table.f1()` | F1 分数 |
| `table.accuracy()` | 总体准确率 |
| `table.summary()` | 所有指标字典 |

### 快捷函数

```python
metvane.csi(fcst, obs, threshold=20, axis=(1, 2))
metvane.pod(fcst, obs, threshold=20)
metvane.far(fcst, obs, threshold=20)
metvane.hss(fcst, obs, threshold=20)
metvane.ets(fcst, obs, threshold=20)
```

---

## 空间指标 — `metvane.spatial`

### metvane.fss

分数技巧评分（Fractions Skill Score）。

```python
>>> metvane.fss(fcst_2d, obs_2d, threshold=20, window_size=5)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `fcst`, `obs` | 2-D array | 预报和观测场 |
| `threshold` | float | 事件阈值 |
| `window_size` | int | 邻域窗口大小（像素） |
| `op` | str | 比较运算符 |

---

## 概率指标 — `metvane.probabilistic`

### metvane.crps_ensemble

集合预报的连续排序概率评分。

```python
>>> metvane.crps_ensemble(ensemble_fcst, obs, member_axis=0)
```

### metvane.brier_score

Brier 评分。

```python
>>> metvane.brier_score(prob_fcst, binary_obs)
```

---

## 累加器 — `metvane.accumulator`

### ContinuousAccumulator

```python
acc = metvane.ContinuousAccumulator(
    metrics=["rmse", "mae", "bias"],
    preserve_axes=[0],
    weights=w,
    weight_mode="mean",      # 或 "multiply"；preserve_axes 覆盖全部轴 = 逐格点累加
    backend="torch",
    device="cuda",
)
acc.update(fcst_chunk, obs_chunk)
result = acc.compute()  # dict[str, array]
acc.reset()
```

### ContingencyAccumulator

```python
ca = metvane.ContingencyAccumulator(
    thresholds=[20, 35, 40],
    preserve_axes=[0],
)
ca.update(fcst_chunk, obs_chunk)
result = ca.compute()           # dict[str, array]
table = ca.as_contingency_table()  # ContingencyTable 对象
ca.reset()
```

---

## 后端工具 — `metvane.core`

### metvane.detect_backend

```python
>>> metvane.detect_backend(np.array([1.0]))
Backend.NUMPY
```

### metvane.convert

```python
>>> metvane.convert(np_array, metvane.Backend.TORCH, device="cuda")
```

### metvane.latitude_weights

```python
>>> w = metvane.latitude_weights(lat, normalize=True)   # cos(lat)/mean；内部 float64 计算，返回 lat 的浮点精度
```

### metvane.broadcast_weights

```python
>>> w_4d = metvane.broadcast_weights(w, target_shape=(4, 10, 32, 64), lat_axis=-2)
```

### metvane.region_mask

```python
>>> mask = metvane.region_mask(lat, lon, "NH")
>>> mask = metvane.region_mask(lat, lon, (15, 55, 70, 140))  # 自定义区域
```

---

## xarray 便捷层 — `metvane.xr_api`

```python
import metvane.xr_api as mxr
```

所有函数用 `reduce_dims` / `preserve_dims` 替换 `axis` 参数。

```python
mxr.rmse(fcst_da, obs_da, preserve_dims="lead_time")
mxr.mae(fcst_da, obs_da, reduce_dims=["lat", "lon"])
mxr.acc(fcst_da, obs_da, clim_da, preserve_dims="lead_time")
mxr.rmse(fcst_da, obs_da, preserve_dims="all")                         # 不归约，保留全部维与坐标
mxr.rmse(fcst_da, obs_da, reduce_dims="time", weights=w_da, weight_mode="multiply")
```

结果保留未被归约的维（原顺序）及只依赖这些维的坐标，无论用 `reduce_dims` 还是 `preserve_dims` 指定。

支持 `xr.Dataset` 自动遍历所有变量。
