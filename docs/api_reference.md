# MetVane API 参考

> 以下为签名与语义说明；可运行示例见 [quickstart](quickstart.md) 与 README。

## 通用约定

- `fcst`、`obs`：numpy 数组或 torch 张量，**形状必须完全相同**（不做隐式广播）。xarray 对象会被拒绝（TypeError），请用 `metvane.xr_api`。
- `axis`：int / int 序列 / `None`（全部归约）/ `()`（不归约）。支持负轴号；越界抛 `AxisError`，重复轴抛 `ValueError`。
- `weights`、`climatology`、`mask`：自动转换到数据的后端、设备和 dtype；必须能按 numpy 规则广播到数据形状。N 维数据不接受 1-D 数组（请用 `w[:, None]` 或 `broadcast_weights`）。
- 缺测：NaN 与 MaskedArray 掩码 = 缺测；`skipna=True`（默认）时排除；`inf` 是数值。
- 零分母：返回 NaN。
- dtype：float16/bfloat16 → float32；float32/float64 保持；整数/布尔 → float64（numpy）或 float32（torch）。
- `backend='numpy'|'torch'` 强制后端；`device` 只能与 `backend='torch'` 同时给出；`backend='xarray'` 已弃用（DeprecationWarning，按 numpy 计算，0.3 起报错）。numpy 权重/气候态转到 GPU 时会缓存（`metvane.core.prepare.clear_conversion_cache()` 清空；原地修改已缓存数组不会被检测）。

## 回归指标

```text
rmse / mse / mae / bias(fcst, obs, *, axis=None, weights=None, skipna=True, weight_mode="mean", mask=None, backend=None, device=None)
continuous_scores(fcst, obs, metrics=("rmse","mae","bias"), *, <同上>) -> dict
acc(fcst, obs, climatology, *, axis=None, weights=None, skipna=True, mask=None, centered=False, mean_over=None, backend=None, device=None)
pearson_correlation(fcst, obs, *, axis=None, weights=None, skipna=True, mask=None, mean_over=None, backend=None, device=None)
wind_vector_rmse(u_fcst, v_fcst, u_obs, v_obs, *, axis=None, weights=None, skipna=True, weight_mode="mean", mask=None, backend=None, device=None)
```

| 参数 | 说明 |
|---|---|
| `weight_mode` | `"mean"`：Σw·x/Σw（加权平均）；`"multiply"`：Σw·x/N（在保留的轴上保留权重，得到面积贡献场）。区域/掩码/含缺测或权重均值不为 1 时 `"multiply"` 不等于加权平均，会发出警告 |
| `mask` | 布尔评估掩码（True = 参与），可广播到数据 |
| `climatology` | 可广播到 fcst 的数组或标量；必须对应每个**验证时刻** |
| `axis`（acc/pearson） | 汇入同一个相关系数的轴（通常为空间轴） |
| `mean_over`（acc/pearson） | 样本轴（起报/时间）：先逐样本计算、再求平均（WB2/ECMWF 口径）；不能与 `axis` 重叠 |
| `centered` | 先减去每个切片的（加权）平均距平 |

`bias` 是平均误差（ME），不是分类的频率偏差。RMSE 跨样本为 `sqrt(平均 MSE)`。

## 分类指标

```text
ContingencyTable(fcst, obs, thresholds, *, op=">=", axis=None, skipna=True, mask=None, backend=None, device=None)
ContingencyTable.from_counts(tp, fp, fn, tn, *, thresholds=None, op=">=")
csi / pod / far / pofd / hss / ets / frequency_bias / f1(fcst, obs, threshold, *, op=">=", axis=None, skipna=True, mask=None, backend=None, device=None)
```

- `thresholds`：标量或序列；`op`：`'>=' '>' '<=' '<' '=='`（其他值报 ValueError）。快捷函数只接受单个阈值。
- `tp/fp/fn/tn`：int64，形状 `(n_thresholds, *保留轴)`；`skipna=False` 且切片含 NaN 时为 float64 NaN。
- 比较在数据自身 dtype 下进行（阈值转换为数据 dtype）。
- `op='>='` 且阈值为 0、或列联表退化（全部为事件）时发出 UserWarning。

| 方法 | 公式 | 别名 |
|---|---|---|
| `csi()` | TP/(TP+FP+FN) | `threat_score` |
| `pod()` | TP/(TP+FN) | `hit_rate` |
| `far()` | FP/(TP+FP)（空报率） | `false_alarm_ratio` |
| `pofd()` | FP/(FP+TN)（误报率） | `false_alarm_rate`（已弃用，FutureWarning） |
| `hss()` | 2(TP·TN−FP·FN)/((TP+FN)(FN+TN)+(TP+FP)(FP+TN)) | `heidke_skill_score` |
| `ets()` | D/(D+N(FP+FN))，D=TP·TN−FP·FN | `equitable_threat_score`, `gilbert_skill_score` |
| `bias_score()` | (TP+FP)/(TP+FN) | `frequency_bias` |
| `f1()` | 2TP/(2TP+FP+FN) | |
| `pc()` | (TP+TN)/N | `proportion_correct`, `accuracy` |
| `summary(metrics=None)` | `None` → 全部；名称大小写不敏感，接受别名（如 `'TS'`）；`[]` → `{}` | |
| `n` | TP+FP+FN+TN（有效点数） | |

所有分数为 float64，零分母为 NaN。

## 空间指标

```text
fss(fcst, obs, threshold, *, window_size, op=">=", axis=None, skipna=True, mask=None, no_event_value=nan)
fss_components(fcst, obs, thresholds, window_sizes, *, op=">=", axis=None, skipna=True, mask=None) -> dict
fss_from_components(num, den, *, no_event_value=nan)
```

- 输入 `(..., H, W)`；`window_size` 为正奇数；`axis` 指前导轴（最后两维总被归约），`None` = 全部池化，`()` = 每个场一个值。
- 边界补零；缺测点按窗口内有效点数归一化，只累计至少含一个有效点的窗口。
- 两场都无事件 → NaN（`no_event_value` 可改；0.1 版返回 1.0）。
- `fss_components` 返回 `num`、`den`（float64，`(n_thr, n_win, *kept)`）、`n_obs_events`、`n_valid`（int64）；FSS_useful = 0.5 + f_o/2。

## 概率指标（实验性）

```text
crps_ensemble(ensemble_fcst, obs, *, member_axis=0, axis=None, fair=False, weights=None, skipna=True, mask=None)
brier_score(prob_fcst, obs, *, threshold=None, op=">=", axis=None, weights=None, skipna=True, mask=None)
```

- CRPS 用排序闭式公式（O(M log M)）；`fair=True` 为无偏估计（WB2 使用）；`axis` 按**集合输入**的轴号，不能包含成员轴；obs 形状须等于集合去掉成员轴。
- Brier：概率须在 [0, 1]（疑似百分比会报错）；obs 为 0/1，或给出 `threshold`/`op` 由连续观测生成事件。

## 累加器

```text
ContinuousAccumulator(metrics=("rmse",), *, axis=None, preserve_axes=None, weights=None, climatology=None,
                      spatial_axes=None, weight_mode="mean", skipna=True, backend=None, device=None)
    .update(fcst, obs, *, climatology=None, weights=None, mask=None)
ContingencyAccumulator(thresholds, *, op=">=", axis=None, preserve_axes=None, skipna=True, backend=None, device=None)
    .update(fcst, obs, *, mask=None);  .compute(metrics=None);  .as_contingency_table()
FSSAccumulator(thresholds, window_sizes, *, op=">=", axis=None, preserve_axes=None, skipna=True, backend=None, device=None)
    .update(fcst, obs, *, mask=None);  .compute(no_event_value=nan) -> {"fss", "fss_useful", "base_rate"}
共同方法：compute(), reset(), merge(other), state_dict(), load_state_dict(d), stats
```

- `axis`（要归约的轴）与 `preserve_axes`（要保留的轴）互斥；默认全部归约。
- 只能沿被归约的轴分块；首块记录维数和保留轴长度，之后不一致会报错。
- `update` 是原子的（校验全部通过才写入状态）；状态为 float64/int64；xarray 输入报 TypeError。
- `ContinuousAccumulator` 的 metrics：`rmse mse mae bias acc acc_mean`。`acc` 为池化 ACC，`acc_mean` 为 `spatial_axes` 上逐样本 ACC 的平均。时变气候态通过 `update(climatology=...)` 逐块传入；静态气候态（构造参数）沿样本轴广播时会警告一次。

## 权重与区域

```text
latitude_weights(lat, *, normalize=True, method="cos"|"area", dtype=None)
broadcast_weights(weights, target_shape, *, lat_axis=-2)
region_mask(lat, lon, region)
REGIONS
```

- `latitude_weights`：纬度单位为度（|lat|>90 报错，疑似弧度会警告）；内部 float64；归一化为均值 1；返回与输入同类型（numpy / torch / DataArray）。`'cos'` 为格点 cos(lat)（WeatherBench 口径，极点行权重约为 0），`'area'` 为以中点为边界的单元面积（非等距纬度、极点行更合理；非等距纬度用 `'cos'` 时会警告）。
- `broadcast_weights`：校验 `lat_axis` 范围与长度，返回广播视图。
- `region_mask`：`REGIONS` 统一用 -180..180 表示，按 360 取模比较，0..360 与 -180..180 网格结果一致；`lon_min > lon_max` 表示跨日界线；支持同形状 2-D 经纬度；内置区域为空报错，自定义区域为空警告；torch 输入返回 torch 布尔张量。

## xarray 层 — `metvane.xr_api`

```text
rmse / mse / mae / bias(fcst, obs, *, reduce_dims=None, preserve_dims=None, weights=None, skipna=True,
                        weight_mode="mean", mask=None, join="reorder", coord_tol=1e-4, join_vars="inner")
acc(fcst, obs, climatology, *, reduce_dims, preserve_dims, weights, skipna, mask, centered=False, mean_over=None, join, coord_tol, join_vars)
pearson_correlation(fcst, obs, *, ..., mean_over=None)
wind_vector_rmse(u_fcst, v_fcst, u_obs, v_obs, *, reduce_dims, preserve_dims, weights, skipna, weight_mode, mask, join, coord_tol)
contingency_table(fcst, obs, thresholds, *, op, reduce_dims, preserve_dims, skipna, mask, join, coord_tol) -> Dataset(tp, fp, fn, tn)
scores_from_counts(counts, metrics=None) -> Dataset
categorical_scores(fcst, obs, thresholds, *, metrics=None, op, reduce_dims, preserve_dims, skipna, mask, join, coord_tol) -> Dataset
align_inputs(fcst, obs, *(name, array), join="reorder", coord_tol=1e-4)
```

- 对齐：obs 须与 fcst 同维（按名转置）；坐标为同一集合不同顺序时按 fcst 重排（`join='reorder'`），`'exact'` 要求完全相同，`'inner'` 只评估公共坐标；不一致时报错并指出维名。气候态/权重/掩码的维须为 fcst 维的子集，坐标可更大（如全球权重配区域数据）。ndarray 权重只接受与 fcst 完全相同的形状。
- `reduce_dims` / `preserve_dims` 只能给一个；未知维名报错；`preserve_dims='all'` 保留全部维。
- 使用 xarray 原生运算，dask 输入保持惰性；结果带 `name`（指标名）和 `units`。
- Dataset：逐公共变量计算；变量集合不一致时警告（`join_vars='exact'` 报错）；权重可按变量传 dict；权重维不是变量维子集时报错并给出变量名。

## 后端工具

```text
detect_backend(*arrays)            # 优先级 torch > xarray > numpy，与参数顺序无关
convert(x, Backend.TORCH | Backend.NUMPY, *, device=None)   # 处理负步长视图、标量、MaskedArray
```
