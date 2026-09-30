# MetVane 架构设计文档

## 1. 设计目标

- 中短期天气预报：RMSE、MAE、ACC、风矢量 RMSE 等回归指标
- 短临预报：CSI、POD、FAR 等分类指标，FSS 空间指标
- 多后端：NumPy、PyTorch（含 GPU）共用一套实现；xarray 层按维名/坐标对齐
- 灵活聚合：整体 / 按预报时刻 / 按区域 / 逐格点
- 大数据集：累加器分块处理，可跨进程合并

## 2. 分层

```text
          metvane.rmse / csi / fss / ...        函数式 API（numpy / torch）
          metvane.xr_api.*                      xarray 层（对齐 + xarray 原生运算，dask 惰性）
          *Accumulator                          分块累加
                     │
          core/prepare.py                       统一输入准备层
          ├─ resolve_backend / prepare          后端、设备、dtype 规则，拒绝 xarray
          ├─ normalize_axis                     轴规范化（负轴、越界、重复、() = 不归约）
          ├─ check_same_shape                   禁止隐式广播
          ├─ align_to / align_mask              权重、气候态、掩码转到数据的后端/设备/dtype，并做广播检查
          ├─ validity                           NaN / 掩码 → 有效掩码（inf 是数值）
          └─ safe_divide                        零分母 → NaN
                     │
          continuous/_impl.weighted_sums        共享的充分统计量（函数式与累加器共用）
          continuous/_impl.correlation_sums
          categorical/_impl.compute_contingency int64 计数
          spatial/_impl.fss_components          积分图、可池化分量
                     │
          core/array_ns (NumpyNS / TorchNS)     只封装 numpy 与 torch 有差异的操作
```

## 3. 命名空间适配器（array_ns）

`get_namespace(*arrays)` 返回 `NumpyNS` 或 `TorchNS`；指标实现只调用 `xp.sum(x, axis, keepdims=, dtype=)`、`xp.where`、`xp.nan_like` 等统一接口。
torch 下 `axis=()` 被特殊处理为"不归约"（`torch.sum(dim=())` 会归约全部维）。

dtype 规则（`as_float`）：float16/bfloat16 → float32；float32/float64 保持（torch 不降精度）；整数/布尔 → float64（numpy）/ float32（torch）；MaskedArray → 掩码处为 NaN。

## 4. 统一的统计口径

| 规则 | 实现位置 |
|---|---|
| NaN = 缺测，`inf` = 数值 | `prepare.validity` |
| 零分母 → NaN | `prepare.safe_divide`（双层 where，无 0/0 警告） |
| 0 权重处不产生 0·inf | `weighted_sums` 先把 w==0 处的数据置 0 |
| 分类计数 int64，分数 float64 | `compute_contingency`、`ContingencyTable._f64` |
| 累加器状态 float64 | `weighted_sums(dtype=float64)` |

函数式 API 与累加器调用同一个 `weighted_sums` / `correlation_sums`，因此沿被归约轴分块累加的结果与一次性计算一致（含 NaN、权重、`weight_mode`）。

## 5. 累加器

`accumulator/_base.AccumulatorBase` 提供：
- `axis=` 与 `preserve_axes=` 互斥的轴解析；
- 首块记录维数与保留轴长度，之后不一致即报错（防止沿保留轴分块被逐元素相加）；
- 原子 `update`（统计量先算进局部变量，全部成功后再提交）；
- `merge`（校验配置）、`state_dict` / `load_state_dict`（带版本号）、只读 `stats`。

`ContinuousAccumulator` 支持逐块气候态（`update(climatology=...)`）和逐样本 ACC 平均（`acc_mean`）；`FSSAccumulator` 累加 FSS 的分子、分母与基率计数。

## 6. xarray 层

1. `align_inputs`：obs 同维校验并按 fcst 转置；坐标重排/子集选择（`join='reorder'|'exact'|'inner'`，数值坐标按 `coord_tol` 匹配）；气候态、权重、掩码在共享维上同样对齐。
2. 维名校验（`reduce_dims` / `preserve_dims`，未知维名报错）。
3. 用 xarray 原生运算计算加权和，再求比值；在 `xr.set_options(arithmetic_join="exact")` 下运算，避免隐式内连接。dask 输入保持惰性。

## 7. 数据流

```text
输入 (numpy / torch)
  → resolve_backend / ensure_same_backend（拒绝 xarray；多个 torch 输入须在同一设备）
  → as_float（dtype 规则）
  → normalize_axis / check_same_shape / align_to / align_mask
  → validity
  → weighted_sums / correlation_sums / compute_contingency / fss_components
  → safe_divide
  → 结果（与输入同后端、同设备）
```
