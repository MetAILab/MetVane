# MetVane

**Meteorological Verification and Numerical forecast Evaluation**

A unified evaluation framework for medium-range weather forecasting and nowcasting.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Language / 语言**: **English** | [中文](README.md)

## Features

- **Backends**: one implementation for NumPy and PyTorch (CPU, CUDA); results stay on the input backend/device
- **xarray layer**: aligns by dimension name and coordinate (reversed latitude, transposed dims), keeps dask lazy
- **Four metric families**: continuous, categorical, spatial (FSS), probabilistic (experimental)
- **One missing-data rule**: NaN (and MaskedArray masks) = missing, skipped by every metric; `inf` is a value
- **Accumulators**: chunked `update/compute`, int64/float64 state, `merge`, `state_dict`
- **Meteorological utilities**: latitude weights (cos / cell area), region masks (0..360 and -180..180 longitudes)

## Installation

```bash
cd MetVane
pip install -e .              # numpy only
pip install -e ".[torch]"     # + PyTorch
pip install -e ".[xarray]"    # + xarray
pip install -e ".[dev]"       # test dependencies
```

## Quick start

### Functional API (numpy / torch)

```python
import numpy as np
import metvane

rng = np.random.default_rng(0)
lat = np.linspace(90, -90, 181)
fcst = rng.standard_normal((4, 10, 181, 360)).astype(np.float32)   # (init, lead, lat, lon)
obs = rng.standard_normal((4, 10, 181, 360)).astype(np.float32)

w = metvane.latitude_weights(lat)[:, None]         # (nlat, 1): broadcasts along latitude
rmse = metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w)          # (10,): one value per lead
acc = metvane.acc(fcst, obs, 0.0, axis=(2, 3), mean_over=0, weights=w)  # per-init spatial ACC, averaged
```

> A bare 1-D latitude weight array for N-D data is rejected (numpy would align it with the last
> axis — longitude).  Use `w[:, None]` or `metvane.broadcast_weights(w, fcst.shape, lat_axis=-2)`.
> `fcst` and `obs` must have identical shapes; there is no implicit broadcasting.

### PyTorch / GPU

```python
import torch
import metvane

device = "cuda" if torch.cuda.is_available() else "cpu"
f = torch.randn(10, 181, 360, device=device)
o = torch.randn(10, 181, 360, device=device)
w = metvane.latitude_weights(np.linspace(90, -90, 181))[:, None]   # numpy weights move to the data's device
metvane.rmse(f, o, axis=(1, 2), weights=w)          # result on the same device; float64 is not down-cast
```

### xarray (aligned by name and coordinate)

```python
import xarray as xr
import metvane.xr_api as mxr

coords = {"lead_time": np.arange(10), "latitude": lat, "longitude": np.arange(360.0)}
f_da = xr.DataArray(fcst[0], dims=("lead_time", "latitude", "longitude"), coords=coords)
o_da = xr.DataArray(obs[0], dims=("lead_time", "latitude", "longitude"), coords=coords)
o_da = o_da.sortby("latitude")                      # different latitude order: aligned by coordinate
w_da = metvane.latitude_weights(f_da.latitude)      # DataArray weights
mxr.rmse(f_da, o_da, preserve_dims="lead_time", weights=w_da)
```

Misspelled dimension names (e.g. `lat` vs `latitude`) raise; dask inputs give lazy results (`.compute()`).

### Categorical metrics (nowcasting)

```python
from metvane import ContingencyTable

radar_f = rng.uniform(0, 60, (6, 20, 64, 64))       # (case, lead, H, W) dBZ
radar_o = rng.uniform(0, 60, (6, 20, 64, 64))
radar_o[:, :, :8, :8] = np.nan                      # radar blind zone = missing, excluded from all counts

table = ContingencyTable(radar_f, radar_o, thresholds=[20, 35, 40], axis=(0, 2, 3))
table.csi().shape                                   # (3, 20): (threshold, lead)
scores = table.summary(["csi", "pod", "far", "frequency_bias"])
```

### Accumulators (split chunks only along reduced axes)

```python
accum = metvane.ContinuousAccumulator(["rmse", "mae", "bias"], preserve_axes=[1], weights=w)
for s in range(0, 4, 3):                            # chunks along init (axis 0, reduced)
    accum.update(fcst[s:s + 3], obs[s:s + 3])
res = accum.compute()                               # equals metvane.rmse(..., axis=(0, 2, 3))
np.testing.assert_allclose(res["rmse"], rmse, rtol=1e-5)
```

Splitting along a preserved axis is detected (kept-axis lengths must match across chunks).

## Scoring conventions

| Topic | MetVane convention |
|---|---|
| Missing data | NaN / MaskedArray mask / `mask=False` points are excluded from every statistic; with `skipna=False` slices containing NaN are NaN; `inf` is a value (a diverging model is not hidden) |
| Zero denominators | undefined → **NaN** (CSI without events, FSS with no events in either field, slices with zero weight) — skipped by `nanmean` |
| RMSE across samples | `sqrt(sample-mean MSE)` (WeatherBench convention), not the mean of per-sample RMSEs |
| ACC | uncentred by default; `axis` = spatial axes and `mean_over` = sample axes → per-sample spatial ACC averaged (WB2 / ECMWF); sample axes inside `axis` → pooled ACC; `centered=True` for centred ACC |
| Categorical | counts are exact int64, scores float64; prefer pooling tables (accumulators / sums) before computing scores |
| FSS | zero padding at edges; missing points handled by normalising with the valid count per window; pooled over fields as `1 − Σnum/Σden` (`fss_components` / `FSSAccumulator`), which differs from the mean of per-frame FSS |
| Latitude weights | `latitude_weights` has mean 1; when weighting yourself use `(w*e).sum()/w.sum()`, with the subset's own weight sum for regions; `method='area'` gives cell-area weights |
| `weight_mode` | `"mean"` (default) Σw·x/Σw; `"multiply"` Σw·x/N gives an area-contribution field when latitude is kept; use `"mean"` for regions, masks or missing data (a warning is issued otherwise) |
| Terms | FAR = false alarm **ratio** FP/(TP+FP); POFD = FP/(FP+TN) (`false_alarm_rate` is a deprecated alias); `metvane.bias` is the mean error, the frequency bias is `frequency_bias` |

### Full-grid (per-point) evaluation and `weight_mode`

```python
rmse_map = metvane.rmse(fcst, obs, axis=(0,))                        # (lead, lat, lon) per-point RMSE
contrib = metvane.rmse(fcst, obs, axis=(0,), weights=w, weight_mode="multiply")
# plain grid mean of contrib**2 == latitude-weighted MSE (global grid, mean-1 weights, no missing data)
np.testing.assert_allclose((contrib ** 2).mean(axis=(1, 2)),
                           metvane.mse(fcst, obs, axis=(0, 2, 3), weights=w), rtol=1e-4)
```

## Functional `axis` vs accumulators

| Goal | Functional | Accumulator |
|---|---|---|
| keep lead (axis 1) | `axis=(0, 2, 3)` | `preserve_axes=[1]` or `axis=(0, 2, 3)` |
| reduce everything | `axis=None` | default |
| no reduction (per point) | `axis=()` | `preserve_axes` covering all axes |

Categorical results have shape `(n_thresholds, *kept_axes)`.

## Metrics

### Top-level functions

| Function | Description |
|------|------|
| `rmse` / `mse` / `mae` / `bias` | continuous metrics (`bias` = mean error) |
| `continuous_scores` | several continuous metrics in one pass |
| `acc` | anomaly correlation (`mean_over`, `centered`) |
| `pearson_correlation` | weighted Pearson correlation |
| `wind_vector_rmse` | wind vector RMSE |
| `csi` / `pod` / `far` / `pofd` / `hss` / `ets` / `frequency_bias` / `f1` | single-threshold categorical shortcuts |
| `fss` / `fss_components` / `fss_from_components` | Fractions Skill Score and its poolable components |
| `crps_ensemble` (`fair=`) / `brier_score` (`threshold=`) | probabilistic (experimental) |

### `ContingencyTable` methods

`csi` (`threat_score`), `pod` (`hit_rate`), `far` (`false_alarm_ratio`), `pofd`, `hss`, `ets` (`gilbert_skill_score`),
`bias_score` (`frequency_bias`), `f1`, `pc` (`proportion_correct`, formerly `accuracy`), `summary(metrics)` (case-insensitive,
aliases accepted), `from_counts(tp, fp, fn, tn)`.

### Accumulators

`ContinuousAccumulator` (rmse/mse/mae/bias/acc/acc_mean), `ContingencyAccumulator`, `FSSAccumulator`.

## Tests and examples

```bash
cd MetVane
PYTHONPATH=src python -m pytest tests            # CPU (CUDA tests are skipped)
sbatch scripts/run_gpu_tests.sh                   # full suite incl. CUDA tests on a GPU node
PYTHONPATH=src python scripts/coverage.py --fail-under 90   # line coverage (stdlib only)
python examples/example_medium_range.py
python examples/example_nowcasting.py
python examples/example_gpu.py
python examples/example_xarray.py
```

See [CHANGELOG.md](CHANGELOG.md) for changes.

## License

Apache License 2.0
