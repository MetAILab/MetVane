# MetVane

**Meteorological Verification and Numerical forecast Evaluation**

A unified evaluation framework for medium-range weather forecasting and nowcasting.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Language**: **English** | [中文](README.md)

## Features

- **Multi-backend support**: NumPy / PyTorch / xarray — automatic format detection, zero-code switching
- **GPU acceleration**: PyTorch tensors computed on GPU automatically, no extra configuration needed
- **Array namespace adapter**: A single codebase for both NumPy and PyTorch, eliminating code duplication
- **Four metric categories**: Continuous (regression), categorical, spatial, and probabilistic
- **Flexible aggregation**: Overall, per-lead-time, or region-specific evaluation
- **Accumulator mode**: `update/compute` pattern for chunk-wise processing of large datasets
- **xarray convenience layer**: Named-dimension aggregation via `preserve_dims` / `reduce_dims`
- **Meteorological utilities**: Latitude weighting, region masking, weight broadcasting

## Installation

```bash
# Basic install (numpy only)
pip install -e .

# With PyTorch support
pip install -e ".[torch]"

# With xarray support
pip install -e ".[xarray]"

# Full dependencies
pip install -e ".[all]"

# Development mode (includes test dependencies)
pip install -e ".[dev]"
```

## Quick Start

### Functional API

```python
import numpy as np
import metvane

fcst = np.random.randn(10, 181, 360).astype(np.float32)
obs  = np.random.randn(10, 181, 360).astype(np.float32)
lat  = np.linspace(-90, 90, 181, dtype=np.float32)

# Latitude-weighted RMSE, per lead time
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
# Result stays on GPU automatically
```

### xarray Named Dimensions

```python
import xarray as xr
import metvane.xr_api as mxr

fcst = xr.DataArray(data, dims=["lead_time", "lat", "lon"])
obs  = xr.DataArray(data, dims=["lead_time", "lat", "lon"])

rmse = mxr.rmse(fcst, obs, preserve_dims="lead_time")
# Automatically reduces lat and lon, preserves lead_time
```

### Categorical Metrics (Nowcasting)

```python
import metvane
from metvane.categorical import ContingencyTable

table = ContingencyTable(fcst, obs, thresholds=[20, 35, 40], axis=(1, 2))
print(table.csi())      # shape: (3, n_leadtimes)
print(table.pod())
print(table.summary())   # dict of all scores
```

### Accumulator Mode (Large Datasets)

```python
acc = metvane.ContinuousAccumulator(
    ["rmse", "mae", "bias"],
    preserve_axes=[0],
)
for chunk_f, chunk_o in data_loader:
    acc.update(chunk_f, chunk_o)

result = acc.compute()  # {"rmse": array, "mae": array, "bias": array}
```

## Available Metrics

### Continuous (Regression) — `metvane.continuous`

| Function | Description |
|----------|-------------|
| `rmse` | Root Mean Squared Error |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `bias` | Additive bias (fcst − obs) |
| `acc` | Anomaly Correlation Coefficient |
| `pearson_correlation` | Weighted Pearson correlation |
| `wind_vector_rmse` | Wind vector RMSE |

### Categorical — `metvane.categorical`

| Function / Method | Description |
|-------------------|-------------|
| `ContingencyTable` | Multi-threshold contingency table |
| `csi` / `threat_score` | Critical Success Index |
| `pod` / `hit_rate` | Probability of Detection |
| `far` | False Alarm Ratio |
| `hss` | Heidke Skill Score |
| `ets` | Equitable Threat Score |
| `bias_score` | Frequency Bias |
| `f1` | F1 Score |
| `accuracy` | Overall accuracy |

### Spatial — `metvane.spatial`

| Function | Description |
|----------|-------------|
| `fss` | Fractions Skill Score |

### Probabilistic — `metvane.probabilistic`

| Function | Description |
|----------|-------------|
| `crps_ensemble` | Continuous Ranked Probability Score (ensemble) |
| `brier_score` | Brier Score |


## Running Tests

```bash
cd metvane
pip install -e ".[dev]"
pytest tests/ -v
```

## Running Examples

```bash
cd metvane
python examples/example_medium_range.py
python examples/example_nowcasting.py
python examples/example_gpu.py
python examples/example_xarray.py
```

## License

Apache License 2.0
