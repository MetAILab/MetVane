"""MetVane — A unified evaluation framework for weather forecasting and nowcasting.

Quick start
-----------
>>> import numpy as np, metvane
>>> lat = np.linspace(90, -90, 181)
>>> fcst, obs = np.random.randn(2, 4, 181, 360)
>>> w = metvane.latitude_weights(lat)[:, None]          # (nlat, 1): broadcasts along latitude
>>> metvane.rmse(fcst, obs, axis=(1, 2), weights=w)     # one value per sample

Three ways to use MetVane
~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Functional API (numpy / torch)** — positional axes, identical shapes::

       metvane.rmse(fcst, obs, axis=(0, 2, 3), weights=w)

2. **xarray layer** — named dims, coordinate alignment, dask-lazy::

       import metvane.xr_api as mxr
       mxr.rmse(fcst_da, obs_da, preserve_dims="lead_time", weights=w_da)

3. **Chunked accumulators** — split chunks only along reduced axes::

       acc = metvane.ContinuousAccumulator(["rmse"], preserve_axes=[1], weights=w)
       for f, o in loader:              # (n_init_chunk, n_lead, lat, lon)
           acc.update(f, o)
       acc.compute()
"""

__version__ = "0.2.0"

# --- continuous (regression) metrics ---
from .continuous import (rmse, mse, mae, bias, acc, pearson_correlation, wind_vector_rmse,
                         continuous_scores)

# --- categorical metrics ---
from .categorical import ContingencyTable, csi, pod, far, pofd, hss, ets, frequency_bias, f1

# --- spatial metrics ---
from .spatial import fss, fss_components, fss_from_components

# --- probabilistic metrics (experimental) ---
from .probabilistic import crps_ensemble, brier_score

# --- accumulators ---
from .accumulator import ContinuousAccumulator, ContingencyAccumulator, FSSAccumulator

# --- weighting utilities ---
from .core.weighting import latitude_weights, broadcast_weights, region_mask, REGIONS

# --- backend utilities ---
from .core.backend import Backend, detect_backend, convert

# --- xarray layer available as metvane.xr_api (imports xarray lazily) ---
from . import xr_api

__all__ = [
    # continuous
    "rmse", "mse", "mae", "bias", "acc", "pearson_correlation", "wind_vector_rmse", "continuous_scores",
    # categorical
    "ContingencyTable", "csi", "pod", "far", "pofd", "hss", "ets", "frequency_bias", "f1",
    # spatial
    "fss", "fss_components", "fss_from_components",
    # probabilistic
    "crps_ensemble", "brier_score",
    # accumulators
    "ContinuousAccumulator", "ContingencyAccumulator", "FSSAccumulator",
    # weighting
    "latitude_weights", "broadcast_weights", "region_mask", "REGIONS",
    # backend
    "Backend", "detect_backend", "convert",
    # xarray
    "xr_api",
]
