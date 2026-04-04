"""MetVane — A unified evaluation framework for weather forecasting and nowcasting.

Quick start
-----------
>>> import metvane
>>> result = metvane.rmse(fcst, obs, axis=(2, 3), weights=metvane.latitude_weights(lat))

Three ways to use MetVane
~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Functional API (numpy core)**::

       metvane.rmse(fcst_np, obs_np, axis=(0, 2, 3))

2. **xarray convenience layer**::

       import metvane.xr_api as mxr
       mxr.rmse(fcst_da, obs_da, preserve_dims='lead_time')

3. **Chunked accumulators**::

       acc = metvane.ContinuousAccumulator(['rmse'], preserve_axes=[0])
       for chunk_f, chunk_o in loader:
           acc.update(chunk_f, chunk_o)
       result = acc.compute()
"""

__version__ = "0.1.0"

# --- continuous (regression) metrics ---
from .continuous import rmse, mse, mae, bias, acc, pearson_correlation, wind_vector_rmse

# --- categorical metrics ---
from .categorical import ContingencyTable, csi, pod, far, hss, ets

# --- spatial metrics ---
from .spatial import fss

# --- probabilistic metrics ---
from .probabilistic import crps_ensemble, brier_score

# --- accumulators ---
from .accumulator import ContinuousAccumulator, ContingencyAccumulator

# --- weighting utilities ---
from .core.weighting import latitude_weights, broadcast_weights, region_mask, REGIONS

# --- backend utilities ---
from .core.backend import Backend, detect_backend, convert

# --- xarray layer available as metvane.xr_api ---
from . import xr_api

__all__ = [
    # continuous
    "rmse", "mse", "mae", "bias", "acc", "pearson_correlation", "wind_vector_rmse",
    # categorical
    "ContingencyTable", "csi", "pod", "far", "hss", "ets",
    # spatial
    "fss",
    # probabilistic
    "crps_ensemble", "brier_score",
    # accumulators
    "ContinuousAccumulator", "ContingencyAccumulator",
    # weighting
    "latitude_weights", "broadcast_weights", "region_mask", "REGIONS",
    # backend
    "Backend", "detect_backend", "convert",
    # xarray
    "xr_api",
]
