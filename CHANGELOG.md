# Changelog

## 0.2.0 — 2026-09

Fixes all findings of the 2026-09 code review.  **Many results change on purpose**; inputs
without NaN, with identical shapes and coordinates and without zero denominators give the
same results as 0.1.0 (up to floating-point rounding).

### Breaking changes and migration

| Area | 0.1.0 | 0.2.0 | Migration |
|---|---|---|---|
| Categorical / FSS with NaN | NaN treated as "no event" (counted as FP/FN/TN) | NaN (and masked elements, `mask=False`) excluded from all counts; `skipna=False` → NaN | none — results become correct; pass `mask=` for explicit evaluation regions |
| Zero denominators in scores | `+1e-8` → 0 or ~1e8 | **NaN** (CSI/POD/FAR/POFD/HSS/ETS/bias/F1/PC) | use `np.nanmean`, or pool counts before scoring |
| HSS/ETS | `+1e-8` bias | exact integer forms, NaN when undefined | — |
| FSS | returned 1.0 when neither field has events; 2-D only; any window accepted | NaN (`no_event_value=`); `(..., H, W)` pooled via `axis=`; window must be a positive odd int; returns a 0-d array on the input backend | use `no_event_value=1.0` for the old value |
| Counts dtype | float (float32 on torch, inexact beyond 2²⁴) | int64 | — |
| Accumulator state | float32 on torch; chunks split along preserved axes silently added | float64; kept-axis lengths checked (`ValueError`) | split chunks along reduced axes only |
| Accumulator NaN | one NaN poisoned the slice | `skipna=True` as the functional API | — |
| Shapes | `fcst`/`obs` broadcast silently (`(N,)` vs `(N,1)` → N² counts) | must be identical (`ValueError`) | reshape explicitly |
| 1-D weights for N-D data | aligned with the last axis (longitude) | `ValueError` | `w[:, None]` or `broadcast_weights(w, shape, lat_axis=-2)` |
| Weights backend | numpy weights with torch data crashed | converted to the data's backend/device/dtype | — |
| float64 on torch | down-cast to float32 | kept | — |
| float16 on numpy | kept (overflow) | computed in float32 | — |
| `inf` with weights | dropped as missing | kept as a value (weighted == unweighted) | — |
| MaskedArray | mask dropped on the weighted path | masked = NaN everywhere | — |
| xarray in the functional API / accumulators | positional `.values` | `TypeError` → use `metvane.xr_api` | — |
| `xr_api` | positional pairing, `broadcast_like` on weights, eager `.values`, unknown dims ignored | aligned by name/coordinate (`join=`), xarray-native & dask-lazy, unknown dims raise, results named with units | fix dimension names |
| `region_mask` | mixed 0..360 / -180..180 conventions | modulo-360 matching, dateline boxes, 2-D lat/lon, torch output, empty built-in region raises | — |
| `device=` without `backend=` | ignored | `ValueError` | pass `backend="torch"` |
| `backend="xarray"` | returned numpy silently | `DeprecationWarning` (still numpy); will raise in 0.3; accumulators raise | use `metvane.xr_api` |
| `detect_backend` | depended on argument order | torch > xarray > numpy | — |
| `ContingencyTable.summary` | case-sensitive, no aliases, `[]` → all, key `accuracy` | case-insensitive, aliases, `[]` → `{}`, key `pc` | use `"pc"` (or `summary(["accuracy"])`) |
| `false_alarm_rate()` | silently POFD | `FutureWarning` (it is POFD) | use `pofd()` |
| `crps_ensemble` | O(M²), numpy only, `axis` in result numbering, obs shape unchecked | O(M log M), torch, `axis` numbered as in the ensemble input, obs shape checked, `fair=` | renumber `axis` |
| `brier_score` | no validation | range / 0-1 checks, `threshold=` | — |
| `latitude_weights` | no checks, DataArray → ndarray | |lat|>90 raises, radians warn, `method="area"`, `dtype=`, returns input type | — |
| `broadcast_weights` | `lat_axis` unchecked | range and length checked | — |

### Added

- `mask=` on all functional metrics, `ContingencyTable`, FSS and accumulator `update`.
- `skipna`, `mask`, `centered`, `mean_over` for `acc` / `pearson_correlation`
  (`mean_over` = per-sample ACC averaged over sample axes, the WeatherBench-2 / ECMWF convention).
- `continuous_scores` (several continuous metrics in one pass).
- `ContingencyTable.from_counts`, `.n`, `.pc()`; shortcuts `pofd`, `frequency_bias`, `f1`.
- `fss_components`, `fss_from_components`, `FSSAccumulator` (pooled FSS, FSS_useful, base rate).
- Accumulators: `axis=` (exclusive with `preserve_axes=`), `merge`, `state_dict` / `load_state_dict`,
  `stats`, atomic `update`, per-chunk `climatology=` / `weights=` / `mask=`, `acc_mean` metric.
- `xr_api`: `pearson_correlation`, `wind_vector_rmse`, `contingency_table`, `scores_from_counts`,
  `categorical_scores`, `align_inputs`; Dataset support for all metrics with `join_vars=` and
  per-variable weight dicts; `__all__`.
- Warnings: `op='>='` with threshold 0, degenerate contingency tables, `weight_mode="multiply"`
  where it no longer equals a weighted mean, static climatology broadcast along sample axes,
  non-uniform latitude spacing with `method="cos"`.
- Tests for every review item, golden values (Finley table, CRPS closed forms, FSS vs
  `scipy.ndimage.uniform_filter`), runnable examples and doc snippets, CUDA tests
  (`pytest -m gpu`, `scripts/run_gpu_tests.sh`).
- `py.typed`.
- Cached numpy → torch conversion of weights / climatology (`metvane.core.prepare.clear_conversion_cache`).
- `scripts/coverage.py`: stdlib line coverage with `--fail-under` (no coverage / pytest-cov dependency).

### Packaging

- Version single-sourced in `metvane.__version__` (`dynamic = ["version"]`).
- SPDX `license = "Apache-2.0"`, `setuptools>=77`; scipy dropped from runtime extras; netCDF4 moved
  to the `examples` extra; torch / xarray are only imported when used.
