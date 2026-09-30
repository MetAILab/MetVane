"""Shared machinery of the chunked accumulators.

* reduction axes: ``axis=`` (axes to reduce, like the functional API) **or**
  ``preserve_axes=`` (axes to keep) — mutually exclusive; resolved on the first update
* chunks may only be split along **reduced** axes: the number of dimensions and the lengths
  of the kept axes are recorded on the first update and every later chunk must match
  (otherwise chunks would be silently added element-wise)
* ``update`` is atomic: all chunk statistics are computed first and committed together
* float64 / int64 state, ``merge``, ``state_dict`` / ``load_state_dict``, read-only ``stats``
* xarray inputs are rejected (positional computation would ignore coordinates)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.array_ns import get_namespace
from ..core.backend import Backend, _is_torch, convert, ensure_same_backend, reject_xarray
from ..core.prepare import complement_axes, normalize_axis, resolve_backend

STATE_VERSION = 1


class AccumulatorBase:
    """Base class (not public API)."""

    def __init__(self, *, axis: Any = None, preserve_axes: Any = None,
                 backend: Optional[str] = None, device: Any = None):
        if axis is not None and preserve_axes is not None:
            raise ValueError("axis= 与 preserve_axes= 互斥：axis 是要归约的轴，preserve_axes 是要保留的轴")
        if backend is None and device is not None:
            raise ValueError("只给了 device= 而没有 backend=：请同时指定 backend='torch'")
        if backend == "xarray":
            raise ValueError("累加器不支持 backend='xarray'；xarray 数据请先 xr.align(join='exact') 后取 .data，"
                             "或使用 metvane.xr_api")
        self._axis_arg = axis
        self._preserve_arg = None if preserve_axes is None else (
            (int(preserve_axes),) if isinstance(preserve_axes, (int, np.integer)) else tuple(preserve_axes))
        self.preserve_axes = self._preserve_arg
        self._forced_backend = backend
        self._device = device
        self.reset()

    # ------------------------------------------------------------------ config
    def _config(self) -> dict:
        return {"axis": self._axis_arg if self._axis_arg is None or isinstance(self._axis_arg, int)
                else tuple(self._axis_arg), "preserve_axes": self._preserve_arg}

    # ------------------------------------------------------------------ inputs
    def _inputs(self, *arrays):
        reject_xarray(*arrays, where=f"{type(self).__name__}.update")
        if self._forced_backend is not None:
            arrays = resolve_backend(arrays, self._forced_backend, self._device)
        arrays = ensure_same_backend(*arrays)
        for a in arrays:
            if _is_torch(a) and a.device.type == "mps":
                raise ValueError("MPS 设备不支持 float64 累加状态；请在 CPU/CUDA 上累加")
        return arrays

    def _reduced_axes(self, ndim: int) -> tuple[int, ...]:
        if self._ndim is not None and ndim != self._ndim:
            raise ValueError(f"数据块维数 {ndim} 与首块维数 {self._ndim} 不一致")
        if self._axis_arg is not None:
            ax = normalize_axis(self._axis_arg, ndim, name="axis")
            return tuple(range(ndim)) if ax is None else ax
        if self._preserve_arg is not None:
            keep = normalize_axis(self._preserve_arg, ndim, name="preserve_axes")
            return complement_axes(keep, ndim)
        return tuple(range(ndim))

    def _check_kept(self, shape: tuple, red: tuple[int, ...]) -> tuple[int, ...]:
        kept = tuple(int(shape[i]) for i in range(len(shape)) if i not in red)
        if self._kept_shape is not None and kept != self._kept_shape:
            keep_ax = complement_axes(red, len(shape))
            raise ValueError(f"保留轴 {keep_ax} 的长度 {kept} 与首块 {self._kept_shape} 不一致：保留轴不能跨块切分，"
                             f"只能沿被归约的轴（如起报时间/样本轴）分块")
        return kept

    # ------------------------------------------------------------------ state
    def _commit(self, chunk: dict, ndim: int, red: tuple, kept: tuple) -> None:
        if self._state is None:
            self._state = dict(chunk)
        else:
            xp = get_namespace(*chunk.values())
            for k, v in chunk.items():
                cur = self._state[k]
                if get_namespace(cur).name != xp.name or (xp.name == "torch" and cur.device != v.device):
                    cur = convert(cur, Backend(xp.name), device=getattr(v, "device", None))
                self._state[k] = cur + v
        self._ndim, self._red, self._kept_shape = ndim, red, kept

    def reset(self) -> None:
        """Clear all accumulated state (the axis configuration is re-resolved on the next update)."""
        self._state: Optional[dict] = None
        self._ndim: Optional[int] = None
        self._red: Optional[tuple] = None
        self._kept_shape: Optional[tuple] = None
        self._extra_reset()

    def _extra_reset(self) -> None:
        pass

    def _require_state(self) -> dict:
        if self._state is None:
            raise RuntimeError("No data accumulated（尚未累加任何数据），请先调用 update()")
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        """Read-only copy of the accumulated sufficient statistics."""
        st = self._require_state()
        return {k: (v.clone() if _is_torch(v) else np.array(v, copy=True)) for k, v in st.items()}

    def merge(self, other: "AccumulatorBase") -> "AccumulatorBase":
        """Add another accumulator's state (e.g. from another process); configs must match."""
        if type(other) is not type(self):
            raise TypeError(f"不能合并 {type(other).__name__} 到 {type(self).__name__}")
        if other._config() != self._config():
            raise ValueError(f"配置不一致，不能合并：{self._config()} vs {other._config()}")
        if other._state is None:
            return self
        if self._state is not None:
            if (self._ndim, self._kept_shape) != (other._ndim, other._kept_shape):
                raise ValueError(f"保留轴形状不一致：{self._kept_shape} vs {other._kept_shape}")
        self._commit(other._state, other._ndim, other._red, other._kept_shape)
        self._merge_extra(other)
        return self

    def _merge_extra(self, other) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state (numpy float64/int64 arrays + config + version)."""
        st = {} if self._state is None else {k: np.asarray(convert(v, Backend.NUMPY)) for k, v in self._state.items()}
        return {"version": STATE_VERSION, "class": type(self).__name__, "config": self._config(),
                "ndim": self._ndim, "reduced": self._red, "kept_shape": self._kept_shape, "state": st}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("version") != STATE_VERSION:
            raise ValueError(f"state_dict 版本 {d.get('version')} 不受支持（需要 {STATE_VERSION}）")
        if d.get("class") != type(self).__name__:
            raise ValueError(f"state_dict 属于 {d.get('class')}，不是 {type(self).__name__}")
        if d["config"] != self._config():
            raise ValueError(f"配置不一致：{d['config']} vs {self._config()}")
        self.reset()
        if d["state"]:
            self._state = {k: np.array(v, copy=True) for k, v in d["state"].items()}
            self._ndim = d["ndim"]
            self._red = None if d["reduced"] is None else tuple(d["reduced"])
            self._kept_shape = None if d["kept_shape"] is None else tuple(d["kept_shape"])
