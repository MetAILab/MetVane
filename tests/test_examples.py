"""The shipped examples must run and their internal assertions (accumulator == functional) hold."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["example_medium_range", "example_nowcasting", "example_xarray"])
def test_example_runs(name):
    _load(name).main()


def test_example_gpu_on_cpu():
    pytest.importorskip("torch")
    _load("example_gpu").main(device="cpu")
