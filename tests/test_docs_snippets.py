"""Every ```python block of README.md, README_en.md and docs/quickstart.md must run
(blocks of one file share a namespace, in order)."""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ["README.md", "README_en.md", "docs/quickstart.md"]


def _blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S)


@pytest.mark.parametrize("name", FILES)
def test_snippets_run(name):
    pytest.importorskip("torch")
    pytest.importorskip("xarray")
    ns: dict = {}
    blocks = _blocks(ROOT / name)
    assert blocks, name
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{name}[block {i}]", "exec"), ns)
        except Exception as e:  # pragma: no cover - failure path
            raise AssertionError(f"{name} block {i} failed: {e!r}\n{code}") from e
