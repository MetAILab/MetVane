#!/usr/bin/env python3
"""Stdlib line/branch-free coverage for metvane (no coverage/pytest-cov needed).

    PYTHONPATH=src python scripts/coverage.py [--fail-under 90] [pytest args...]

Traces executed lines of files under src/metvane while running pytest (all threads), and
compares them with the executable lines of every code object (``co_lines``).
"""
import argparse
import os
import sys
import threading
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "src", "metvane") + os.sep
hits: dict = {}


def _tracer(frame, event, arg):
    fn = frame.f_code.co_filename
    if not fn.startswith(PKG):
        return None
    lines = hits.setdefault(fn, set())

    def local(frame, event, arg):
        if event == "line":
            lines.add(frame.f_lineno)
        return local
    lines.add(frame.f_lineno)
    return local


def _executable(path):
    code = compile(open(path, encoding="utf-8").read(), path, "exec")
    out, stack = set(), [code]
    while stack:
        c = stack.pop()
        out |= {ln for _, _, ln in c.co_lines() if ln is not None}
        stack += [k for k in c.co_consts if isinstance(k, types.CodeType)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-under", type=float, default=90.0)
    a, rest = ap.parse_known_args()
    import pytest
    sys.settrace(_tracer)
    threading.settrace(_tracer)
    rc = pytest.main(["-q", "-p", "no:cacheprovider", *(rest or ["tests"])])
    sys.settrace(None)
    threading.settrace(None)
    tot = cov = 0
    rows = []
    for dp, _, fs in os.walk(PKG):
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                ex = _executable(p)
                hit = ex & hits.get(p, set())
                tot, cov = tot + len(ex), cov + len(hit)
                rows.append((os.path.relpath(p, ROOT), len(ex), len(hit), sorted(ex - hit)))
    for name, n, h, miss in sorted(rows):
        print(f"{name:48s} {h:5d}/{n:<5d} {100 * h / max(n, 1):6.1f}%  miss {miss[:12]}{'...' if len(miss) > 12 else ''}")
    pct = 100 * cov / max(tot, 1)
    print(f"TOTAL {cov}/{tot} = {pct:.1f}% (fail-under {a.fail_under})")
    if rc != 0:
        return int(rc)
    return 0 if pct >= a.fail_under else 2


if __name__ == "__main__":
    sys.exit(main())
