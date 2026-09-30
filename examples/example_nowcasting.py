"""Example: radar nowcasting evaluation.

Data layout ``(n_case, n_lead, H, W)`` in dBZ (0-60), 20 lead times.  Shows CSI/POD/FAR/HSS per
threshold and lead time, handling of missing (NaN) radar pixels, pooled FSS and chunked
accumulation along the case axis.
"""

import numpy as np

import metvane
from metvane import ContingencyTable


def _simulate(rng, n_case, n_lead, h, w):
    base = rng.uniform(0, 60, (n_case, 1, h, w)).astype(np.float32)
    obs = np.clip(base + rng.normal(0, 3, (n_case, n_lead, h, w)).astype(np.float32), 0, 60)
    err = np.linspace(1, 10, n_lead, dtype=np.float32)[None, :, None, None]
    fcst = np.clip(obs + err * rng.standard_normal(obs.shape).astype(np.float32), 0, 65)
    obs[:, :, :8, :8] = np.nan                      # radar blind zone: missing, not "no echo"
    return fcst, obs


def main():
    rng = np.random.default_rng(42)
    n_case, n_lead, h, w = 6, 20, 64, 64
    fcst, obs = _simulate(rng, n_case, n_lead, h, w)
    thresholds = [20.0, 35.0, 40.0]

    # 1. per-lead scores: reduce case, H, W; NaN pixels are excluded from all counts
    table = ContingencyTable(fcst, obs, thresholds, axis=(0, 2, 3))
    scores = table.summary(["csi", "pod", "far", "hss", "frequency_bias"])
    assert scores["csi"].shape == (len(thresholds), n_lead)
    for name, vals in scores.items():
        print(f"{name:15s} lead0: {np.round(vals[:, 0], 3)}  lead19: {np.round(vals[:, -1], 3)}")

    # 2. accumulator: chunks along the case axis, keep lead (axis 1)
    ca = metvane.ContingencyAccumulator(thresholds, preserve_axes=[1])
    for start in range(0, n_case, 4):               # uneven chunks 4/2
        ca.update(fcst[start:start + 4], obs[start:start + 4])
    res = ca.compute(["csi", "pod", "far"])
    assert res["csi"].shape == (len(thresholds), n_lead)
    np.testing.assert_array_equal(ca.as_contingency_table().tp, table.tp)
    np.testing.assert_allclose(res["csi"], scores["csi"])
    print("ContingencyAccumulator == ContingencyTable ✓")

    # 3. FSS pooled over cases per lead (1 - Σnum/Σden); NaN windows handled
    fa = metvane.FSSAccumulator([35.0], [1, 5, 11], preserve_axes=[1])
    fa.update(fcst, obs)
    fss = fa.compute()["fss"]                       # (n_thr, n_win, n_lead)
    print("FSS ≥35 dBZ lead 0, windows 1/5/11:", np.round(fss[0, :, 0], 3))
    one = metvane.fss(fcst[:, 0], obs[:, 0], 35.0, window_size=5)
    np.testing.assert_allclose(fss[0, 1, 0], one)
    return {"table": table, "fss": fss}


if __name__ == "__main__":
    main()
