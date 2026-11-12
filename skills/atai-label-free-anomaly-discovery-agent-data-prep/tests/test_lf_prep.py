"""Synthetic tests for the Label-Free AD data-prep scripts, plus a run on the shipped sample.

    pytest skills/atai-label-free-anomaly-discovery-agent-data-prep/tests/ -v
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
sys.path.insert(0, REF)
import lf_prep  # noqa: E402
import preflight  # noqa: E402
import preflight_raw  # noqa: E402
import prepare  # noqa: E402

SAMPLE = os.path.join(REF, "sample_data", "metropt3_2020-07-08_2020-07-16.csv.gz")
STEP = 10


def raw(path, n=40_000, jitter=True, gap_at=None, frozen_at=None, label=False, seed=0):
    """n rows ~every 10 s: two analog channels, one 0/1 switch, optionally a gap and a frozen stretch."""
    rng = np.random.default_rng(seed)
    t = 1_600_000_000 + np.arange(n) * STEP + (rng.uniform(-1, 1, n) if jitter else 0)
    if gap_at:
        t[gap_at:] += 3 * 3600                                  # a 3-hour gap
    x = rng.normal(0, 1, (n, 2))
    if frozen_at:
        x[frozen_at:frozen_at + 400] = x[frozen_at]              # ~67 min of every channel unchanged
    d = pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                      "a": x[:, 0], "b": x[:, 1], "switch": (np.arange(n) // 50) % 2})
    if label:
        d["label"] = "normal"
    d.to_csv(path, index=False)
    return path


def test_channels_default_drops_binary(tmp_path):
    df = lf_prep.read_raw([raw(tmp_path / "r.csv", n=500)], "timestamp")
    keep, out = lf_prep.pick_channels(df, "timestamp", None)
    assert keep == ["a", "b"] and out == ["switch"]
    assert lf_prep.pick_channels(df, "timestamp", "a")[0] == ["a"]
    with pytest.raises(SystemExit):
        lf_prep.pick_channels(df, "timestamp", "nope")


def test_grid_is_exact_and_never_crosses_a_gap(tmp_path):
    df = lf_prep.read_raw([raw(tmp_path / "r.csv", n=3000, gap_at=1500)], "timestamp")
    g = lf_prep.grid(df, "timestamp", ["a", "b"], STEP, max_fill=5)
    dt = np.diff(lf_prep.epoch_s(g.index))
    assert (dt % STEP == 0).all()
    assert (dt > STEP).sum() == 1                                # one gap, kept
    assert dt.max() >= 3 * 3600


def test_frozen_stretch_is_cut(tmp_path):
    df = lf_prep.read_raw([raw(tmp_path / "r.csv", n=5000, jitter=False, frozen_at=2000)], "timestamp")
    g = lf_prep.grid(df, "timestamp", ["a", "b"], STEP, max_fill=5)
    m = lf_prep.frozen_mask(g, STEP, 180)
    assert 380 <= m.sum() <= 400
    assert not lf_prep.frozen_mask(g[~m], STEP, 180).any()


def test_epoch_millis_are_recognised(tmp_path):
    p = tmp_path / "ms.csv"
    pd.DataFrame({"timestamp": 1_600_000_000_000 + np.arange(100) * 10_000, "a": np.arange(100.0)}).to_csv(p, index=False)
    df = lf_prep.read_raw([str(p)], "timestamp")
    assert df.timestamp.iloc[0] == pd.Timestamp(1_600_000_000, unit="s")


def test_prepare_splits_by_days_and_drops_labels(tmp_path):
    src = raw(tmp_path / "r.csv", n=40_000, label=True)          # ~4.6 days
    out = tmp_path / "prepared"
    assert prepare.main(["--input", str(src), "--step-s", "10", "--split-days", "2", "--out", str(out)]) == 0
    rep = json.loads((out / "prepare_report.json").read_text())
    assert len(rep["files"]) == 3 and rep["channels"] == ["a", "b"]
    for f in rep["files"]:
        d = pd.read_csv(out / f["file"])
        assert list(d.columns) == ["timestamp", "a", "b"]
        assert (np.diff(d.timestamp) % 10 == 0).all()


def test_preflight_passes_a_good_file_and_fails_a_short_one(tmp_path):
    out = tmp_path / "prepared"
    prepare.main(["--input", str(raw(tmp_path / "r.csv", n=40_000)), "--step-s", "10", "--out", str(out)])
    assert preflight.main(["--prepared", str(out), "--window", "360"]) == 0
    short = tmp_path / "short"
    prepare.main(["--input", str(raw(tmp_path / "s.csv", n=5_000)), "--step-s", "10", "--out", str(short)])
    assert preflight.main(["--prepared", str(short), "--window", "360"]) == 1     # 13 windows: too few
    assert preflight.main(["--prepared", str(out), "--window", "2000"]) == 1      # over Omega's 1,024


def test_preflight_raw_warns_but_passes(tmp_path):
    src = raw(tmp_path / "r.csv", n=5000, gap_at=2500, frozen_at=1000, label=True)
    assert preflight_raw.main(["--input", str(src), "--step-s", "10"]) == 0


def test_k_and_the_hiding_limit():
    assert preflight.k_of(375) == 75 and preflight.k_of(500) == 100 and preflight.k_of(10) == 3


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample not shipped")
def test_shipped_sample_end_to_end(tmp_path):
    out = tmp_path / "prepared"
    assert preflight_raw.main(["--input", SAMPLE, "--step-s", "10"]) == 0
    assert prepare.main(["--input", SAMPLE, "--step-s", "10", "--out", str(out)]) == 0
    rep = json.loads((out / "prepare_report.json").read_text())
    assert rep["channels"] == ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
    assert rep["files"][0]["rows"] == 56_449
    assert preflight.main(["--prepared", str(out), "--window", "360"]) == 0
    n, valid = lf_prep.window_counts(pd.read_csv(out / rep["files"][0]["file"]).timestamp.to_numpy(), 10, 360)
    assert (n, valid) == (156, 142)
