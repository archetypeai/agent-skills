"""Synthetic end-to-end tests for the Omega Encoder agent data-prep scripts, and role files
broken on purpose. Offline: no API key, no network.

    pytest skills/atai-omega-encoder-agent-data-prep/tests/ -v
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
sys.path.insert(0, REF)
import build_roles  # noqa: E402
import prepare  # noqa: E402
import preflight_roles  # noqa: E402

STEP = 0.01
CH = ["torque", "force", "acc"]


def segment(n=1400, onset=400, start=0.0, jitter=0.0, gap_at=None, back_at=None, seed=0):
    rng = np.random.default_rng(seed)
    t = start + np.arange(n) * STEP + rng.uniform(-jitter, jitter, n)
    if gap_at is not None:
        t[gap_at:] += 5.0                         # a 5 s pause: not one take
    if back_at is not None:
        t[back_at:] -= 3.0                        # the clock jumps back
    x = rng.normal(0, 0.1, (n, len(CH)))
    x[onset:onset + 300, 0] += 1.0                # the torque acts from the onset
    return pd.DataFrame({"timestamp": t, **{c: x[:, i] for i, c in enumerate(CH)}})


def run(argv, main):
    old = sys.argv
    sys.argv = ["x", *argv]
    try:
        return main()
    finally:
        sys.argv = old


@pytest.fixture
def dataset(tmp_path):
    """Library (normal, 2 groups), validation and test with both classes, a delivery;
    every recording's clock starts at 0, as recorders restart per session."""
    rows = []
    spec = [("library", "g0", "normal", 6), ("library", "g1", "normal", 6),
            ("validation", "g2", "normal", 3), ("validation", "g2", "fault", 2),
            ("test", "g3", "normal", 3), ("test", "g3", "fault", 2),
            ("delivery", "g4", "normal", 2), ("delivery", "g4", "fault", 1)]
    i = 0
    for role, group, label, n in spec:
        for _ in range(n):
            d = segment(seed=i, jitter=0.0005)
            if label == "fault":
                d["torque"] *= 0.7                 # weaker, still over the onset level
            name = f"s{i:03d}"
            d.to_csv(tmp_path / f"{name}.csv", index=False)
            rows.append({"segment": name, "file": f"{name}.csv", "role": role, "group": group, "label": label})
            i += 1
    pd.DataFrame(rows).to_csv(tmp_path / "index.csv", index=False)
    return tmp_path


def prep(ds, rows=1024, windows=(1024, 512, 256)):
    assert run(["--index", str(ds / "index.csv"), "--out", str(ds / "prepared"), "--rows", str(rows),
                "--onset-channel", "torque", "--onset-above", "0.5", "--pre-rows", "256"], prepare.main) == 0
    assert run(["--prepared", str(ds / "prepared"), "--out", str(ds / "roles"),
                "--windows", *map(str, windows)], build_roles.main) == 0
    return ds / "roles"


def test_cut_aligns_on_the_onset_and_regrids():
    d = segment(onset=500, jitter=0.0005)
    w, why, offset = prepare.cut(d, 1024, "torque", 0.5, 256, 10)
    assert why == "" and len(w) == 1024
    assert np.allclose(np.diff(w["timestamp"]), STEP)
    assert w["torque"].iloc[256] > 0.5 > w["torque"].iloc[:250].max()   # onset sits at row 256
    assert offset < 1


def test_cut_drops_gaps_backsteps_and_short_segments():
    assert "gap" in prepare.cut(segment(gap_at=700), 1024, None, 0, 0, 10)[1]
    assert "back" in prepare.cut(segment(back_at=700), 1024, None, 0, 0, 10)[1]
    assert "too short" in prepare.cut(segment(n=900), 1024, None, 0, 0, 10)[1]
    assert "never above" in prepare.cut(segment(), 1024, "torque", 99, 0, 10)[1]


def test_step_override_for_heavy_jitter():
    d = segment(jitter=0.003)                                  # 30% of a step
    w, why, _ = prepare.cut(d, 1024, None, 0, 0, 10, nominal=STEP)
    assert why == "" and np.allclose(np.diff(w["timestamp"]), STEP)


def test_a_skipped_sample_is_absorbed_not_dropped():
    d = segment()
    d = d.drop(index=600).reset_index(drop=True)              # one sample missing: a 0.02 s step
    w, why, offset = prepare.cut(d, 1024, None, 0, 0, 10)
    assert why == "" and 0.5 < offset < 1.5


def test_end_to_end_passes_preflight(dataset):
    roles = prep(dataset)
    assert preflight_roles.check(str(roles), k=5) == 0
    m = json.load(open(roles / "manifest.json"))
    assert m["rows"] == 1024 and m["candidate_windows"] == [1024, 512, 256]
    assert m["rebased_timestamps"] is True                    # every clock started at 0
    assert len({f["start"] for f in m["files"]}) == len(m["files"])
    lib = pd.concat(pd.read_csv(roles / f["path"]) for f in m["files"] if f["role"] == "library")
    assert abs(lib[CH].mean()).max() < 1e-3 and abs(lib[CH].std() - 1).max() < 1e-2   # z-scored on the library
    val = json.load(open(roles / "labels" / "validation.json"))["operations"]
    assert {o["class"] for o in val} == {"normal", "fault"}
    assert not (roles / "labels" / "library.json").exists()
    one = pd.read_csv(roles / m["files"][0]["path"])
    assert list(one.columns) == ["timestamp", *CH]


def test_windows_that_do_not_divide_the_rows_are_refused(dataset):
    with pytest.raises(SystemExit):
        prep(dataset, rows=1000, windows=(1000, 256))


def test_a_labelled_library_segment_is_refused(dataset):
    idx = pd.read_csv(dataset / "index.csv")
    idx.loc[0, "label"] = "fault"
    idx.to_csv(dataset / "index.csv", index=False)
    with pytest.raises(SystemExit):
        prep(dataset)


@pytest.mark.parametrize("breakage", ["label_column", "short_file", "shared_start", "group_in_two_roles"])
def test_preflight_fails_on_broken_role_files(dataset, breakage):
    roles = prep(dataset)
    m = json.load(open(roles / "manifest.json"))
    f0 = roles / m["files"][0]["path"]
    if breakage == "label_column":
        d = pd.read_csv(f0)
        d["label"] = 0
        d.to_csv(f0, index=False)
    elif breakage == "short_file":
        pd.read_csv(f0).iloc[:1000].to_csv(f0, index=False)
    elif breakage == "shared_start":
        m["files"][1]["start"] = m["files"][0]["start"]
        shutil.copy(f0, roles / m["files"][1]["path"])
    elif breakage == "group_in_two_roles":
        m["groups"]["test"].append(m["groups"]["library"][0])
    json.dump(m, open(roles / "manifest.json", "w"))
    assert preflight_roles.check(str(roles), k=5) == 1
