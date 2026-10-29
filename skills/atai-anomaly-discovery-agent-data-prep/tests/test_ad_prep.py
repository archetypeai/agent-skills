"""Synthetic end-to-end tests for the AD data-prep scripts, and role files broken on purpose.

    pytest skills/atai-anomaly-discovery-agent-data-prep/tests/ -v
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
sys.path.insert(0, REF)
import build_roles  # noqa: E402
import check_baseline  # noqa: E402
import prepare  # noqa: E402
import preflight_roles  # noqa: E402

STEP = 60_000            # 1 minute
CH = ["a", "b", "c"]


def recording(path, start_s, n, onset=None, shift_after=None, jitter=False, gap_at=None, seed=0):
    rng = np.random.default_rng(seed)
    t = start_s + np.arange(n) * STEP / 1000
    if jitter:
        t = t + rng.uniform(-0.2, 0.2, n)
    if gap_at:
        t[gap_at:] += 3600                       # a one-hour gap: split, never interpolated
    x = rng.normal(0, 1, (n, len(CH)))
    if shift_after is not None:
        x[shift_after:] += 3.0                   # a level shift: drift between halves
    lab = np.array(["normal"] * n, dtype=object)
    if onset is not None:
        lab[onset:] = "anomaly"
        x[onset:] += np.linspace(0, 6, n - onset)[:, None]
    d = pd.DataFrame({"timestamp": t, **{c: x[:, i] for i, c in enumerate(CH)}, "label": lab})
    d.to_csv(path, index=False)


@pytest.fixture()
def built(tmp_path):
    rows = []
    for i in range(6):
        recording(tmp_path / f"lib{i}.csv", 1_700_000_000 + i * 86400, 300, jitter=i == 0, gap_at=150 if i == 1 else None, seed=i)
        rows.append((f"lib{i}", f"lib{i}.csv", "library", f"g{i}", "normal"))
    k = 10
    for role in ("validation", "test", "delivery"):
        recording(tmp_path / f"{role}_n.csv", 1_700_000_000 + k * 86400, 300, seed=k)
        rows.append((f"{role}_n", f"{role}_n.csv", role, f"g{k}", "normal"))
        k += 1
        for f in ("f1", "f2"):
            recording(tmp_path / f"{role}_{f}.csv", 1_700_000_000 + k * 86400, 300, onset=100, seed=k)
            rows.append((f"{role}_{f}", f"{role}_{f}.csv", role, f"g{k}", f))
            k += 1
    pd.DataFrame(rows, columns=["recording", "file", "role", "group", "type"]).to_csv(tmp_path / "recordings.csv", index=False)
    idx, prep, roles = str(tmp_path / "recordings.csv"), str(tmp_path / "prepared"), str(tmp_path / "roles")
    prepare.main(["--index", idx, "--out", prep, "--step-ms", str(STEP), "--min-rows", "50", "--workers", "1"])
    build_roles.main(["--index", idx, "--prepared", prep, "--out", roles, "--per-file", "2"])
    return tmp_path, idx, prep, roles


def preflight(roles, *extra):
    return preflight_roles.main(["--roles", roles, *extra])


def test_prepare_grid_and_gap(built):
    _, _, prep, _ = built
    rep = json.load(open(os.path.join(prep, "prepare_report.json")))
    r1 = next(r for r in rep["recordings"] if r["recording"] == "lib1")
    assert len(r1["segments"]) == 2                     # split at the hour gap
    d0 = pd.read_parquet(os.path.join(prep, "lib0.parquet"))
    assert (np.diff(d0.timestamp) == STEP).all()        # jitter removed: an exact grid


def test_whole_flow_passes(built):
    _, idx, prep, roles = built
    assert check_baseline.main(["--index", idx, "--prepared", prep, "--window", "10", "--step", "5",
                                "--k", "5", "--pca", "3"]) == 0
    assert preflight(roles, "--prepared", prep, "--index", idx, "--window", "10", "--step", "5", "--k", "5", "--pca", "3") == 0
    m = json.load(open(os.path.join(roles, "manifest.json")))
    assert len(m["library"]) == 7                       # lib1 split into two segments
    assert any(f.get("calibration") for f in m["delivery"])
    assert "label" not in pd.read_csv(os.path.join(roles, m["delivery"][0]["file"]), nrows=1)
    assert "label" in pd.read_csv(os.path.join(roles, m["test"][0]["file"]), nrows=1)


def test_baseline_flags_impure_library(tmp_path):
    recording(tmp_path / "x.csv", 1_700_000_000, 200, onset=150)
    pd.DataFrame([("x", "x.csv", "library", "", "")], columns=["recording", "file", "role", "group", "type"]).to_csv(
        tmp_path / "r.csv", index=False)
    prepare.main(["--index", str(tmp_path / "r.csv"), "--out", str(tmp_path / "p"), "--step-ms", str(STEP),
                  "--min-rows", "50", "--workers", "1"])
    assert check_baseline.main(["--index", str(tmp_path / "r.csv"), "--prepared", str(tmp_path / "p"),
                                "--window", "10", "--step", "5", "--k", "5", "--pca", "3"]) == 1


def test_baseline_warns_on_drift(tmp_path, capsys):
    rows = []
    for i in range(6):
        recording(tmp_path / f"l{i}.csv", 1_700_000_000 + i * 86400, 300, shift_after=150 if i == 0 else None, seed=i)
        rows.append((f"l{i}", f"l{i}.csv", "library", "", ""))
    pd.DataFrame(rows, columns=["recording", "file", "role", "group", "type"]).to_csv(tmp_path / "r.csv", index=False)
    prepare.main(["--index", str(tmp_path / "r.csv"), "--out", str(tmp_path / "p"), "--step-ms", str(STEP),
                  "--min-rows", "50", "--workers", "1"])
    check_baseline.main(["--index", str(tmp_path / "r.csv"), "--prepared", str(tmp_path / "p"),
                         "--window", "10", "--step", "5", "--k", "5", "--pca", "3"])
    out = capsys.readouterr().out
    assert "WARN  stability" in out and "l0:" in out


def test_coverage_fails_when_library_is_thin(built):
    _, idx, prep, roles = built
    assert preflight(roles, "--window", "200", "--step", "200", "--k", "20", "--pca", "20") == 1


def _rewrite(roles, rel, fn):
    p = os.path.join(roles, rel)
    d = pd.read_csv(p, dtype={"timestamp": str})
    fn(d).to_csv(p, index=False)


def test_breaks(built):
    _, _, _, roles = built
    m = json.load(open(os.path.join(roles, "manifest.json")))
    test_file = next(f for f in m["test"] if f["anomaly"])["file"]
    lib_file = m["library"][0]["file"]
    deliv = next(f for f in m["delivery"] if not f.get("calibration"))
    cases = [
        ("label in a library file", lib_file, lambda d: d.assign(label="normal")),
        ("label in a delivery file", deliv["file"], lambda d: d.assign(label="normal")),
        ("a jump inside a library file", lib_file,
         lambda d: d.assign(timestamp=[f"{float(t) + (7200 if i > 20 else 0):.3f}" for i, t in enumerate(d.timestamp)])),
        ("bad timestamp format", test_file, lambda d: d.assign(timestamp=d.timestamp.str.replace(r"\.\d+$", "", regex=True))),
        ("label disagrees with sidecar", test_file, lambda d: d.assign(label="normal")),
        ("unknown label", test_file, lambda d: d.assign(label=d.label.replace("anomaly", "fault"))),
    ]
    for name, rel, fn in cases:
        backup = open(os.path.join(roles, rel)).read()
        _rewrite(roles, rel, fn)
        assert preflight(roles) == 1, name
        open(os.path.join(roles, rel), "w").write(backup)
    assert preflight(roles) == 0


def test_missing_class_and_two_roles(built):
    _, _, _, roles = built
    mp = os.path.join(roles, "manifest.json")
    m = json.load(open(mp))
    keep = json.dumps(m)
    m["test"] = [f for f in m["test"] if f["anomaly"] == 0]      # test without incidents
    json.dump(m, open(mp, "w"))
    assert preflight(roles) == 1
    m = json.loads(keep)
    m["test"][0]["recordings"].append(m["library"][0]["recording"])   # a recording in two roles
    json.dump(m, open(mp, "w"))
    assert preflight(roles) == 1
