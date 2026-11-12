"""Synthetic end-to-end tests for the RED data-prep scripts, and role files broken on purpose.

    pytest skills/atai-rare-event-detection-agent-data-prep/tests/ -v
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
import check_events  # noqa: E402
import prepare  # noqa: E402
import preflight_roles  # noqa: E402

N = 12_000
# (start, length, raw label) of each incident, in time order; #3 and #4 are close together
INCIDENTS = [(1_000, 300, "BROKE"), (2_000, 400, "BROKE"), (6_000, 200, "BROKE"), (6_400, 150, "BROKE"),
             (9_000, 120, "BROKE")]
CLASSES = ["normal=OK", "breakdown=BROKE"]


def recording(path, incidents=INCIDENTS, gap_at=None, jitter=False):
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2024-01-01").value / 1e9 + 60.0 * np.arange(N)
    if jitter:
        t = t + rng.uniform(-5, 5, N)
    if gap_at:
        t[gap_at:] += 3600
    lab = np.array(["OK"] * N, dtype=object)
    x = rng.normal(0, 1, (N, 3))
    for lo, n, raw in incidents:
        lab[lo:lo + n] = raw
        x[lo:lo + n] += 4.0 if raw == "BROKE" else -4.0
    d = pd.DataFrame({"": np.arange(N), "timestamp": pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
                      "good_a": x[:, 0], "good_b": x[:, 1], "good_c": x[:, 2]})
    d["dropout"] = rng.normal(0, 1, N)                 # dark exactly during incidents
    for lo, n, _ in incidents:
        d.loc[lo:lo + n - 1, "dropout"] = np.nan
    d["dead"] = 1.0
    d["gappy"] = rng.normal(0, 1, N)
    d.loc[500:540, "gappy"] = np.nan                    # a 41-row missing run
    d["status"] = lab
    d.to_csv(path, index=False)


@pytest.fixture()
def prepared(tmp_path):
    recording(tmp_path / "rec.csv")
    prepare.main(["--input", str(tmp_path / "rec.csv"), "--label-col", "status", "--classes", *CLASSES,
                  "--out", str(tmp_path / "prepared")])
    return tmp_path


def build(tmp, *extra):
    return build_roles.main(["--prepared", str(tmp / "prepared"), "--out", str(tmp / "roles"), "--training", "1,2",
                             "--validation", "3", "--test", "4,5", "--delivery", "5", "--context", "1000",
                             "--delivery-context", "300", "--normal-rows", "800", "--top", "3", *extra])


def test_prepare_grid_incidents_and_gaps(tmp_path):
    recording(tmp_path / "rec.csv", gap_at=7_000, jitter=True)
    prepare.main(["--input", str(tmp_path / "rec.csv"), "--label-col", "status", "--classes", *CLASSES,
                  "--out", str(tmp_path / "p")])
    rep = json.load(open(tmp_path / "p" / "prepare_report.json"))
    d = pd.read_parquet(tmp_path / "p" / "recording.parquet")
    assert [i["n_rows"] for i in rep["incidents"]] == [300, 400, 200, 150, 120]
    assert rep["timestamps_snapped"] > N // 2 and len(rep["segments"]) == 2        # jitter snapped, gap kept
    steps = np.diff(d.timestamp.to_numpy())
    assert set(steps) == {60, 3660}
    assert rep["dropped_columns"] == ["Unnamed: 0"]                                  # the unnamed index


def test_unlisted_label_stops(tmp_path):
    recording(tmp_path / "rec.csv")
    with pytest.raises(SystemExit):
        prepare.main(["--input", str(tmp_path / "rec.csv"), "--label-col", "status",
                      "--classes", "normal=OK", "breakdown=FAIL", "--out", str(tmp_path / "p")])


def test_check_events(prepared, capsys):
    args = ["--prepared", str(prepared / "prepared"), "--validation", "3", "--test", "4,5", "--context", "1000"]
    assert check_events.main([*args, "--training", "1,2"]) == 0
    out = capsys.readouterr().out
    assert "dropout: dropped, dropout tracks the events" in out
    assert "gappy: dropped, long gap" in out and "dead: dropped, dead" in out
    assert "cannot fill half a window: #5 (120 rows)" in out                       # at 256
    with pytest.raises(SystemExit):                                                 # an incident in two roles
        check_events.main([*args, "--training", "1,3"])
    assert check_events.main(["--prepared", str(prepared / "prepared"), "--training", "", "--validation", "3",
                              "--test", "4,5"]) == 1                               # no training incident


def test_build_and_preflight(prepared):
    assert build(prepared) == 0
    roles = prepared / "roles"
    m = json.load(open(roles / "manifest.json"))
    assert m["channels"] == ["good_a", "good_b", "good_c"]                          # triage, then the top 3
    assert [f["file"] for f in m["training"]] == ["training/normal.csv", "training/breakdown_inc01.csv",
                                                  "training/breakdown_inc02.csv"]
    v3, t4 = m["validation"][0], m["test"][0]
    assert v3["rows"][1] == t4["rows"][0] == (6_200 + 6_400) // 2                   # cut at the midpoint
    assert m["build_end_row"] == v3["rows"][0]
    tr = pd.read_csv(roles / "training/normal.csv")
    assert "label" not in tr and abs(tr.good_a.mean()) < 0.2                        # z-scored on build-time normal
    assert preflight_roles.main(["--roles", str(roles)]) == 0
    assert preflight_roles.main(["--roles", str(roles), "--windows", "64,512"]) == 1   # #5 can't fill half of 512


def test_tail_delivery(prepared):
    assert build(prepared, "--tail-chunk", "1000") == 0
    m = json.load(open(prepared / "roles" / "manifest.json"))
    tails = [f for f in m["delivery"] if "incident" not in f]
    assert tails and tails[0]["rows"][0] == max(f["rows"][1] for f in m["test"]) and tails[-1]["rows"][1] == N
    assert preflight_roles.main(["--roles", str(prepared / "roles")]) == 0


def _rewrite(roles, rel, fn):
    p = roles / rel
    fn(pd.read_csv(p)).to_csv(p, index=False)


def test_breaks(prepared):
    build(prepared)
    roles = prepared / "roles"
    m = json.load(open(roles / "manifest.json"))
    cases = [
        ("label in a training file", m["training"][1]["file"], lambda d: d.assign(label="breakdown")),
        ("label in a delivery file", m["delivery"][0]["file"], lambda d: d.assign(label="normal")),
        ("one class in validation", m["validation"][0]["file"], lambda d: d.assign(label="normal")),
        ("unknown label", m["test"][0]["file"], lambda d: d.assign(label=d.label.replace("breakdown", "fault"))),
        ("a time jump", m["test"][0]["file"], lambda d: d.assign(timestamp=d.timestamp + (d.index > 50) * 600)),
        ("a missing value", m["test"][0]["file"], lambda d: d.assign(good_a=d.good_a.where(d.index != 7))),
        ("a dropped row", m["test"][0]["file"], lambda d: d.drop(index=100)),
    ]
    for name, rel, fn in cases:
        backup = (roles / rel).read_text()
        _rewrite(roles, rel, fn)
        assert preflight_roles.main(["--roles", str(roles)]) == 1, name
        (roles / rel).write_text(backup)
    assert preflight_roles.main(["--roles", str(roles)]) == 0
    keep = json.dumps(m)
    m["test"][0]["rows"][0] -= 50                                                   # overlaps validation
    json.dump(m, open(roles / "manifest.json", "w"))
    assert preflight_roles.main(["--roles", str(roles)]) == 1
    (roles / "manifest.json").write_text(keep)


def test_two_event_classes(tmp_path):
    inc = [(1_000, 300, "BROKE"), (2_000, 300, "SLIP"), (3_000, 300, "BROKE"), (4_000, 300, "SLIP"),
           (7_000, 200, "BROKE"), (9_000, 200, "SLIP")]
    recording(tmp_path / "rec.csv", incidents=inc)
    prepare.main(["--input", str(tmp_path / "rec.csv"), "--label-col", "status",
                  "--classes", "normal=OK", "breakdown=BROKE", "slip=SLIP", "--out", str(tmp_path / "prepared")])
    assert build_roles.main(["--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "roles"),
                             "--training", "1,2", "--validation", "3,4", "--test", "5,6", "--context", "500",
                             "--normal-rows", "800", "--top", "3"]) == 0
    m = json.load(open(tmp_path / "roles" / "manifest.json"))
    assert m["classes"] == ["normal", "breakdown", "slip"]
    assert sorted(f["class"] for f in m["training"]) == ["breakdown", "normal", "slip"]
    assert preflight_roles.main(["--roles", str(tmp_path / "roles")]) == 0
    shutil.rmtree(tmp_path / "roles")
    assert check_events.main(["--prepared", str(tmp_path / "prepared"), "--training", "1,3",
                              "--validation", "2,4", "--test", "5,6"]) == 1        # slip has no training incident


def _prepared_with_gap(tmp_path, gap_at):
    recording(tmp_path / "rec.csv", gap_at=gap_at)
    prepare.main(["--input", str(tmp_path / "rec.csv"), "--label-col", "status", "--classes", *CLASSES,
                  "--out", str(tmp_path / "prepared")])
    return tmp_path


def test_normal_run_split_at_a_gap(tmp_path):
    tmp = _prepared_with_gap(tmp_path, 3_500)          # the only normal stretch long enough has a gap in it
    assert build(tmp, "--normal-rows", "1400") == 0
    m = json.load(open(tmp / "roles" / "manifest.json"))
    assert m["training"][0]["rows"] == [3_500, 4_900]                               # after the gap, not across it
    assert preflight_roles.main(["--roles", str(tmp / "roles")]) == 0


def test_slice_clipped_to_its_segment(tmp_path):
    tmp = _prepared_with_gap(tmp_path, 5_600)          # inside validation #3's context
    assert build(tmp) == 0
    m = json.load(open(tmp / "roles" / "manifest.json"))
    assert m["validation"][0]["rows"][0] == 5_600 == m["build_end_row"]
    assert preflight_roles.main(["--roles", str(tmp / "roles")]) == 0               # no jump inside a file


def test_every_normal_run(prepared):
    assert build(prepared, "--every-normal-run") == 0
    m = json.load(open(prepared / "roles" / "manifest.json"))
    normal = [f for f in m["training"] if f["class"] == "normal"]
    assert [f["file"] for f in normal] == ["training/normal_01.csv", "training/normal_02.csv"]
    assert [f["rows"] for f in normal] == [[200, 1_000], [4_200, 5_000]]           # each run's last 800 rows
    assert preflight_roles.main(["--roles", str(prepared / "roles")]) == 0
