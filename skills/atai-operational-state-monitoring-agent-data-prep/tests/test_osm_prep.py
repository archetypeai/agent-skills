"""Tests for the OSM data-prep scripts: synthetic recordings through the whole prep,
then deliberately broken role files that the preflight must catch.

No network, no API key. Run:
    pytest skills/atai-operational-state-monitoring-agent-data-prep/tests/ -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REF = Path(__file__).resolve().parent.parent / "references"
sys.path.insert(0, str(REF))

import build_roles  # noqa: E402
import check_states  # noqa: E402
import osm_common  # noqa: E402
import prepare  # noqa: E402
import preflight_roles  # noqa: E402
import split_roles  # noqa: E402

HZ = 50              # 20 ms grid: small and fast
WINDOW = 64
STATES = ["drain", "fill", "spin", "wash"]
# each state has its own tone and level, so they are easy to tell apart
TONE = {"fill": (3.0, 0.5), "wash": (1.0, 0.2), "spin": (9.0, 2.0), "drain": (5.0, 1.0)}


def recording(seed, t0, seconds_per_state=40, gap_after=None, fmt="s"):
    """Jittered ~HZ samples cycling through every state twice; optional gap (seconds) after the first cycle."""
    rng = np.random.default_rng(seed)
    order = ["fill", "wash", "spin", "drain"] * 2
    t, labels, x = [], [], []
    now = t0
    for i, s in enumerate(order):
        n = seconds_per_state * HZ
        dt = 1 / HZ * (1 + rng.uniform(-0.03, 0.03, n))
        tt = now + np.cumsum(dt)
        f, a = TONE[s]
        x.append(np.stack([a * np.sin(2 * np.pi * f * tt), a * np.cos(2 * np.pi * f * tt),
                           rng.normal(0, 0.05, n)], 1))
        t.append(tt)
        labels += [s] * n
        now = tt[-1]
        if gap_after is not None and i == 3:
            now += gap_after
    t, x = np.concatenate(t), np.concatenate(x)
    df = pd.DataFrame({"timestamp": t, "a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "label": labels})
    if fmt == "ms":
        df["timestamp"] = (df.timestamp * 1000).round().astype(np.int64)
    elif fmt == "iso":
        df["timestamp"] = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return df


@pytest.fixture
def workdir(tmp_path):
    specs = [("lib-a", "library", 0, None, "s"), ("lib-b", "library", 1, 30.0, "ms"),
             ("val-a", "validation", 2, None, "iso"), ("test-a", "test", 3, None, "s"),
             ("del-a", "delivery", 4, None, "s")]
    rows = []
    for i, (rec, role, seed, gap, fmt) in enumerate(specs):
        df = recording(seed, 1.7e9 + i * 10_000, gap_after=gap, fmt=fmt)
        path = tmp_path / f"{rec}.csv"
        df.to_csv(path, index=False)
        rows.append({"recording": rec, "file": path.name, "role": role, "group": rec})
    pd.DataFrame(rows).to_csv(tmp_path / "recordings.csv", index=False)
    return tmp_path


def run_prep(w, per_state=8):
    prepare.main(["--index", str(w / "recordings.csv"), "--out", str(w / "prepared"), "--hz", str(HZ),
                  "--min-rows", str(WINDOW), "--max-gap-s", "1.0", "--workers", "1"])
    return build_roles.main(["--index", str(w / "recordings.csv"), "--prepared", str(w / "prepared"),
                             "--out", str(w / "roles"), "--window", str(WINDOW), "--per-state", str(per_state)])


def preflight(w, prepared=True):
    args = ["--roles", str(w / "roles"), "--index", str(w / "recordings.csv")]
    return preflight_roles.main(args + (["--prepared", str(w / "prepared")] if prepared else []))


# --- timestamps and the grid ----------------------------------------------------------

def test_to_seconds_reads_every_timestamp_format():
    s = osm_common.to_seconds(pd.Series([1.7e9, 1.7e9 + 0.02]))
    ms = osm_common.to_seconds(pd.Series([1_700_000_000_000, 1_700_000_000_020]))
    iso = osm_common.to_seconds(pd.Series(["2023-11-14T22:13:20.000Z", "2023-11-14T22:13:20.020Z"]))
    np.testing.assert_allclose(s, ms)
    np.testing.assert_allclose(ms, iso)


def test_timestamp_strings_are_three_decimals():
    assert list(osm_common.timestamp_strings([1_700_000_000_005, 1_700_000_000_120])) == \
        ["1700000000.005", "1700000000.120"]


def test_prepare_gives_an_exact_grid_and_splits_at_gaps(workdir):
    run_prep(workdir)
    d = pd.read_parquet(workdir / "prepared" / "lib-b.parquet")
    assert d.segment.nunique() == 2                      # the 30 s gap is not interpolated across
    for _, seg in d.groupby("segment"):
        assert (np.diff(seg.timestamp.to_numpy()) == 1000 // HZ).all()
    assert set(d.label) == set(STATES)


def test_prepare_labels_each_row_by_the_last_raw_sample(workdir):
    run_prep(workdir)
    raw = pd.read_csv(workdir / "lib-a.csv")
    d = pd.read_parquet(workdir / "prepared" / "lib-a.parquet")
    t = raw.timestamp.to_numpy()
    at = np.searchsorted(t, d.timestamp.to_numpy() / 1000, side="right") - 1
    assert (raw.label.to_numpy()[at] == d.label.to_numpy()).all()


def test_prepare_removes_glitches_before_resampling(workdir):
    raw = pd.read_csv(workdir / "lib-a.csv")
    raw.loc[500, "a"] = 71.5
    raw.to_csv(workdir / "lib-a.csv", index=False)
    rep = prepare.main(["--index", str(workdir / "recordings.csv"), "--out", str(workdir / "prepared"),
                        "--hz", str(HZ), "--min-rows", str(WINDOW), "--abs-max", "10", "--workers", "1"])
    assert next(r for r in rep["recordings"] if r["recording"] == "lib-a")["glitches_removed"] == 1
    assert pd.read_parquet(workdir / "prepared" / "lib-a.parquet").a.abs().max() < 10


# --- the role files --------------------------------------------------------------------

def test_the_whole_prep_passes_the_preflight(workdir):
    m = run_prep(workdir)
    assert m["contract"] == "osm-role-files/v1" and m["states"] == STATES
    assert [f["state"] for f in m["library"]["files"]] == STATES      # ONE file per state
    assert all(f["windows"] == 8 for f in m["library"]["files"])
    assert preflight(workdir) == 0
    assert preflight(workdir, prepared=False) == 0


def test_zscore_comes_from_the_library_only(workdir):
    run_prep(workdir)
    stats = json.load(open(workdir / "roles" / "zscore_stats.json"))
    assert stats["recordings"] == ["lib-a", "lib-b"]
    lib = pd.concat([pd.read_parquet(workdir / "prepared" / f"{r}.parquet") for r in ("lib-a", "lib-b")])
    np.testing.assert_allclose(stats["mean"], lib[["a", "b", "c"]].mean().to_numpy(), atol=1e-6)


def test_library_jumps_fall_only_at_piece_starts(workdir):
    m = run_prep(workdir)
    for f in m["library"]["files"]:
        d = pd.read_csv(workdir / "roles" / f["file"], dtype={"timestamp": str})
        ms = d.timestamp.str.replace(".", "", regex=False).astype(np.int64).to_numpy()
        jumps = set((np.flatnonzero(np.diff(ms) != 1000 // HZ) + 1).tolist())
        assert jumps <= {p["row"] for p in f["pieces"]}
        assert all(p["rows"] % WINDOW == 0 for p in f["pieces"])


def test_delivery_has_no_label_and_its_labels_are_held_back_row_for_row(workdir):
    m = run_prep(workdir)
    e = m["delivery"]["files"][0]
    d = pd.read_csv(workdir / "roles" / e["file"], dtype={"timestamp": str})
    lab = pd.read_csv(workdir / "roles" / e["labels"], dtype={"timestamp": str})
    assert "label" not in d.columns and (d.timestamp == lab.timestamp).all()


def test_split_keeps_groups_together_and_leaves_delivery_alone(tmp_path):
    rows = [{"recording": f"r{i}", "file": f"r{i}.csv", "group": f"g{i // 2}", "role": ""} for i in range(10)]
    rows.append({"recording": "d0", "file": "d0.csv", "group": "d", "role": "delivery"})
    pd.DataFrame(rows).to_csv(tmp_path / "recordings.csv", index=False)
    idx = split_roles.main(["--index", str(tmp_path / "recordings.csv"), "--seed", "7"])
    assert (idx.groupby("group").role.nunique() == 1).all()
    assert idx.set_index("recording").role["d0"] == "delivery"
    again = split_roles.main(["--index", str(tmp_path / "recordings.csv"), "--seed", "7", "--out",
                              str(tmp_path / "again.csv")])
    assert (idx.role.to_numpy() == again.role.to_numpy()).all()       # the seed fixes the split


# --- the preflight catches broken files ---------------------------------------------------

def _edit(path, fn):
    d = pd.read_csv(path, dtype={"timestamp": str})
    fn(d)
    d.to_csv(path, index=False)


def test_preflight_fails_a_jump_inside_a_scored_file(workdir):
    m = run_prep(workdir)
    p = workdir / "roles" / m["validation"]["files"][0]["file"]
    _edit(p, lambda d: d.drop(index=range(100, 120), inplace=True))
    assert preflight(workdir) > 0


def test_preflight_fails_a_jump_inside_a_library_piece(workdir):
    m = run_prep(workdir)
    f = next(f for f in m["library"]["files"] if f["pieces"][0]["rows"] > WINDOW) if any(
        p["rows"] > WINDOW for f in m["library"]["files"] for p in f["pieces"]) else m["library"]["files"][0]
    _edit(workdir / "roles" / f["file"], lambda d: d.drop(index=[WINDOW // 2], inplace=True))
    assert preflight(workdir, prepared=False) > 0


def test_preflight_fails_a_missing_state_in_test(workdir):
    m = run_prep(workdir)
    m["test"]["files"][0]["seconds"].pop("drain")
    json.dump(m, open(workdir / "roles" / "manifest.json", "w"))
    assert preflight(workdir, prepared=False) > 0


def test_preflight_fails_a_library_piece_of_the_wrong_state(workdir):
    m = run_prep(workdir)
    f = m["library"]["files"][0]
    piece = f["pieces"][0]
    p = workdir / "prepared" / f"{piece['recording']}.parquet"
    d = pd.read_parquet(p)
    at = np.flatnonzero(d.timestamp.to_numpy() == piece["start_ms"])[0]
    d.loc[at:at + piece["rows"] - 1, "label"] = "wash" if f["state"] != "wash" else "fill"
    d.to_parquet(p, index=False)                              # the piece's rows are now another state
    assert preflight(workdir) > 0


def test_preflight_fails_wrongly_scaled_values(workdir):
    m = run_prep(workdir)
    p = workdir / "roles" / m["test"]["files"][0]["file"]
    _edit(p, lambda d: d.__setitem__("a", d.a * 2))
    assert preflight(workdir) > 0


def test_preflight_fails_a_bad_timestamp_format(workdir):
    m = run_prep(workdir)
    p = workdir / "roles" / m["test"]["files"][0]["file"]
    _edit(p, lambda d: d.__setitem__("timestamp", d.timestamp.str.slice(0, -1)))
    assert preflight(workdir, prepared=False) > 0


def test_preflight_fails_a_recording_in_two_roles(workdir):
    m = run_prep(workdir)
    m["test"]["files"][0]["recording"] = "lib-a"
    json.dump(m, open(workdir / "roles" / "manifest.json", "w"))
    assert preflight(workdir, prepared=False) > 0


def test_config_size_estimate_fails_too_many_training_files():
    m = {"library": {"files": [{}] * 1600, "per_state": 1}, "validation": {"files": []},
         "test": {"files": []}, "delivery": {"files": []}, "states": [], "search_validation": []}
    n = len(m["library"]["files"])
    assert n * preflight_roles.BYTES_PER_EXAMPLE > 0.9 * preflight_roles.CONFIGMAP_BYTES


# --- check_states --------------------------------------------------------------------------

def test_check_states_separates_distinct_states(workdir):
    run_prep(workdir)
    rep = check_states.check(osm_common.read_index(str(workdir / "recordings.csv")), str(workdir / "prepared"),
                             WINDOW, 5)
    assert rep["balanced_accuracy"] > 0.9


def test_check_states_flags_a_state_the_sensor_cannot_see(workdir):
    run_prep(workdir)
    for rec in ("lib-a", "lib-b", "val-a"):                  # relabel half of wash as "heating": same signal
        p = workdir / "prepared" / f"{rec}.parquet"
        d = pd.read_parquet(p)
        wash = np.flatnonzero(d.label.to_numpy() == "wash")
        d.loc[wash[: len(wash) // 2], "label"] = "heating"
        d.to_parquet(p, index=False)
    rep = check_states.check(osm_common.read_index(str(workdir / "recordings.csv")), str(workdir / "prepared"),
                             WINDOW, 5)
    assert rep["per_state"]["heating"]["recall"] < 0.8
    assert rep["per_state"]["heating"]["mostly_confused_with"] == "wash"
    assert rep["per_state"]["spin"]["recall"] > 0.9
