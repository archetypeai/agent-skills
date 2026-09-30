#!/usr/bin/env python3
"""Step 3: build the role files the OSM platform receives, per the contract in role-files.md.

From the prepared recordings (prepare.py) and the index's roles:

  zscore_stats.json   per-channel mean / std over the LIBRARY recordings only
  library/            ONE file per state, <state>__library.csv: --per-state windows of
                      --window rows, spread evenly over the library recordings that have
                      the state and evenly spaced within each, written as continuous
                      single-state pieces in time order with real timestamps (forward
                      jumps only between pieces, at whole-window boundaries)
  validation/, test/  one continuous file per recording segment, with a `label` column
  delivery/           the same, without `label`; the labels go to delivery_labels/
  manifest.json       every file, the library pieces, the states and the search subset

One file per state (not one per piece) keeps the optimization config far under the
platform's 1 MiB limit (~1,500 training files at most; over it, the job silently never
starts). Every CSV: `timestamp` (epoch seconds, 3 decimals), the channels z-scored
(--decimals), then `label` where the role has one.

    python build_roles.py --index recordings.csv --prepared prepared --out roles --window 1024 --per-state 100
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_common import ROLES, load_prepared, read_index, stem, timestamp_strings  # noqa: E402


def channels_of(df):
    return [c for c in df.columns if c not in ("timestamp", "label", "segment")]


def write_csv(d, path, channels, mean, std, label, decimals):
    out = pd.DataFrame({"timestamp": timestamp_strings(d.timestamp.to_numpy())})
    z = (d[channels].to_numpy(np.float64) - mean) / std
    for j, c in enumerate(channels):
        out[c] = z[:, j]
    if label:
        out["label"] = d.label.astype(str).to_numpy()
    out.to_csv(path, index=False, float_format=f"%.{decimals}f")


def zscore_stats(prepared, library):
    n, s1, s2 = 0, 0.0, 0.0
    for rec in library:
        df = load_prepared(prepared, rec)
        channels = channels_of(df)
        x = df[channels].to_numpy(np.float64)
        n, s1, s2 = n + len(x), s1 + x.sum(0), s2 + (x ** 2).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    std[std < 1e-12] = 1.0                       # a constant channel stays 0, never NaN
    return {"channels": channels, "mean": mean.round(8).tolist(), "std": std.round(8).tolist(), "rows": int(n),
            "from": "all rows of the library recordings", "recordings": sorted(library)}


def candidate_windows(df, window):
    """{state: [(start row, run id)]}: whole single-state windows, runs cut at state changes and segments."""
    st, seg = df.label.astype(str).to_numpy(), df.segment.to_numpy()
    brk = np.r_[True, (st[1:] != st[:-1]) | (seg[1:] != seg[:-1])]
    starts = np.flatnonzero(brk)
    lengths = np.diff(np.r_[starts, len(st)])
    out = defaultdict(list)
    for run, (s0, n) in enumerate(zip(starts, lengths)):
        for k in range(n // window):
            out[st[s0]].append((int(s0 + k * window), run))
    return out


def allocate(available, total):
    """Spread `total` as evenly as possible over recordings, capped by what each has."""
    quota, left = {c: 0 for c in available}, total
    while left > 0:
        open_ = sorted(c for c in available if quota[c] < available[c])
        if not open_:
            break
        share = max(1, left // len(open_))
        for c in open_:
            add = min(share, available[c] - quota[c], left)
            quota[c] += add
            left -= add
            if left == 0:
                break
    return quota


def build_library(prepared, library, states, window, per_state):
    cands = {rec: candidate_windows(load_prepared(prepared, rec), window) for rec in library}
    draw, short = {}, {}
    for state in states:
        have = {r: len(c.get(state, [])) for r, c in cands.items() if c.get(state)}
        quota = allocate(have, per_state)
        if sum(quota.values()) < per_state:
            short[state] = sum(quota.values())
        for r, q in quota.items():
            if q:
                idx = np.unique(np.linspace(0, have[r] - 1, q).round().astype(int))
                draw.setdefault(r, {})[state] = [cands[r][state][i] for i in idx]
    pieces = defaultdict(list)                   # state -> [(start ms, recording, segment, frame)]
    for rec, chosen in draw.items():
        df = load_prepared(prepared, rec)
        for state, wins in chosen.items():
            wins = sorted(wins)
            groups, cur = [], [wins[0]]
            for w in wins[1:]:
                if w[1] == cur[-1][1] and w[0] == cur[-1][0] + window:
                    cur.append(w)
                else:
                    groups.append(cur)
                    cur = [w]
            groups.append(cur)
            for g in groups:
                part = df.iloc[g[0][0]:g[-1][0] + window]
                pieces[state].append((int(part.timestamp.iloc[0]), rec, int(part.segment.iloc[0]), part))
    return pieces, short


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--out", default="roles")
    ap.add_argument("--window", type=int, default=1024, help="rows per library window; the platform window should divide it")
    ap.add_argument("--per-state", type=int, default=400, help="library windows per state")
    ap.add_argument("--states", nargs="+", help="default: every state in the library recordings")
    ap.add_argument("--search-validation", nargs="+", metavar="RECORDING",
                    help="the validation recordings scored during a search (default: all of them)")
    ap.add_argument("--decimals", type=int, default=4)
    a = ap.parse_args(argv)
    idx = read_index(a.index)
    if "role" not in idx or (idx.role == "").any():
        raise SystemExit("every recording needs a role (fill the index or run split_roles.py)")
    by_role = {r: sorted(idx.recording[idx.role == r]) for r in ROLES}
    if not by_role["library"]:
        raise SystemExit("no library recordings")
    report = json.load(open(os.path.join(a.prepared, "prepare_report.json")))
    hz = report["hz"]
    step_ms = int(round(1000 / hz))

    for d in ("library", "validation", "test", "delivery", "delivery_labels"):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        os.makedirs(os.path.join(a.out, d))
    stats = zscore_stats(a.prepared, by_role["library"])
    channels, mean, std = stats["channels"], np.array(stats["mean"]), np.array(stats["std"])
    with open(os.path.join(a.out, "zscore_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    states = a.states or sorted({s for rec in by_role["library"]
                                 for s in load_prepared(a.prepared, rec).label.astype(str).unique()})
    pieces, short = build_library(a.prepared, by_role["library"], states, a.window, a.per_state)
    lib_files = []
    for state in states:
        mine = sorted(pieces.get(state, []), key=lambda p: p[0])
        if not mine:
            raise SystemExit(f"state {state!r}: no whole {a.window}-row window in any library recording")
        part = pd.concat([p[3] for p in mine], ignore_index=True)
        if not (np.diff(part.timestamp.to_numpy()) > 0).all():
            raise SystemExit(f"{state}: library pieces overlap in time")
        fname = f"{stem(state)}__library.csv"
        write_csv(part, os.path.join(a.out, "library", fname), channels, mean, std, False, a.decimals)
        rows, plist = 0, []
        for ms0, rec, seg, p in mine:
            plist.append({"recording": rec, "segment": seg, "start_ms": ms0, "row": rows, "rows": len(p),
                          "windows": len(p) // a.window})
            rows += len(p)
        lib_files.append({"file": f"library/{fname}", "state": state, "rows": rows, "windows": rows // a.window,
                          "pieces": plist})
        print(f"library   {fname:<32} {rows // a.window:>5} windows, {len(plist)} pieces from "
              f"{len({p['recording'] for p in plist})} recordings")
    if short:
        print(f"WARNING: fewer than {a.per_state} windows available for {short}; lower --per-state or add library recordings")

    manifest = {"contract": "osm-role-files/v1", "hz": hz, "step_ms": step_ms, "window": a.window,
                "channels": channels, "states": states, "timestamp_format": "epoch seconds, 3 decimals",
                "decimals": a.decimals, "zscore_stats": "zscore_stats.json",
                "search_validation": sorted(a.search_validation or by_role["validation"]),
                "library": {"per_state": a.per_state, "short": short, "files": lib_files}}
    for role in ("validation", "test", "delivery"):
        files = []
        for rec in by_role[role]:
            df = load_prepared(a.prepared, rec)
            for seg, part in df.groupby("segment", sort=True):
                fname = f"{stem(rec)}__seg{int(seg)}.csv"
                write_csv(part, os.path.join(a.out, role, fname), channels, mean, std, role != "delivery", a.decimals)
                entry = {"file": f"{role}/{fname}", "recording": rec, "segment": int(seg), "rows": len(part),
                         "windows": len(part) // a.window,
                         "seconds": {s: round(int(n) * step_ms / 1000, 1)
                                     for s, n in part.label.astype(str).value_counts().items()}}
                if role == "delivery":
                    pd.DataFrame({"timestamp": timestamp_strings(part.timestamp.to_numpy()),
                                  "label": part.label.astype(str).to_numpy()}).to_csv(
                        os.path.join(a.out, "delivery_labels", fname), index=False)
                    entry["labels"] = f"delivery_labels/{fname}"
                files.append(entry)
        manifest[role] = {"files": files}
        print(f"{role:<9} {len(files):>3} files from {len(by_role[role])} recordings, "
              f"{sum(f['windows'] for f in files):,} windows of {a.window} rows")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {a.out}/ (manifest.json); check it with: python preflight_roles.py --roles {a.out}")
    return manifest


if __name__ == "__main__":
    main()
