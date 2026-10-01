#!/usr/bin/env python3
"""Step 1: put every recording on an exact sample grid, split at gaps, one state per row.

The OSM platform checks each window's sampling rate against its own mean interval
(`sample_rate_interval_tolerance`, default 0.05, relative). Real recordings jitter and
have gaps, so every recording is:

  1. read (CSV or Parquet); timestamps in epoch s, epoch ms or ISO 8601; rows with a
     missing channel value dropped; optionally samples beyond --abs-max removed
     (physically impossible glitches);
  2. cut into segments wherever consecutive samples are more than --max-gap-s apart
     (a gap is never interpolated across);
  3. resampled per segment onto an exact 1000/hz ms grid, by cubic spline (or linear),
     in time relative to the segment start (at epoch seconds a spline is unstable);
  4. labelled per grid row with the state of the last raw sample at or before it;
  5. segments shorter than --min-rows dropped.

Writes <out>/<recording>.parquet (timestamp as int64 ms, channels float32, label,
segment) and <out>/prepare_report.json (rate, gaps, segments, seconds per state).

    python prepare.py --index recordings.csv --out prepared --hz 200 --label-col label
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_common import read_index, read_recording, stem, to_seconds  # noqa: E402


def runs(t, max_gap):
    """(first, last) index pairs of sorted times with no step over max_gap."""
    cut = np.flatnonzero(np.diff(t) > max_gap)
    return list(zip(np.r_[0, cut + 1], np.r_[cut, len(t) - 1]))


def resample(t_rel, v, grid_rel, method):
    if method == "linear" or len(t_rel) < 4:
        return np.interp(grid_rel, t_rel, v)
    return CubicSpline(t_rel, v)(grid_rel)


def prepare_one(rec, path, out, a):
    d = read_recording(path)
    missing = [c for c in [a.time_col, a.label_col, *(a.channels or [])] if c not in d]
    if missing:
        raise SystemExit(f"{rec}: missing column(s) {missing}")
    channels = a.channels or [c for c in d.columns if c not in (a.time_col, a.label_col)
                              and pd.api.types.is_numeric_dtype(d[c])]
    t = to_seconds(d[a.time_col])
    x = d[channels].to_numpy(float)
    lab = d[a.label_col].astype(str).to_numpy()
    keep = np.isfinite(t) & np.isfinite(x).all(1) & (d[a.label_col].notna().to_numpy())
    dropped_nan = int((~keep).sum())
    glitches = 0
    if a.abs_max is not None:
        bad = keep & (np.abs(np.nan_to_num(x)) > a.abs_max).any(1)
        glitches = int(bad.sum())
        keep &= ~bad
    t, x, lab = t[keep], x[keep], lab[keep]
    order = np.argsort(t, kind="stable")
    t, x, lab = t[order], x[order], lab[order]
    first = np.r_[True, np.diff(t) > 0]            # duplicate timestamps: keep the first
    dups = int((~first).sum())
    t, x, lab = t[first], x[first], lab[first]

    step_ms = 1000.0 / a.hz
    if abs(step_ms - round(step_ms)) > 1e-9:
        raise SystemExit(f"--hz {a.hz}: the grid step must be whole milliseconds (e.g. 50, 100, 200, 250, 500, 1000)")
    step_ms = int(round(step_ms))
    frames, segments, short = [], [], []
    for s, e in runs(t, a.max_gap_s):
        grid_ms = np.arange(int(np.ceil(t[s] * 1000 / step_ms)) * step_ms, int(np.floor(t[e] * 1000)) + 1, step_ms,
                            dtype=np.int64)
        if len(grid_ms) < a.min_rows:
            short.append({"start": float(t[s]), "seconds": round(float(t[e] - t[s]), 3)})
            continue
        origin = t[s]
        t_rel, g_rel = t[s:e + 1] - origin, grid_ms / 1000.0 - origin
        seg = pd.DataFrame({"timestamp": grid_ms})
        for j, c in enumerate(channels):
            seg[c] = resample(t_rel, x[s:e + 1, j], g_rel, a.method).round(a.decimals).astype(np.float32)
        at = np.clip(np.searchsorted(t[s:e + 1], grid_ms / 1000.0, side="right") - 1, 0, e - s)
        seg["label"] = lab[s:e + 1][at]
        seg["segment"] = np.int16(len(segments))
        segments.append({"start": grid_ms[0] / 1000, "rows": len(grid_ms)})
        frames.append(seg)
    if not frames:
        raise SystemExit(f"{rec}: no segment of at least {a.min_rows} rows")
    df = pd.concat(frames, ignore_index=True)
    dst = os.path.join(out, stem(rec) + ".parquet")
    df.to_parquet(dst, index=False)
    dt = np.diff(t)
    counts = df.label.value_counts()
    return {"recording": rec, "channels": channels, "raw_rows": int(keep.size), "rows": len(df),
            "raw_rate_hz": round(float(1 / np.median(dt)), 3) if len(dt) else None,
            "gaps_over_max": int((dt > a.max_gap_s).sum()), "longest_gap_s": round(float(dt.max()), 3) if len(dt) else 0,
            "dropped_nan_rows": dropped_nan, "glitches_removed": glitches, "duplicate_timestamps": dups,
            "segments": segments, "dropped_short": short,
            "seconds": {s: round(int(n) * step_ms / 1000, 1) for s, n in counts.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True, help="recordings.csv (recording, file[, role, group])")
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--hz", type=float, required=True, help="the grid rate; 1000/hz must be whole ms")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--channels", nargs="+", help="default: every numeric column except time and label")
    ap.add_argument("--max-gap-s", type=float, default=1.0, help="split a recording where samples are further apart")
    ap.add_argument("--min-rows", type=int, default=1024, help="drop segments shorter than this (one window)")
    ap.add_argument("--abs-max", type=float, help="remove samples with any |value| above this before resampling")
    ap.add_argument("--method", choices=["cubic", "linear"], default="cubic")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    idx = read_index(a.index)
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        reports = list(pool.map(prepare_one, idx.recording, idx.path, [a.out] * len(idx), [a] * len(idx)))
    chans = {tuple(r["channels"]) for r in reports}
    if len(chans) > 1:
        raise SystemExit(f"recordings disagree on channels: {chans}; pass --channels")
    for r in reports:
        print(f"  {r['recording']:<40} {r['rows']:>10,} rows  {len(r['segments'])} segment(s)  "
              f"raw {r['raw_rate_hz']} Hz, {r['gaps_over_max']} gap(s) > {a.max_gap_s} s")
    rep = {"hz": a.hz, "max_gap_s": a.max_gap_s, "min_rows": a.min_rows, "method": a.method,
           "channels": list(chans.pop()), "recordings": reports}
    with open(os.path.join(a.out, "prepare_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"{len(reports)} recordings, {sum(r['rows'] for r in reports):,} rows -> {a.out}/")
    return rep


if __name__ == "__main__":
    main()
