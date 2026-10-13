#!/usr/bin/env python3
"""Step 1: one recording onto an exact sample grid, its labels mapped to classes, its incidents numbered.

The `red` blueprint checks each window's timestamps (monotonic, and each interval within
`sample_rate_interval_tolerance`, default 5%, of the window's mean); a window failing
either is INVALID_STATE, not classified. So:

  1. read the recording (CSV or Parquet); timestamps in epoch s, epoch ms or ISO 8601;
     unnamed index columns and non-numeric columns other than the label are dropped;
  2. sort by time, drop duplicate timestamps, and snap each timestamp to the nearest
     point of the grid (the median interval) when it is off by less than half a step:
     jitter goes, real gaps stay;
  3. number the segments: a gap of more than --max-gap-steps intervals starts a new one
     (never interpolated across);
  4. map the label column to classes with --classes: the first is the background, the
     others are events, e.g. `normal=NORMAL pump_breakdown=BROKEN,RECOVERING`. A value
     not listed stops the script: a silently dropped label is a silently wrong class;
  5. number the incidents: maximal runs of one event class inside one segment, 1, 2, ...
     in time order.

Values are not touched: missing readings stay missing, for check_events.py to judge.

Writes <out>/recording.parquet (timestamp, channels, label, incident, segment) and
<out>/prepare_report.json (grid, segments, incidents, classes).

    python prepare.py --input sample_data/pump_sample.parquet --label-col machine_status \\
        --classes normal=NORMAL pump_breakdown=BROKEN,RECOVERING --out prepared
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from red_common import read_recording, to_seconds  # noqa: E402


def parse_classes(items):
    """['normal=NORMAL', 'pump_breakdown=BROKEN,RECOVERING'] -> (classes, {raw value: class})."""
    classes, mapping = [], {}
    for item in items:
        cls, _, raws = item.partition("=")
        if not cls or not raws:
            raise SystemExit(f"--classes entries look like class=VALUE[,VALUE]: got {item!r}")
        classes.append(cls)
        for raw in raws.split(","):
            if raw in mapping:
                raise SystemExit(f"label value {raw!r} is in two classes")
            mapping[raw] = cls
    if len(classes) < 2:
        raise SystemExit("--classes needs the background class and at least one event class")
    return classes, mapping


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="the recording, CSV or Parquet")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--classes", nargs="+", required=True,
                    help="class=VALUE[,VALUE] ...; the first class is the background")
    ap.add_argument("--max-gap-steps", type=float, default=1.5,
                    help="an interval longer than this many grid steps starts a new segment")
    ap.add_argument("--out", default="prepared")
    a = ap.parse_args(argv)

    classes, mapping = parse_classes(a.classes)
    d = read_recording(a.input)
    for col in (a.time_col, a.label_col):
        if col not in d:
            raise SystemExit(f"{a.input}: no column {col!r}; columns are {list(d.columns)[:12]} ...")
    unnamed = [c for c in d.columns if str(c) == "" or str(c).startswith("Unnamed")]
    raw = d[a.label_col].astype(str)
    unknown = sorted(set(raw) - set(mapping))
    if unknown:
        raise SystemExit(f"label values not in --classes: {unknown}")
    numeric = [c for c in d.columns if c not in (a.time_col, a.label_col, *unnamed)
               and pd.api.types.is_numeric_dtype(d[c])]
    dropped = [c for c in d.columns if c not in (a.time_col, a.label_col, *numeric)]

    t = to_seconds(d[a.time_col])
    order = np.argsort(t, kind="stable")
    t, frame = t[order], d.iloc[order].reset_index(drop=True)
    keep = np.r_[True, np.diff(t) > 0]
    t, frame = t[keep], frame[keep].reset_index(drop=True)
    step = float(np.median(np.diff(t)))
    k = np.round((t - t[0]) / step)
    off = np.abs((t - t[0]) - k * step)
    snapped = int((off > 1e-9).sum())
    near = off < step / 2
    t, frame, k = (t[0] + k * step)[near], frame[near].reset_index(drop=True), k[near]
    seg = np.r_[0, np.cumsum(np.diff(k) > a.max_gap_steps)]

    label = frame[a.label_col].astype(str).map(mapping).to_numpy()
    background = classes[0]
    incident = np.zeros(len(frame), dtype=np.int32)
    incidents = []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        lab = label[idx]
        change = np.r_[True, lab[1:] != lab[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(idx)]
        for a0, b0 in zip(starts, ends):
            if lab[a0] == background:
                continue
            lo, hi = int(idx[a0]), int(idx[b0 - 1]) + 1
            incidents.append({"class": str(lab[a0]), "rows": [lo, hi], "n_rows": hi - lo,
                              "segment": int(s), "start": float(t[lo])})
    incidents.sort(key=lambda i: i["rows"][0])
    for n, inc in enumerate(incidents, 1):
        inc["id"] = n
        incident[inc["rows"][0]:inc["rows"][1]] = n

    integral = np.allclose(t, np.round(t))
    out = pd.DataFrame({"timestamp": np.round(t).astype(np.int64) if integral else t})
    for c in numeric:
        out[c] = frame[c].astype(float).to_numpy()
    out["label"], out["incident"], out["segment"] = label, incident, seg.astype(np.int32)
    os.makedirs(a.out, exist_ok=True)
    out.to_parquet(os.path.join(a.out, "recording.parquet"), index=False)

    segs = [{"segment": int(s), "rows": [int(r[0]), int(r[-1]) + 1]} for s in np.unique(seg)
            for r in [np.flatnonzero(seg == s)]]
    report = {"source": os.path.basename(a.input), "n_rows": len(out), "step_s": step,
              "classes": classes, "background": background, "label_map": mapping,
              "channels": numeric, "dropped_columns": [str(c) for c in dropped],
              "timestamps_snapped": snapped, "rows_off_grid_dropped": int((~near).sum()),
              "segments": segs, "incidents": incidents}
    json.dump(report, open(os.path.join(a.out, "prepare_report.json"), "w"), indent=1)

    print(f"{len(out):,} rows, every {step:g} s, {len(numeric)} numeric channels"
          + (f" ({len(dropped)} other columns dropped: {', '.join(map(str, dropped))})" if dropped else ""))
    print(f"grid: {snapped} timestamps snapped, {int((~near).sum())} rows off the grid dropped, "
          f"{len(segs)} segment(s)")
    counts = pd.Series(label).value_counts()
    print("classes: " + ", ".join(f"{c} {int(counts.get(c, 0)):,}" for c in classes))
    print(f"{len(incidents)} incidents:")
    for inc in incidents:
        when = pd.Timestamp(inc["start"], unit="s").strftime("%Y-%m-%d %H:%M")
        print(f"  #{inc['id']:<3} {inc['class']:<18} rows {inc['rows'][0]:>9,}-{inc['rows'][1]:<9,} "
              f"{inc['n_rows']:>7,} rows  from {when}")
    print(f"wrote {a.out}/recording.parquet and prepare_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
