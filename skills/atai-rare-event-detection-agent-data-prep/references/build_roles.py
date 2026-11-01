#!/usr/bin/env python3
"""Step 3: the role files the RED agent's Optimizations, Evals and delivery read.

From the prepared recording and the incident roles you chose (check_events.py first):

  channels    the triage check_events.py reports (dead, dropout, long gaps), on
              build-time rows only; then the --top channels by separation of the training
              incidents from build-time normal, or exactly --channels;
  values      short missing runs interpolated (longer ones were triaged out), every
              channel z-scored with the build-time normal rows' mean and standard deviation;
  training    one file of normal operation (the first --normal-rows of a normal run before
              any held-out incident; a run never crosses a gap) and one file per training
              incident: each a single class, set in the Optimize request, so no label
              column. --every-normal-run writes one normal file per such run instead, its
              last --normal-rows: one per operating point when the set point changes
              between runs. Those rows can end at the next incident's onset, where any
              pre-fault behaviour would show: check the incident's lead-in first;
  validation, one file per incident: the incident with --context normal rows either side,
  test        cut at the midpoint where two incidents are closer than that (so no row is in
              two incidents' files), with a `label` column;
  delivery    files without a label column (labels held back in labels/): the --delivery
              test incidents with --delivery-context rows either side (inside their test
              files), and/or --tail-chunk the rows after the last scored incident's file, in
              chunks of that many rows (new normal operation: false alarms per day).

Writes <out>/{training,validation,test,delivery,labels}/*.csv and <out>/manifest.json
(contract: red-role-files/v1, see role-files.md).

    python build_roles.py --prepared prepared --out roles --training 1,2 --validation 3 \\
        --test 4 --delivery 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from red_common import (assign, audit_channels, build_end, load_prepared, parse_ids, runs,  # noqa: E402
                        separation, slice_bounds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--out", default="roles")
    ap.add_argument("--training", required=True, help="incident ids, e.g. 1,2")
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--delivery", default="", help="test incidents to deliver again, e.g. 4")
    ap.add_argument("--tail-chunk", type=int, default=0,
                    help="also deliver the rows after the last scored file, in chunks of this many rows")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--delivery-context", type=int, default=1024)
    ap.add_argument("--normal-rows", type=int, default=8192)
    ap.add_argument("--every-normal-run", action="store_true",
                    help="one normal file per run of --normal-rows before the first held-out incident (its last rows)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--channels", nargs="+", help="use exactly these channels (no triage or selection)")
    ap.add_argument("--max-nan-fraction", type=float, default=0.01)
    ap.add_argument("--max-nan-run", type=int, default=16)
    a = ap.parse_args(argv)

    df, rep = load_prepared(a.prepared)
    delivery = parse_ids(a.delivery)
    roles = assign(rep, parse_ids(a.training), parse_ids(a.validation), parse_ids(a.test), delivery)
    bad = [i for i in delivery if roles.get(i) != "test"]
    if bad:
        raise SystemExit(f"--delivery {bad}: deliver test incidents (their files bound the delivery slices)")
    incidents, n = rep["incidents"], rep["n_rows"]
    background = rep["background"]
    labels = df["label"].to_numpy()
    event_mask = labels != background
    end = build_end(rep, roles, a.context)

    if a.channels:
        channels = a.channels
    else:
        audit = audit_channels(df, event_mask, end, a.max_nan_fraction, a.max_nan_run)
        kept = [c for c, r in audit.items() if r["decision"] == "keep"]
        train = np.zeros(n, bool)
        for inc in incidents:
            if roles.get(inc["id"]) == "training":
                train[inc["rows"][0]:inc["rows"][1]] = True
        normal = ~event_mask
        normal[end:] = False
        sep = separation(df, kept, train, normal)
        channels = sorted(sorted(kept, key=sep.get, reverse=True)[:a.top])

    seg = df["segment"].to_numpy()
    z = pd.DataFrame(index=df.index)
    stats = {}
    build_normal = ~event_mask
    build_normal[end:] = False
    for c in channels:
        x = df[c].groupby(seg).transform(lambda s: s.interpolate(limit_direction="both"))
        mu, sd = float(x[build_normal].mean()), float(x[build_normal].std(ddof=1) or 1.0)
        stats[c] = {"mean": mu, "std": sd}
        z[c] = (x - mu) / sd
    if z.isna().any().any():
        raise SystemExit("missing values remain after interpolation: a channel is empty in a segment")

    out = a.out
    for sub in ("training", "validation", "test", "delivery", "labels"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    ts = df["timestamp"].to_numpy()

    def write(rel, lo, hi, with_label):
        if len(np.unique(seg[lo:hi])) > 1:
            raise SystemExit(f"{rel}: rows {lo}-{hi} cross a gap in the recording")
        d = z.iloc[lo:hi].copy()
        d.insert(0, "timestamp", ts[lo:hi])
        if with_label:
            d["label"] = labels[lo:hi]
        d.to_csv(os.path.join(out, rel), index=False, float_format="%.6f")
        return {"file": rel, "rows": [int(lo), int(hi)], "n_rows": int(hi - lo),
                "event_rows": int(event_mask[lo:hi].sum())}

    def write_labels(entry):
        lo, hi = entry["rows"]
        rel = "labels/" + os.path.basename(entry["file"]).replace(".csv", "_labels.csv")
        pd.DataFrame({"timestamp": ts[lo:hi], "label": labels[lo:hi]}).to_csv(os.path.join(out, rel), index=False)
        entry["labels"] = rel

    m = {"contract": "red-role-files/v1", "classes": rep["classes"], "background": background,
         "timestamp_column": "timestamp", "channels": channels, "step_s": rep["step_s"],
         "source": rep["source"], "build_end_row": int(end), "zscore": stats,
         "training": [], "validation": [], "test": [], "delivery": []}

    stretches = []                                     # normal runs split at gaps, so none crosses one
    for lo, hi in runs((~event_mask)[:end]):
        cuts = [lo, *(lo + 1 + np.flatnonzero(np.diff(seg[lo:hi]))), hi]
        stretches += [(int(p), int(q)) for p, q in zip(cuts, cuts[1:])]
    normal_runs = [(lo, hi) for lo, hi in stretches if hi - lo >= a.normal_rows]
    if not normal_runs:
        longest = max((hi - lo for lo, hi in stretches), default=0)
        raise SystemExit(f"no normal run of {a.normal_rows} rows before row {end} (longest without a gap: "
                         f"{longest}); lower --normal-rows")
    # one file per run (e.g. per set point): its last rows, away from a set-point change at its start,
    # but they can end at the next incident's onset, where pre-fault behaviour would show
    if a.every_normal_run:
        for k, (lo, hi) in enumerate(normal_runs, 1):
            m["training"].append({**write(f"training/{background}_{k:02d}.csv", hi - a.normal_rows, hi, False),
                                  "class": background})
    else:
        lo = normal_runs[0][0]
        m["training"].append({**write(f"training/{background}.csv", lo, lo + a.normal_rows, False),
                              "class": background})
    for inc in incidents:
        if roles.get(inc["id"]) == "training":
            lo, hi = inc["rows"]
            m["training"].append({**write(f"training/{inc['class']}_inc{inc['id']:02d}.csv", lo, hi, False),
                                  "class": inc["class"], "incident": inc["id"]})

    bounds = slice_bounds(incidents, n, a.context, rep.get("segments"))
    for inc in incidents:
        role = roles.get(inc["id"])
        if role in ("validation", "test"):
            lo, hi = bounds[inc["id"]]
            m[role].append({**write(f"{role}/inc{inc['id']:02d}.csv", lo, hi, True), "incident": inc["id"],
                            "class": inc["class"], "incident_rows": inc["rows"]})

    for i in delivery:
        inc = incidents[i - 1]
        a0, b0 = inc["rows"]
        lo, hi = max(a0 - a.delivery_context, bounds[i][0]), min(b0 + a.delivery_context, bounds[i][1])
        e = {**write(f"delivery/inc{i:02d}.csv", lo, hi, False), "incident": i, "class": inc["class"],
             "incident_rows": inc["rows"]}
        write_labels(e)
        m["delivery"].append(e)
    if a.tail_chunk:
        scored_end = max(bounds[i][1] for i, r in roles.items() if r in ("validation", "test"))
        for k, lo in enumerate(range(scored_end, n, a.tail_chunk), 1):
            hi = min(lo + a.tail_chunk, n)
            for s in np.unique(seg[lo:hi]):            # a chunk never crosses a gap
                idx = np.flatnonzero(seg[lo:hi] == s) + lo
                name = f"delivery/tail_{k:02d}" + (f"_{s}" if len(np.unique(seg[lo:hi])) > 1 else "") + ".csv"
                e = write(name, int(idx[0]), int(idx[-1]) + 1, False)
                write_labels(e)
                m["delivery"].append(e)

    json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"channels ({len(channels)}): {' '.join(channels)}")
    print(f"z-scored with normal rows before row {end:,}")
    for role in ("training", "validation", "test", "delivery"):
        rows = sum(f["n_rows"] for f in m[role])
        ev = sum(f["event_rows"] for f in m[role])
        print(f"{role:<10} {len(m[role]):>2} files  {rows:>8,} rows  {ev:>7,} event rows")
    print(f"wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
