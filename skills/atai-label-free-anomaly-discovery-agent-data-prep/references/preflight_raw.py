#!/usr/bin/env python3
"""Step 1: check the raw data before any prep decision. Read-only.

    python preflight_raw.py --input sample_data/metropt3_2020-07-08_2020-07-16.csv.gz --step-s 10

Each WARN names the prepare.py decision it leads to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_prep import Report, pick_channels, read_raw  # noqa: E402

LABEL_NAMES = {"label", "labels", "anomaly", "anomalies", "failure", "fault", "target", "class", "y"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", required=True, help="raw CSV file(s), plain or .gz")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--channels", help="comma-separated; default: every numeric, non-binary column")
    ap.add_argument("--step-s", type=float, required=True, help="the intended sample interval, seconds")
    ap.add_argument("--max-fill-s", type=float, default=None, help="holes up to this are jitter (default 5 steps)")
    ap.add_argument("--frozen-min", type=float, default=30, help="every channel unchanged this long = frozen logger")
    a = ap.parse_args(argv)
    max_fill_s = a.max_fill_s if a.max_fill_s is not None else 5 * a.step_s

    r = Report(f"raw data: {', '.join(a.input)}")
    df = read_raw(a.input, a.timestamp_col)
    chans, left_out = pick_channels(df, a.timestamp_col, a.channels)
    r.add("PASS" if chans else "FAIL", "channels", f"{len(chans)} kept: {', '.join(chans)}")
    if left_out:
        r.add("INFO", "left out", f"{', '.join(left_out)} (binary or not chosen) -> prepare.py keeps the same {len(chans)}")
    labels = [c for c in df.columns if str(c).lower() in LABEL_NAMES]
    r.add("WARN" if labels else "PASS", "no label column",
          f"found {labels}: od never reads it; prepare.py leaves it out" if labels else "none: nothing to hold back")

    t = df[a.timestamp_col]
    r.add("FAIL" if t.isna().any() else "PASS", "timestamps parse", f"{int(t.isna().sum())} unparseable of {len(df):,}")
    dt = t.diff().dt.total_seconds().dropna().to_numpy()
    dup = int((dt <= 0).sum())
    r.add("WARN" if dup else "PASS", "strictly increasing",
          f"{dup} repeated or backward steps -> prepare.py averages rows that share a slot" if dup else "yes")
    off = float(np.mean(np.abs(dt[dt <= max_fill_s] - a.step_s) > 0.05 * a.step_s)) if len(dt) else 0.0
    r.add("WARN" if off > 0.01 else "PASS", "cadence",
          f"median step {np.median(dt):.3g} s; {off:.1%} of steps off {a.step_s:g} s by over 5% "
          "-> prepare.py snaps rows to an exact grid" if off > 0.01 else f"steps of {a.step_s:g} s")
    gaps = dt[dt > max_fill_s]
    r.add("WARN" if len(gaps) else "PASS", "gaps",
          f"{len(gaps)} longer than {max_fill_s:g} s (longest {gaps.max() / 3600:.1f} h) -> kept as gaps; "
          "od marks a window that crosses one invalid" if len(gaps) else "none")
    span_h = (t.max() - t.min()).total_seconds() / 3600
    r.add("INFO", "span", f"{t.min()} .. {t.max()} ({span_h:,.0f} h), {len(df):,} rows")

    x = df[chans].apply(lambda s: s.astype(float))
    nan = int(x.isna().sum().sum())
    r.add("WARN" if nan else "PASS", "missing values", f"{nan} empty cells -> filled only inside short holes" if nan else "none")
    inf = int(np.isinf(x.to_numpy()).sum())
    r.add("FAIL" if inf else "PASS", "finite", f"{inf} infinite values" if inf else "all finite")
    const = [c for c in chans if x[c].nunique() <= 1]
    r.add("FAIL" if const else "PASS", "constant channels", f"{const}: drop them with --channels" if const else "none")
    r.add("INFO", "ranges", ", ".join(f"{c} {x[c].min():.3g}..{x[c].max():.3g}" for c in chans))

    rows = max(2, int(round(a.frozen_min * 60 / a.step_s)))
    same = (x.diff().abs().sum(axis=1) == 0).to_numpy()
    run = np.diff(np.flatnonzero(np.r_[True, ~same, True]))           # lengths of unchanged runs
    frozen = run[run >= rows]
    r.add("WARN" if len(frozen) else "PASS", "frozen logger",
          f"{len(frozen)} stretches where every channel holds one value for {a.frozen_min:g} min or more "
          f"({frozen.sum() * a.step_s / 3600:.0f} h) -> prepare.py cuts them" if len(frozen) else "none")
    return r.done()


if __name__ == "__main__":
    raise SystemExit(main())
