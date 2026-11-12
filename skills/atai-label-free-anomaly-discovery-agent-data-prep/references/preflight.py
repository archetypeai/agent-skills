#!/usr/bin/env python3
"""Step 3: will od work on these files? Read-only.

    python preflight.py --prepared prepared --window 360

od reads one CSV per batch session: a timestamp column and numeric channels. It cuts
windows of `window_size` rows from row 0 (step = window here), marks a window invalid when
its sample spacing varies by more than 5% (it crosses a gap), and fits LOF on the valid
windows of that file, with K = 20% of them (at least 5, at most 100). Per file:

- the rules: columns, no label column, finite values, strictly increasing timestamps on
  the grid, no frozen stretch left, a window of 16-1,024 samples (Omega's range);
- the fit: enough gap-free windows for LOF, the share lost to gaps, the longest episode
  that still stands out (about K windows: longer, it forms its own neighbourhood), and
  the file's size, since od holds a whole file's windows in memory.
`--expect-event-h` (how long the events you look for last, if anyone knows) adds a check
that the window is shorter than they are and K longer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_prep import Report, window_counts  # noqa: E402

MIN_VALID = 50          # LOF on fewer windows means little
MAX_GAP_SHARE = 0.25    # over a quarter of the windows invalid: cut the file differently
MAX_WINDOWS = 20_000    # a very long file: split it (od keeps every window's features in memory)
LABEL_NAMES = {"label", "labels", "anomaly", "anomalies", "failure", "fault", "target", "class", "y"}


def k_of(valid: int) -> int:
    return int(min(100, max(5, round(0.2 * valid)), max(1, valid // 3)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepared", type=Path, default=Path("prepared"))
    ap.add_argument("--window", type=int, required=True, help="window_size in rows (step = window)")
    ap.add_argument("--frozen-min", type=float, default=30)
    ap.add_argument("--expect-event-h", type=float, help="how long the events of interest last, in hours")
    a = ap.parse_args(argv)
    rep = json.loads((a.prepared / "prepare_report.json").read_text())
    step, chans = rep["step_s"], rep["channels"]
    r = Report(f"prepared files in {a.prepared}/ (window {a.window} rows = {a.window * step / 3600:.3g} h)")
    r.add("PASS" if 16 <= a.window <= 1024 else "FAIL", "window size", f"{a.window} rows (Omega takes 16-1,024)")
    for f in rep["files"]:
        name = f["file"]
        d = pd.read_csv(a.prepared / name)
        ok = list(d.columns) == ["timestamp", *chans]
        r.add("PASS" if ok else "FAIL", f"{name}: schema", f"timestamp + {len(chans)} channels"
              if ok else f"columns {list(d.columns)}")
        lab = [c for c in d.columns if str(c).lower() in LABEL_NAMES]
        r.add("FAIL" if lab else "PASS", f"{name}: no label", f"{lab}" if lab else "none")
        x = d[chans].to_numpy(dtype=float)
        r.add("PASS" if np.isfinite(x).all() else "FAIL", f"{name}: finite", f"{len(d):,} rows")
        ts = d.timestamp.to_numpy()
        dt = np.diff(ts)
        r.add("PASS" if (dt > 0).all() else "FAIL", f"{name}: increasing", "timestamps strictly increasing")
        r.add("PASS" if np.allclose(dt % step, 0) else "FAIL", f"{name}: on the grid", f"steps are multiples of {step:g} s")
        same = (np.abs(np.diff(x, axis=0)).sum(axis=1) == 0) & (dt == step)
        run = np.diff(np.flatnonzero(np.r_[True, ~same, True])).max() if len(same) else 1
        r.add("PASS" if run < a.frozen_min * 60 / step else "FAIL", f"{name}: no frozen stretch",
              f"longest unchanged run {run} rows")
        n, valid = window_counts(ts, step, a.window)
        share = 1 - valid / n if n else 1.0
        r.add("PASS" if valid >= MIN_VALID else "FAIL", f"{name}: windows",
              f"{n} windows, {valid} gap-free" + ("" if valid >= MIN_VALID else
              f": under {MIN_VALID}, too few for LOF; use a longer file or a shorter window"))
        r.add("WARN" if share > MAX_GAP_SHARE else "PASS", f"{name}: lost to gaps",
              f"{share:.0%} of windows cross a gap" + (": a shorter window, or files cut at the gaps"
                                                         if share > MAX_GAP_SHARE else ""))
        r.add("WARN" if n > MAX_WINDOWS else "PASS", f"{name}: size",
              f"{n:,} windows" + (": split the file (--split-days)" if n > MAX_WINDOWS else ""))
        k = k_of(valid)
        hide_h = k * a.window * step / 3600
        r.add("INFO", f"{name}: LOF neighbours", f"K = {k}: an episode longer than ~{hide_h:.3g} h would form its "
              "own neighbourhood and hide; anomalies should stay a small share of the file")
        if a.expect_event_h:
            win_h = a.window * step / 3600
            r.add("PASS" if win_h <= a.expect_event_h else "WARN", f"{name}: window vs events",
                  f"window {win_h:.3g} h, events ~{a.expect_event_h:g} h" +
                  ("" if win_h <= a.expect_event_h else ": a window longer than the event dilutes it"))
            r.add("PASS" if hide_h > a.expect_event_h else "WARN", f"{name}: K vs events",
                  f"an event up to ~{hide_h:.3g} h still stands out" +
                  ("" if hide_h > a.expect_event_h else ": longer files, or a longer window"))
    return r.done()


if __name__ == "__main__":
    raise SystemExit(main())
