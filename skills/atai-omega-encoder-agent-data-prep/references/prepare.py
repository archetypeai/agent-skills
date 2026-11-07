#!/usr/bin/env python3
"""Step 1: one segment per recording, on an exact grid, cut to a fixed number of rows.

    python prepare.py --index sample_data/index.csv --out prepared --rows 1024 \\
        --onset-channel torque_current --onset-above 0.05 --pre-rows 256

For every segment in the index (one operation: a tightening, a cycle, a batch):
- **The step.** `--step` (the recorder's nominal rate) when given; otherwise the mean of the
  regular differences between timestamps, to 4 significant figures. Give `--step` when the timestamps
  jitter by more than a few percent of a step.
- **The cut.** `--rows` rows, starting `--pre-rows` before the first row where
  `--onset-channel` rises above `--onset-above` (so every segment lines up on the same
  event), or from the first row without an onset channel. `--rows` must be a multiple of
  every window you will try (Step 3 checks).
- **Continuity.** A cut with a step more than `--max-step` times the median (a gap: a pause,
  a restart) or a step back by as much is not one take: the segment is dropped, never
  interpolated. Shorter hiccups (a skipped or repeated sample) are absorbed by the grid.
- **The grid.** Timestamps are re-gridded to t0 + i * step: recorders jitter, and the agent's
  windows count rows, not seconds. Values are not resampled. How far the recorded
  timestamps sat from the grid, in steps, goes in the index (`max_offset_steps`).

Writes <out>/<segment>.csv (timestamp + channels) and <out>/index.csv (the input index plus
step, start and what was dropped, and why).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe_common import read_table  # noqa: E402


def cut(df: pd.DataFrame, rows: int, onset_channel: str | None, onset_above: float, pre_rows: int,
        max_step: float, nominal: float | None = None) -> tuple[pd.DataFrame | None, str, float]:
    """The segment's `rows`-row cut on an exact grid (or None and the reason), and how far its
    recorded timestamps sat from that grid, in steps. Rows stay in recorded order: sorting
    would hide a step back in time."""
    if "timestamp" not in df.columns:
        return None, "no timestamp column", 0.0
    df = df.reset_index(drop=True)
    t = df["timestamp"].to_numpy(float)
    if len(t) < 2:
        return None, "fewer than 2 rows", 0.0
    # The nominal step: the mean of the regular differences (within half a median of the
    # median; jitter cancels across a recording, where the median keeps it), to 4 significant
    # figures (0.01, 0.005, 0.001…), so the grid is not a hair off the recorder's rate.
    diffs = np.diff(t)
    med = float(np.median(diffs))
    regular = diffs[np.abs(diffs - med) < 0.5 * abs(med)]
    step = nominal or float(f"{(regular.mean() if len(regular) else med):.4g}")
    if step <= 0:
        return None, "timestamps do not increase", 0.0
    start = 0
    if onset_channel:
        if onset_channel not in df.columns:
            return None, f"no {onset_channel} column", 0.0
        on = np.flatnonzero(df[onset_channel].to_numpy(float) > onset_above)
        if not len(on):
            return None, f"{onset_channel} never above {onset_above}", 0.0
        start = max(0, int(on[0]) - pre_rows)
    if start + rows > len(df):
        return None, f"too short: {len(df) - start} rows from the cut's start, need {rows}", 0.0
    w = df.iloc[start:start + rows].reset_index(drop=True)
    d = np.diff(w["timestamp"].to_numpy(float))
    if (d < -max_step * step).any():
        return None, f"timestamps go back {-d.min():.3g} s inside the cut", 0.0
    if (d > max_step * step).any():
        return None, f"a gap of {d.max():.3g} s inside the cut (step {step:.3g} s)", 0.0
    grid = w["timestamp"].iloc[0] + np.arange(rows) * step
    offset = float(np.abs(w["timestamp"].to_numpy(float) - grid).max() / step)
    w["timestamp"] = np.round(grid, 9)
    return w, "", offset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="CSV: segment, file, role, group, label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, required=True, help="rows per segment: a multiple of every window you will try")
    ap.add_argument("--onset-channel", help="align each cut on the first row where this channel rises above --onset-above")
    ap.add_argument("--onset-above", type=float, default=0.0)
    ap.add_argument("--pre-rows", type=int, default=0, help="rows kept before the onset")
    ap.add_argument("--step", type=float, help="the sample step in seconds (default: estimated per segment)")
    ap.add_argument("--max-step", type=float, default=10.0,
                    help="a step over this many times the median (or back by as much) is a gap (default 10)")
    a = ap.parse_args()

    index = pd.read_csv(a.index)
    missing = {"segment", "file", "role", "group", "label"} - set(index.columns)
    if missing:
        sys.exit(f"{a.index}: missing columns {sorted(missing)}")
    base = os.path.dirname(os.path.abspath(a.index))
    os.makedirs(a.out, exist_ok=True)
    out = []
    for r in index.itertuples(index=False):
        w, why, offset = cut(read_table(os.path.join(base, r.file)), a.rows, a.onset_channel, a.onset_above,
                     a.pre_rows, a.max_step, a.step)
        row = r._asdict()
        if w is None:
            row.update(kept=False, reason=why)
        else:
            w.to_csv(os.path.join(a.out, f"{r.segment}.csv"), index=False)
            step = float(w["timestamp"].iloc[1] - w["timestamp"].iloc[0])
            row.update(kept=True, reason="", step_s=round(step, 9), start=float(w["timestamp"].iloc[0]),
                       max_offset_steps=round(offset, 2))
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(a.out, "index.csv"), index=False)
    kept = res[res.kept]
    print(f"kept {len(kept)} of {len(res)} segments, {a.rows} rows each")
    for role, g in kept.groupby("role", sort=False):
        print(f"  {role:<11} " + ", ".join(f"{k} {v}" for k, v in g["label"].value_counts().items()))
    for r in res[~res.kept].itertuples():
        print(f"  dropped {r.segment}: {r.reason}")
    steps = kept["step_s"].round(9).unique() if len(kept) else []
    if len(steps) > 1:
        print(f"  WARNING: {len(steps)} different sample steps: {sorted(steps)[:5]}")
    print(f"wrote {a.out}/ ({len(kept)} segments) and {a.out}/index.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
