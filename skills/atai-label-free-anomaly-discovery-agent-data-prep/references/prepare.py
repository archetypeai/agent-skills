#!/usr/bin/env python3
"""Step 2: raw data -> the CSV files od reads, one per batch session.

    python prepare.py --input sample_data/metropt3_2020-07-08_2020-07-16.csv.gz \\
        --step-s 10 --out prepared
    python prepare.py --input raw.csv --step-s 60 --split-days 14 --out prepared
    python prepare.py --input raw.csv --step-s 60 --split 2024-03-01..2024-03-22 2024-07-01..2024-07-22

1. An exact grid: each row snapped to a `--step-s` slot (the mean when two land in one).
   Holes up to `--max-fill-s` are sampling jitter and take the last value; longer ones
   are gaps and stay gaps. Nothing is interpolated across a gap.
2. Frozen-logger stretches cut: every channel unchanged for `--frozen-min` or more.
3. Files: the whole span (default), consecutive pieces of `--split-days`, or the date
   ranges given to `--split`.
Values keep their units: od scales what it needs, on the file it scores. Any label
column is left out. Writes <out>/<stem>[_<range>].csv (timestamp in epoch seconds, then
the channels) and <out>/prepare_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_prep import epoch_s, frozen_mask, grid, pick_channels, read_raw, rel  # noqa: E402


def ranges(g: pd.DataFrame, split: list[str] | None, split_days: float | None) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if split:
        out = []
        for s in split:
            lo, hi = s.split("..")
            out.append((f"{lo}_{hi}", pd.Timestamp(lo), pd.Timestamp(hi)))
        return out
    lo, hi = g.index.min(), g.index.max() + pd.Timedelta(seconds=1)
    if not split_days:
        return [("", lo, hi)]
    out, step = [], pd.Timedelta(days=split_days)
    start = lo.normalize()
    while start < hi:
        end = start + step
        out.append((f"{start:%Y-%m-%d}_{end:%Y-%m-%d}", start, end))
        start = end
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--channels", help="comma-separated; default: every numeric, non-binary column")
    ap.add_argument("--step-s", type=float, required=True)
    ap.add_argument("--max-fill-s", type=float, default=None, help="default: 5 steps")
    ap.add_argument("--frozen-min", type=float, default=30)
    ap.add_argument("--split", nargs="+", metavar="FROM..TO", help="date ranges, one file each ([FROM, TO))")
    ap.add_argument("--split-days", type=float, help="consecutive files of this many days")
    ap.add_argument("--out", type=Path, default=Path("prepared"))
    a = ap.parse_args(argv)
    max_fill = int(round((a.max_fill_s if a.max_fill_s is not None else 5 * a.step_s) / a.step_s))

    df = read_raw(a.input, a.timestamp_col)
    chans, _ = pick_channels(df, a.timestamp_col, a.channels)
    g = grid(df, a.timestamp_col, chans, a.step_s, max_fill)
    frozen = frozen_mask(g, a.step_s, max(2, int(round(a.frozen_min * 60 / a.step_s))))
    print(f"grid: {len(df):,} raw rows -> {len(g):,} rows at {a.step_s:g} s; "
          f"frozen-logger rows cut: {int(frozen.sum()):,} ({frozen.sum() * a.step_s / 3600:.1f} h)")
    g = g[~frozen]

    a.out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.input[0]).name.split(".")[0]
    report = {"inputs": a.input, "step_s": a.step_s, "channels": chans, "max_fill_rows": max_fill,
              "frozen_min": a.frozen_min, "frozen_rows_cut": int(frozen.sum()), "files": []}
    for tag, lo, hi in ranges(g, a.split, a.split_days):
        d = g[(g.index >= lo) & (g.index < hi)]
        if d.empty:
            print(f"  {tag or stem}: no data, skipped")
            continue
        ts = epoch_s(d.index)
        pieces = int((pd.Series(ts).diff() != a.step_s).sum())
        out = a.out / (f"{stem}_{tag}.csv" if tag else f"{stem}.csv")
        d.insert(0, "timestamp", ts)
        d.to_csv(out, index=False, float_format="%.6g")
        report["files"].append({"file": out.name, "from": str(lo), "to": str(hi), "rows": len(d),
                                "hours_of_data": round(len(d) * a.step_s / 3600, 1), "pieces": pieces})
        print(f"  {out.name}: {len(d):,} rows ({len(d) * a.step_s / 3600:.1f} h of data) in {pieces} continuous pieces")
    (a.out / "prepare_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {len(report['files'])} file(s) and prepare_report.json to {rel(a.out)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
