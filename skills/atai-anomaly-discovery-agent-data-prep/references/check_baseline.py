#!/usr/bin/env python3
"""Step 2: is the library a trustworthy picture of normal?

The detector is fitted on the library and on nothing else, so whatever the library
holds is "normal" by definition. Four checks, before any platform time is spent:

  purity       every library row is labelled `normal`
  stability    within each library recording, the second half does not drift from the
               first more than the library's recordings typically do: the mean shift
               (RMS over channels, in library standard deviations) and the change in
               overall RMS are compared with the library's median, and flagged above
               --outlier x the median or the floors --min-shift / --min-rms-change.
               Normal operation varies on its own (healthy TEP runs change RMS by up to
               ±23% between halves), so a fixed threshold would flag good data; a
               recording that drifts much more than its peers is run-in, a level shift
               or early degradation that would be fitted as normal
  agreement    library recordings resemble each other: each one's overall RMS (z-scored
               with the library's statistics) within --max-rms-dev of the median, so one
               odd recording does not stretch "normal"
  coverage     enough library windows at --window / --step for --k neighbours and --pca
               components; a library too thin does not error, the platform silently clamps
               them to the number of windows minus one

Exit 1 on a FAIL (purity, coverage); stability and agreement WARN, because only you can
say whether a drifting recording is still acceptable operation.

    python check_baseline.py --index recordings.csv --prepared prepared --window 48 --step 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_common import channels_of, load_prepared, read_index  # noqa: E402


def windows(rows, window, step):
    return 0 if rows < window else (rows - window) // step + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--window", type=int, required=True, help="the platform window you plan to search or pin")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--k", type=int, default=20, help="the largest n_neighbors you plan to try")
    ap.add_argument("--pca", type=int, default=20, help="the largest pca_components you plan to try")
    ap.add_argument("--outlier", type=float, default=3.0, help="flag drift above this x the library's median")
    ap.add_argument("--min-shift", type=float, default=0.5, help="never flag a mean shift below this (sd)")
    ap.add_argument("--min-rms-change", type=float, default=0.25, help="never flag an RMS change below this")
    ap.add_argument("--max-rms-dev", type=float, default=0.25)
    a = ap.parse_args(argv)
    idx = read_index(a.index)
    lib = sorted(idx.recording[idx.role == "library"])
    if not lib:
        raise SystemExit("no library recordings")
    frames = {r: load_prepared(a.prepared, r) for r in lib}
    channels = channels_of(next(iter(frames.values())))
    allx = np.concatenate([f[channels].to_numpy(np.float64) for f in frames.values()])
    mean, std = allx.mean(0), allx.std(0)
    std[std < 1e-12] = 1.0
    fails = warns = 0

    impure = {r: int((f.label != "normal").sum()) for r, f in frames.items() if (f.label != "normal").any()}
    if impure:
        fails += 1
        print(f"  FAIL  purity       {len(impure)} library recording(s) hold non-normal rows: {impure}")
    else:
        print(f"  PASS  purity       {len(lib)} library recordings, every row `normal`")

    rms, shift, change = {}, {}, {}
    for r, f in frames.items():
        z = (f[channels].to_numpy(np.float64) - mean) / std
        rms[r] = float(np.sqrt((z ** 2).mean()))
        h1, h2 = z[: len(z) // 2], z[len(z) // 2:]
        shift[r] = float(np.sqrt(((h2.mean(0) - h1.mean(0)) ** 2).mean()))
        r1, r2 = np.sqrt((h1 ** 2).mean()), np.sqrt((h2 ** 2).mean())
        change[r] = float((r2 - r1) / r1)
    lim_shift = max(a.min_shift, a.outlier * float(np.median(list(shift.values()))))
    lim_change = max(a.min_rms_change, a.outlier * float(np.median(np.abs(list(change.values())))))
    drift = {r: f"mean shift {shift[r]:.2f} sd (limit {lim_shift:.2f}), RMS change {change[r]:+.0%} "
                f"(limit ±{lim_change:.0%})"
             for r in frames if shift[r] > lim_shift or abs(change[r]) > lim_change}
    if drift:
        warns += 1
        print(f"  WARN  stability    {len(drift)} recording(s) drift between halves more than their peers:")
        for r, why in drift.items():
            print(f"          {r}: {why}")
    else:
        print(f"  PASS  stability    no recording drifts between halves beyond its peers "
              f"(mean shift <= {lim_shift:.2f} sd, RMS change within ±{lim_change:.0%})")

    med = float(np.median(list(rms.values())))
    odd = {r: round(v, 3) for r, v in rms.items() if abs(v - med) > a.max_rms_dev * med}
    if odd:
        warns += 1
        print(f"  WARN  agreement    {len(odd)} recording(s) more than {a.max_rms_dev:.0%} off the median RMS {med:.3f}: {odd}")
    else:
        print(f"  PASS  agreement    every recording's RMS within {a.max_rms_dev:.0%} of the median {med:.3f}")

    n = sum(windows(len(f), a.window, a.step) for f in frames.values())
    need = max(a.k, a.pca) + 1
    if n < need:
        fails += 1
        print(f"  FAIL  coverage     {n:,} library windows at window {a.window}, step {a.step}: "
              f"fewer than k {a.k} / PCA {a.pca} need ({need})")
    else:
        print(f"  PASS  coverage     {n:,} library windows at window {a.window}, step {a.step} "
              f"support k {a.k} and PCA {a.pca} without clamping")
    print(f"RESULT: {'FAIL' if fails else 'PASS'} ({fails} fail, {warns} warn)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
