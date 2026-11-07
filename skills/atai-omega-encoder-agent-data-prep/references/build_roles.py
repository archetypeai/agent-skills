#!/usr/bin/env python3
"""Step 2: the role files the Omega Encoder agent receives, and everything held back from it.

    python build_roles.py --prepared prepared --out roles --windows 1024 512 256

- **Z-score from the library only:** every channel, with the library's mean and std, so
  nothing about validation, test or delivery leaks into the scale.
- **One file per segment:** <out>/<role>/<role>_<segment>.csv, exactly `timestamp` + the
  channels. The agent embeds every numeric column but the first, so a label, id or index
  column would be embedded as a sensor.
- **Distinct start times:** a batch session's outputs are matched back to its files by each
  window's start timestamp. Recorders restart their clocks (per session, per run), so two
  segments' time ranges can overlap; when any do, every segment's timestamps are re-based to
  its own slot (segment i starts at i * 2 * rows * step), keeping its exact grid. The
  original start stays in the manifest (`source_start`).
- **Labels held back** in <out>/labels/<role>.json, read only by whatever scores that role;
  the library has none (it is normal by construction).
- **manifest.json:** channels, step, rows per segment, candidate windows, the z-score, the
  groups in each role, and every file with its start and end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe_common import NORMAL, ROLES, divides  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", required=True, help="prepare.py's output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[1024], help="every window size you will try")
    ap.add_argument("--channels", nargs="+", help="channels to keep (default: every column but timestamp)")
    a = ap.parse_args()

    idx = pd.read_csv(os.path.join(a.prepared, "index.csv"))
    idx = idx[idx["kept"]].reset_index(drop=True)
    if not len(idx):
        sys.exit("no kept segments in the prepared index")
    frames = {r.segment: pd.read_csv(os.path.join(a.prepared, f"{r.segment}.csv")) for r in idx.itertuples()}
    first = next(iter(frames.values()))
    channels = a.channels or [c for c in first.columns if c != "timestamp"]
    rows = len(first)
    step = float(idx["step_s"].median())
    bad = divides(rows, a.windows)
    if bad:
        sys.exit(f"windows {bad} do not divide {rows} rows per segment: choose --rows in prepare.py again")

    lib = pd.concat([frames[s] for s in idx.loc[idx.role == "library", "segment"]])
    if not len(lib):
        sys.exit("no library segments")
    mean = lib[channels].mean()
    std = lib[channels].std().replace(0, 1.0)

    # Re-base when any two segments' time ranges overlap: their windows' start times could
    # then collide, and a batch session's outputs are matched to files by window start.
    span = (rows - 1) * step
    ordered = np.sort(idx["start"].to_numpy(float))
    rebase = bool(len(ordered) > 1 and (np.diff(ordered) <= span).any())
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    files, labels = [], {r: [] for r in ROLES if r != "library"}
    for i, r in enumerate(idx.itertuples()):
        d = frames[r.segment]
        out = pd.DataFrame({"timestamp": d["timestamp"]})
        if rebase:
            out["timestamp"] = np.round(i * 2 * rows * step + np.arange(rows) * step, 6)
        for c in channels:
            out[c] = ((d[c] - mean[c]) / std[c]).round(6)
        name = f"{r.role}_{r.segment}.csv"
        path = os.path.join(r.role, name)
        os.makedirs(os.path.join(a.out, r.role), exist_ok=True)
        out.to_csv(os.path.join(a.out, path), index=False)
        files.append({"role": r.role, "path": path.replace(os.sep, "/"), "group": str(r.group), "rows": rows,
                      "start": float(out["timestamp"].iloc[0]), "end": float(out["timestamp"].iloc[-1]),
                      "source_start": float(r.start)})
        if r.role != "library":
            labels[r.role].append({"file": name, "class": r.label, "group": str(r.group)})
        elif r.label != NORMAL:
            sys.exit(f"{r.segment}: a library segment labelled {r.label!r}: the library must be normal only")

    for role, ops in labels.items():
        with open(os.path.join(a.out, "labels", f"{role}.json"), "w") as f:
            json.dump({"role": role, "operations": ops}, f, indent=1)
    manifest = {"channels": channels, "step_s": step, "rows": rows, "candidate_windows": a.windows,
                "rebased_timestamps": bool(rebase),
                "zscore": {"mean": mean.round(6).to_dict(), "std": std.round(6).to_dict()},
                "groups": {role: sorted(idx.loc[idx.role == role, "group"].astype(str).unique()) for role in ROLES},
                "files": files}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for role in ROLES:
        g = idx[idx.role == role]
        print(f"{role:<11}{len(g):>4} files  ({', '.join(f'{k} {v}' for k, v in g['label'].value_counts().items())})")
    print(f"{len(channels)} channels, {rows} rows each, step {step:g} s; z-scored with the library's "
          f"{len(lib):,} rows" + ("; timestamps re-based (some segments' time ranges overlapped)" if rebase else ""))
    print(f"wrote {a.out}/ (role files, labels/, manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
