#!/usr/bin/env python3
"""Step 4: check the role files against every rule, before any upload.

Per file: manifest vs disk; the header (timestamp, the manifest's channels, `label`
only in validation and test); the timestamp format; finite values; an exact grid inside
each recording, and a gap the platform will reject (> 5% over the step) at every
boundary between joined recordings; labels only `normal` / `anomaly`, matching the
sidecar row for row; delivery files without labels, each with its sidecar.

Per role: both classes in validation and in test (macro-F1 scores a missing class 0);
no recording in two roles (calibration.csv, a delivered copy of validation's normal
runs, is the one exception); the z-score fitted on the library only; with --index, no
group in two roles; with --prepared, every library row `normal`; with --window/--step,
enough library windows for --k and --pca; the optimization config's size against the
platform's 1 MiB limit. Exit 1 on any FAIL.

    python preflight_roles.py --roles roles --prepared prepared --index recordings.csv --window 48 --step 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_common import LABELS, load_prepared, read_index  # noqa: E402

TS = re.compile(r"^\d+\.\d{3}$")
BYTES_PER_EXAMPLE = 636            # measured: the optimization config grows ~636 B per example
CONFIG_LIMIT = 1 << 20


class Report:
    def __init__(self):
        self.fails = self.warns = 0

    def line(self, status, name, msg):
        if status == "FAIL":
            self.fails += 1
        elif status == "WARN":
            self.warns += 1
        print(f"  {status:<5} {name:<22} {msg}")


def check_file(roles, m, role, entry, rep, problems):
    path = os.path.join(roles, entry["file"])
    if not os.path.exists(path):
        problems.append(f"{entry['file']}: missing")
        return None
    d = pd.read_csv(path, dtype={"timestamp": str})
    want = ["timestamp", *m["channels"]] + (["label"] if role in ("validation", "test") else [])
    if list(d.columns) != want:
        problems.append(f"{entry['file']}: columns {list(d.columns)[:3]}… are not timestamp, the "
                        f"{len(m['channels'])} channels{', label' if 'label' in want else ''}")
        return None
    if not d.timestamp.str.match(TS).all():
        problems.append(f"{entry['file']}: timestamps must be epoch seconds with 3 decimals")
    if not np.isfinite(d[m["channels"]].to_numpy(float)).all():
        problems.append(f"{entry['file']}: non-finite values")
    t = (d.timestamp.astype(float) * 1000).round().astype(np.int64).to_numpy()
    if role == "library":
        if not (np.diff(t) == m["step_ms"]).all():
            problems.append(f"{entry['file']}: not an exact {m['step_ms']} ms grid")
        return d
    side = pd.read_csv(os.path.join(roles, entry["labels"]), dtype={"timestamp": str, "type": str})
    if len(side) != len(d) or not (side.timestamp.to_numpy() == d.timestamp.to_numpy()).all():
        problems.append(f"{entry['labels']}: does not match {entry['file']} row for row")
        return d
    if not side.label.isin(LABELS).all():
        problems.append(f"{entry['labels']}: labels other than {list(LABELS)}")
    if "label" in d and not (d.label.to_numpy() == side.label.to_numpy()).all():
        problems.append(f"{entry['file']}: `label` disagrees with its sidecar")
    steps = np.diff(t)
    # a boundary is a change of recording, or the exact seam build_roles.py puts between the
    # segments of one recording that prepare.py split at a gap
    seam_ms = int(m["seam_s"] * 1000)                 # as build_roles.py computes it
    boundary = (side.recording.to_numpy()[1:] != side.recording.to_numpy()[:-1]) | (steps == seam_ms)
    inside, seams = steps[~boundary], steps[boundary]
    if not (inside == m["step_ms"]).all():
        problems.append(f"{entry['file']}: not an exact {m['step_ms']} ms grid inside a recording")
    if (seams <= 1.05 * m["step_ms"]).any():
        problems.append(f"{entry['file']}: a boundary between recordings the platform would not reject "
                        f"(gap within 5% of the step): windows would mix two recordings")
    return side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--prepared", help="prepare.py's output: also check every library row is `normal`")
    ap.add_argument("--index", help="recordings.csv: also check no group is in two roles")
    ap.add_argument("--window", type=int, help="with --step: check library coverage for --k and --pca")
    ap.add_argument("--step", type=int)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--pca", type=int, default=20)
    a = ap.parse_args(argv)
    m = json.load(open(os.path.join(a.roles, "manifest.json")))
    if m.get("contract") != "ad-role-files/v1":
        raise SystemExit(f"{a.roles}/manifest.json is not an ad-role-files/v1 manifest")
    rep = Report()

    listed = {f["file"] for role in ("library", "validation", "test", "delivery") for f in m[role]}
    on_disk = {os.path.relpath(p, a.roles) for role in ("library", "validation", "test", "delivery")
               for p in glob.glob(os.path.join(a.roles, role, "*.csv"))}
    rep.line("PASS" if listed == on_disk else "FAIL", "manifest vs disk",
             f"{len(listed)} files" if listed == on_disk else
             f"unlisted {sorted(on_disk - listed)[:3]}, missing {sorted(listed - on_disk)[:3]}")

    owner, classes = {}, {}
    for role in ("library", "validation", "test", "delivery"):
        problems, n = [], 0
        for entry in m[role]:
            res = check_file(a.roles, m, role, entry, rep, problems)
            n += 1
            if role == "library":
                recs = [entry["recording"]]
            else:
                recs = entry["recordings"]
                if res is not None and "label" in res:
                    for lab, cnt in res.label.value_counts().items():
                        classes.setdefault(role, {}).setdefault(lab, 0)
                        classes[role][lab] += int(cnt)
            if entry.get("calibration"):
                continue
            for r in recs:
                if owner.setdefault(r, role) != role:
                    problems.append(f"recording {r} is in {owner[r]} and {role}")
        rep.line("FAIL" if problems else "PASS", f"files: {role}",
                 "; ".join(problems[:3]) + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else "")
                 if problems else f"{n} files")

    for role in ("validation", "test"):
        c = classes.get(role, {})
        ok = c.get("normal", 0) > 0 and c.get("anomaly", 0) > 0
        rep.line("PASS" if ok else "FAIL", f"both classes: {role}",
                 f"normal {c.get('normal', 0):,} / anomaly {c.get('anomaly', 0):,} rows")

    stats = json.load(open(os.path.join(a.roles, m["zscore_stats"])))
    lib = sorted({f["recording"] for f in m["library"]})
    rep.line("PASS" if sorted(stats["recordings"]) == lib else "FAIL", "z-score from library",
             f"fitted on {len(stats['recordings'])} recordings, the library has {len(lib)}")

    if a.index:
        idx = read_index(a.index)
        g = idx[idx.group != ""].groupby("group").role.nunique()
        split = sorted(g[g > 1].index)
        rep.line("FAIL" if split else "PASS", "groups", f"in two roles: {split[:5]}" if split else
                 f"{idx.group.replace('', np.nan).nunique()} groups, each in one role")
    if a.prepared:
        impure = [r for r in lib if (load_prepared(a.prepared, r).label != "normal").any()]
        rep.line("FAIL" if impure else "PASS", "library all normal",
                 f"non-normal rows in {impure}" if impure else f"{len(lib)} recordings")
    if a.window and a.step:
        n = sum(max(0, (f["rows"] - a.window) // a.step + 1) for f in m["library"])
        need = max(a.k, a.pca) + 1
        rep.line("PASS" if n >= need else "FAIL", "library coverage",
                 f"{n:,} windows at window {a.window}, step {a.step}; k {a.k} / PCA {a.pca} need {need}")
    examples = len(m["library"]) + len(m["validation"])
    size = examples * BYTES_PER_EXAMPLE
    rep.line("PASS" if size < CONFIG_LIMIT else "FAIL", "config size",
             f"~{size // 1024 or 1} KiB for {examples} examples (limit 1 MiB)")
    print(f"RESULT: {'FAIL' if rep.fails else 'PASS'}, {rep.fails} fail(s), {rep.warns} warning(s)"
          + ("" if rep.fails else "; ready for the AD agent skill (Path 2)"))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
