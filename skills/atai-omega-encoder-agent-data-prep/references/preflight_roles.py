#!/usr/bin/env python3
"""Step 3: check the role files against every rule, before anything is uploaded. Read-only.

    python preflight_roles.py --roles roles [--k 10]

Exit status is non-zero only on FAIL. What each rule protects is in SKILL.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oe_common import NORMAL, ROLES, SCORED, Report, divides  # noqa: E402


def check(roles_dir: str, k: int = 10) -> int:
    r = Report(f"role preflight ({roles_dir})")
    with open(os.path.join(roles_dir, "manifest.json")) as f:
        m = json.load(f)
    cols = ["timestamp", *m["channels"]]
    rows, step, windows = m["rows"], m["step_s"], m["candidate_windows"]
    problems, checks = [], 0
    for f in m["files"]:
        path = os.path.join(roles_dir, f["path"])
        if not os.path.exists(path):
            problems.append(f"{f['path']}: missing")
            continue
        df = pd.read_csv(path)
        checks += 6
        if list(df.columns) != cols:
            extra = [c for c in df.columns if c not in cols]
            problems.append(f"{f['path']}: columns differ" + (f" (extra: {extra}, which would be embedded)" if extra else ""))
            continue
        if len(df) != rows:
            problems.append(f"{f['path']}: {len(df)} rows, not {rows}")
        if df.isna().any().any():
            problems.append(f"{f['path']}: NaN")
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
            problems.append(f"{f['path']}: a non-numeric column")
        d = np.diff(df["timestamp"].to_numpy(float))
        if len(d) and not np.allclose(d, step, rtol=1e-4, atol=1e-6):
            problems.append(f"{f['path']}: off the {step:g} s grid")
        if not np.isclose(df["timestamp"].iloc[0], f["start"]):
            problems.append(f"{f['path']}: start does not match the manifest")
    r.add("FAIL" if problems else "PASS", "file format",
          "; ".join(problems[:3]) if problems else
          f"{len(m['files'])} files: timestamp + {len(m['channels'])} channels, {rows} rows, {step:g} s grid")

    on_disk = sorted(os.path.relpath(os.path.join(dp, n), roles_dir).replace(os.sep, "/")
                     for dp, _, ns in os.walk(roles_dir) for n in ns
                     if n.endswith(".csv") and os.path.basename(dp) in ROLES)
    listed = sorted(f["path"] for f in m["files"])
    r.add("PASS" if on_disk == listed else "FAIL", "manifest matches disk",
          f"{len(listed)} files" if on_disk == listed else f"{len(set(on_disk) ^ set(listed))} files differ")

    names = [os.path.basename(f["path"]) for f in m["files"]]
    dup = len(names) - len(set(names))
    r.add("FAIL" if dup else "PASS", "unique upload names",
          f"{dup} duplicates (an upload replaces a file of the same name)" if dup else f"{len(names)} distinct")

    spans = sorted((f["start"], f["end"], f["path"]) for f in m["files"])
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] <= a[1]]
    r.add("FAIL" if overlaps else "PASS", "no overlapping time ranges",
          f"{len(overlaps)} pairs overlap, e.g. {overlaps[0]} (batch outputs map back to files by window "
          "start, so windows of two files could be confused)" if overlaps
          else "every file has its own time range")

    bad = divides(rows, windows)
    r.add("FAIL" if bad else "PASS", "whole windows per segment",
          f"windows {bad} do not divide {rows}" if bad else
          ", ".join(f"{w}: {rows // w}" for w in windows) + " window(s) per segment, no remainder")

    for role in SCORED:
        p = os.path.join(roles_dir, "labels", f"{role}.json")
        files = {os.path.basename(f["path"]) for f in m["files"] if f["role"] == role}
        if not files:
            r.add("INFO", f"labels: {role}", "no files in this role")
            continue
        lab = json.load(open(p))["operations"] if os.path.exists(p) else []
        covered = {o["file"] for o in lab}
        counts = pd.Series([o["class"] for o in lab]).value_counts().to_dict() if lab else {}
        ok = covered == files
        both = NORMAL in counts and len(counts) > 1
        r.add("FAIL" if not ok else ("PASS" if both or role == "delivery" else "WARN"), f"labels: {role}",
              ", ".join(f"{k} {v}" for k, v in counts.items()) if ok else
              f"{len(files ^ covered)} files without a label, or labels without a file")
    r.add("FAIL" if os.path.exists(os.path.join(roles_dir, "labels", "library.json")) else "PASS",
          "library is normal-only", "no labels file: every library segment is normal")

    lib_windows = sum(1 for f in m["files"] if f["role"] == "library") * rows // max(windows)
    r.add("PASS" if lib_windows > k else "FAIL", "library size",
          f"{lib_windows} library windows at window {max(windows)}, for k up to {k}")

    seen = {}
    for role in ROLES:
        for g in m["groups"].get(role, []):
            seen.setdefault(g, set()).add(role)
    shared_groups = {g: sorted(v) for g, v in seen.items() if len(v) > 1}
    r.add("FAIL" if shared_groups else "PASS", "no group in two roles",
          f"{shared_groups}" if shared_groups else "disjoint")
    print(f"  {checks} file checks")
    return r.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--k", type=int, default=10, help="the largest k the detector will try")
    a = ap.parse_args()
    return check(a.roles, a.k)


if __name__ == "__main__":
    sys.exit(main())
