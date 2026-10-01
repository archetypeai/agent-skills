#!/usr/bin/env python3
"""Step 4: check the role files against the platform's rules, read-only, before any upload.

Checks every rule in role-files.md, so a problem shows up here instead of as a failed
(or silently stuck) optimization:

  per file   header and channel order; timestamp format (epoch seconds, 3 decimals);
             an exact grid (no jump anywhere in validation / test / delivery; in a
             library file, forward jumps only where a piece starts); finite values;
             labels only from the manifest's states; at least one whole window;
             delivery: no label column, and a held-back labels file row-for-row
  vs data    (only if --prepared is given) library pieces are one state throughout,
             and values match the prepared data z-scored with the library statistics
  per role   the manifest matches the disk; no recording (or group) in two roles; the
             statistics come from the library recordings only; the library holds
             per_state windows of every state; EVERY state is present in validation,
             the search subset, test and delivery (a missing state scores F1 = 0);
             the optimization config stays far under the platform's 1 MiB limit

Prints one line per check (PASS / WARN / FAIL) and exits 1 on any FAIL.

    python preflight_roles.py --roles roles [--prepared prepared --index recordings.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_common import load_prepared, read_index  # noqa: E402

TS = re.compile(r"^\d+\.\d{3}$")
CONFIGMAP_BYTES = 1_048_576
BYTES_PER_EXAMPLE = 636          # measured on dev: pretty-printed config, file lists twice
SCALE_TOL = 1e-3


def check_file(roles, entry, role, m, stats, prepared):
    """{check: (level, message)} for one role file."""
    path = os.path.join(roles, entry["file"])
    out = {}
    if not os.path.exists(path):
        return {"exists": ("FAIL", f"{entry['file']} is in the manifest but not on disk")}
    raw = pd.read_csv(path, dtype={"timestamp": str})
    want = ["timestamp", *m["channels"]] + (["label"] if role in ("validation", "test") else [])
    out["header"] = ("PASS", "") if list(raw.columns) == want else ("FAIL", f"columns {list(raw.columns)}, want {want}")
    if out["header"][0] == "FAIL":
        return out
    ts = raw.timestamp.astype(str)
    bad = (~ts.str.match(TS)).sum()
    out["timestamp format"] = ("PASS", "") if not bad else ("FAIL", f"{bad} timestamps not 'epoch seconds, 3 decimals'")
    ms = (ts.str.replace(".", "", regex=False)).astype(np.int64).to_numpy()
    step = np.diff(ms)
    starts = {p["row"] for p in entry.get("pieces", [])}
    jumps = np.flatnonzero(step != m["step_ms"]) + 1
    if role == "library":
        inside = [int(j) for j in jumps if j not in starts]
        backward = int((step <= 0).sum())
        out["grid"] = ("PASS", f"{len(jumps)} jump(s), all at piece starts") if not inside and not backward else (
            "FAIL", f"{len(inside)} jump(s) inside a piece (rows {inside[:3]}), {backward} backward")
    else:
        out["grid"] = ("PASS", "") if not len(jumps) else (
            "FAIL", f"{len(jumps)} jump(s) (first at row {int(jumps[0])}): a scored window across one fails the whole "
                    f"trial or eval; split the file at the gap")
    x = raw[m["channels"]].to_numpy(np.float64)
    out["values"] = ("PASS", "") if np.isfinite(x).all() else ("FAIL", f"{(~np.isfinite(x)).sum()} non-finite values")
    out["windows"] = ("PASS", "") if len(raw) >= m["window"] else ("FAIL", f"{len(raw)} rows < one window ({m['window']})")
    if role in ("validation", "test"):
        unknown = sorted(set(raw.label.astype(str)) - set(m["states"]))
        blank = int(raw.label.isna().sum())
        out["labels"] = ("PASS", "") if not unknown and not blank else ("FAIL", f"unknown {unknown}, blank {blank}")
    if role == "delivery":
        lab = os.path.join(roles, entry.get("labels", ""))
        if not entry.get("labels") or not os.path.exists(lab):
            out["held-back labels"] = ("FAIL", "no labels file in delivery_labels/")
        else:
            s = pd.read_csv(lab, dtype={"timestamp": str})
            same = len(s) == len(raw) and (s.timestamp.astype(str).to_numpy() == ts.to_numpy()).all()
            unknown = sorted(set(s.label.astype(str)) - set(m["states"]))
            out["held-back labels"] = ("PASS", "") if same and not unknown else (
                "FAIL", f"row-for-row with the delivery file: {same}; unknown labels {unknown}")
    if role == "library" and "state" in entry:
        prefix = os.path.basename(entry["file"]).split("__")[0]
        ok = prefix == entry["state"] and entry["state"] in m["states"]
        out["state by filename"] = ("PASS", "") if ok else ("FAIL", f"file prefix {prefix!r}, manifest {entry['state']!r}")
    if prepared:
        out.update(check_against_prepared(raw, ms, entry, role, m, stats, prepared))
    return out


def check_against_prepared(raw, ms, entry, role, m, stats, prepared):
    out, mean, std = {}, np.array(stats["mean"]), np.array(stats["std"])
    spans = ([(p["recording"], p["row"], p["rows"]) for p in entry["pieces"]] if role == "library"
             else [(entry["recording"], 0, len(raw))])
    wrong, err = [], 0.0
    for rec, row, n in spans:
        try:
            p = load_prepared(prepared, rec)
        except FileNotFoundError:
            return {"vs prepared": ("WARN", f"no prepared data for {rec}: not checked")}
        at = np.searchsorted(p.timestamp.to_numpy(), ms[row:row + n])
        found = (at < len(p)) & (p.timestamp.to_numpy()[np.minimum(at, len(p) - 1)] == ms[row:row + n])
        if not found.all():
            wrong.append(f"{rec}@row{row}: timestamps not in the prepared data")
            continue
        if role == "library" and not (p.label.astype(str).to_numpy()[at] == entry["state"]).all():
            wrong.append(f"{rec}@row{row}: not all {entry['state']}")
        want = (p[m["channels"]].to_numpy(np.float64)[at] - mean) / std
        err = max(err, float(np.abs(raw[m["channels"]].to_numpy(np.float64)[row:row + n] - want).max()))
    tol = max(SCALE_TOL, 0.5 * 10 ** -m.get("decimals", 4) / float(np.min(std)) + 10 ** -m.get("decimals", 4))
    out["vs prepared"] = ("PASS", f"max scaling error {err:.1e}") if not wrong and err <= tol else (
        "FAIL", "; ".join(wrong[:3]) or f"max scaling error {err:.1e} > {tol:.1e}")
    return out


def role_checks(roles, m, stats, index):
    out = {}
    listed = {f["file"] for r in ("library", "validation", "test", "delivery") for f in m[r]["files"]}
    on_disk = {f"{r}/{n}" for r in ("library", "validation", "test", "delivery")
               for n in os.listdir(os.path.join(roles, r)) if n.endswith(".csv")}
    out["manifest"] = ("PASS", "") if listed == on_disk else (
        "FAIL", f"on disk only {sorted(on_disk - listed)[:3]}, listed only {sorted(listed - on_disk)[:3]}")
    role_of = defaultdict(set)
    for r in ("validation", "test", "delivery"):
        for f in m[r]["files"]:
            role_of[f["recording"]].add(r)
    for f in m["library"]["files"]:
        for p in f["pieces"]:
            role_of[p["recording"]].add("library")
    two = sorted(r for r, s in role_of.items() if len(s) > 1)
    msg = [f"recordings in two roles: {two[:3]}"] if two else []
    if index is not None and "group" in index and index.group.str.len().any():
        g = index.set_index("recording").group.to_dict()
        groups = defaultdict(set)
        for rec, s in role_of.items():
            groups[g.get(rec, rec)] |= s
        msg += [f"groups in two roles: {sorted(k for k, s in groups.items() if len(s) > 1)[:3]}"] \
            if any(len(s) > 1 for s in groups.values()) else []
    out["separation"] = ("FAIL", "; ".join(msg)) if msg else ("PASS", "")
    lib_recs = {p["recording"] for f in m["library"]["files"] for p in f["pieces"]}
    out["normalisation"] = ("PASS", "") if lib_recs <= set(stats.get("recordings", [])) and not (
        set(stats.get("recordings", [])) & {r for r, s in role_of.items() if "library" not in s}) else (
        "FAIL", "zscore_stats.json must come from the library recordings only")
    per = {f["state"]: f["windows"] for f in m["library"]["files"]}
    missing = [s for s in m["states"] if s not in per]
    low = {s: n for s, n in per.items() if n < m["library"]["per_state"]}
    out["library balance"] = ("FAIL", f"no library file for {missing}") if missing else (
        ("WARN", f"below per_state {m['library']['per_state']}: {low}") if low else ("PASS", str(per)))
    sets = {"validation": m["validation"]["files"],
            "validation (search subset)": [f for f in m["validation"]["files"]
                                           if f["recording"] in set(m.get("search_validation") or [])]
            or m["validation"]["files"],
            "test": m["test"]["files"], "delivery": m["delivery"]["files"]}
    for name, files in sets.items():
        secs = Counter()
        for f in files:
            secs.update(f.get("seconds", {}))
        miss = [s for s in m["states"] if secs[s] == 0]
        out[f"all states: {name}"] = ("FAIL", f"missing {miss}: the platform scores a missing state as F1 = 0") \
            if files and miss else (("WARN", "no files") if not files else ("PASS", ""))
    n = len(m["library"]["files"]) + len(sets["validation (search subset)"])
    est = n * BYTES_PER_EXAMPLE
    out["config size"] = ("PASS", f"~{est / 1024:.0f} KiB for {n} examples") if est < 0.25 * CONFIGMAP_BYTES else (
        ("WARN" if est < 0.9 * CONFIGMAP_BYTES else "FAIL"),
        f"~{est / 1024:.0f} KiB of the 1,024 KiB limit for {n} examples: over it the job never starts")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--prepared", help="the prepare.py output, to check pieces and scaling against it")
    ap.add_argument("--index", help="recordings.csv, to check that groups stay in one role")
    a = ap.parse_args(argv)
    m = json.load(open(os.path.join(a.roles, "manifest.json")))
    stats = json.load(open(os.path.join(a.roles, m["zscore_stats"])))
    index = read_index(a.index) if a.index else None
    per_file = defaultdict(list)
    for role in ("library", "validation", "test", "delivery"):
        for e in m[role]["files"]:
            for check, res in check_file(a.roles, e, role, m, stats, a.prepared).items():
                per_file[check].append((res, e["file"]))
    fails = 0
    print(f"preflight: {sum(len(m[r]['files']) for r in ('library', 'validation', 'test', 'delivery'))} role files in {a.roles}/")
    for check, results in per_file.items():
        c = Counter(r[0][0] for r in results)
        level = "FAIL" if c["FAIL"] else ("WARN" if c["WARN"] else "PASS")
        fails += c["FAIL"]
        print(f"  {level:<4}  {check:<22} {c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail")
        for (lv, msg), f in results:
            if lv != "PASS":
                print(f"          {lv}  {f}: {msg}")
    for check, (lv, msg) in role_checks(a.roles, m, stats, index).items():
        fails += lv == "FAIL"
        print(f"  {lv:<4}  {check:<22} {msg}")
    print("RESULT: FAIL" if fails else "RESULT: PASS, ready for the OSM agent skill (Path 2)")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
