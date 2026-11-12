#!/usr/bin/env python3
"""Step 4: check the role files against every rule Optimize, Evals and delivery depend on.

    python preflight_roles.py --roles roles [--windows 16,32,64]

PASS / FAIL per rule; exits 1 on any FAIL. The rules and why each matters: role-files.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--windows", default="16,32,64", help="the window sizes you will search")
    a = ap.parse_args(argv)
    roles = a.roles
    p = os.path.join(roles, "manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: run build_roles.py first")
    m = json.load(open(p))
    if m.get("contract") != "red-role-files/v1":
        raise SystemExit(f"{p} is not a red-role-files/v1 manifest")
    windows = [int(w) for w in a.windows.split(",")]
    classes, channels, background = m["classes"], m["channels"], m["background"]
    files = {r: m[r] for r in ("training", "validation", "test", "delivery")}
    fails = 0

    def check(ok, rule, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {rule}{'  ' + detail if detail and not ok else ''}")

    data = {}
    for fs in files.values():
        for f in fs:
            path = os.path.join(roles, f["file"])
            if not os.path.exists(path):
                check(False, "every file in the manifest exists", f["file"])
                return 1
            data[f["file"]] = pd.read_csv(path)

    bad = [k for k, d in data.items() if [c for c in d.columns if c not in ("timestamp", "label")] != channels]
    check(not bad, "the manifest's channels, in order, in every file", ", ".join(bad))
    bad = [k for k, d in data.items() if not np.allclose(np.diff(d.timestamp.to_numpy(dtype=float)), m["step_s"])]
    check(not bad, f"timestamps on an exact {m['step_s']:g} s grid, no jump inside a file", ", ".join(bad))
    bad = [k for k, d in data.items() if d[channels].isna().any().any() or not np.isfinite(d[channels].to_numpy()).all()]
    check(not bad, "finite values only", ", ".join(bad))
    bad = [f["file"] for fs in files.values() for f in fs if len(data[f["file"]]) != f["n_rows"]]
    check(not bad, "row counts match the manifest", ", ".join(bad))

    spans = sorted((f["rows"][0], f["rows"][1], f["file"]) for r in ("training", "validation", "test") for f in files[r])
    clash = [f"{x[2]} / {y[2]}" for x, y in zip(spans, spans[1:]) if y[0] < x[1]]
    check(not clash, "no row in two of training, validation and test", "; ".join(clash))
    tests = {t["incident"]: t for t in files["test"]}
    inside, tail_clash = True, []
    for d in files["delivery"]:
        if "incident" in d:
            t = tests.get(d["incident"])
            inside &= bool(t) and t["rows"][0] <= d["rows"][0] and d["rows"][1] <= t["rows"][1]
        else:
            tail_clash += [f"{d['file']} / {x[2]}" for x in spans if d["rows"][0] < x[1] and x[0] < d["rows"][1]]
    check(inside, "each delivered incident lies inside its test file")
    check(not tail_clash, "tail delivery files share no row with training, validation or test", "; ".join(tail_clash))

    bad = [f["file"] for r in ("training", "delivery") for f in files[r] if "label" in data[f["file"]]]
    check(not bad, "no label column in training or delivery files", ", ".join(bad))
    bad = []
    for r in ("validation", "test"):
        for f in files[r]:
            d = data[f["file"]]
            if "label" not in d or not set(d.label) <= set(classes) or {background, f["class"]} - set(d.label):
                bad.append(f["file"])
    check(not bad, f"validation and test files: a label column of {classes}, with the background and the incident's class",
          ", ".join(bad))
    for r in ("validation", "test"):
        check(bool(files[r]), f"{r} has at least one incident")
    bad = [f["file"] for f in files["training"] if f["class"] not in classes
           or f["event_rows"] != (0 if f["class"] == background else f["n_rows"])]
    check(not bad, "each training file is one class, in the class list", ", ".join(bad))
    check({f["class"] for f in files["training"]} == set(classes), f"training covers every class {classes}")

    bad = []
    for d in files["delivery"]:
        lab = pd.read_csv(os.path.join(roles, d["labels"]))
        if len(lab) != d["n_rows"] or not (lab.timestamp.to_numpy() == data[d["file"]].timestamp.to_numpy()).all():
            bad.append(d["labels"])
        elif "incident" in d:
            t = tests[d["incident"]]
            off = d["rows"][0] - t["rows"][0]
            if not (lab.label.to_numpy() == data[t["file"]].label.to_numpy()[off:off + len(lab)]).all():
                bad.append(d["labels"])
    check(not bad, "delivery label files line up row for row (and with the test labels)", ", ".join(bad))

    short = [f"inc{f['incident']:02d} ({f['incident_rows'][1] - f['incident_rows'][0]} rows) at {w}"
             for r in ("validation", "test") for f in files[r] for w in windows
             if 2 * (f["incident_rows"][1] - f["incident_rows"][0]) < w]
    check(not short, f"every scored incident can fill half a window of {a.windows}", "; ".join(short))
    short = [f"{f['file']} at {w}" for f in files["training"] for w in windows if f["n_rows"] < w]
    check(not short, "every training file is at least one window long", "; ".join(short))

    print(f"\n{'all checks pass' if not fails else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
