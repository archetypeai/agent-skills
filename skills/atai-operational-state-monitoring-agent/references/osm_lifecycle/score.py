#!/usr/bin/env python3
"""Path 2, step 4: score the delivered predictions against the labels held back. Local only.

Each prediction in <out>/delivery/<file>.csv is paired with the label of its window's
last row: the row of roles/delivery_labels/<file>.csv whose timestamp equals the
prediction's `finish_timestamp` (the same `last_record` rule the platform scores with).
Windows the platform marked invalid, and any whose finish time matches no labelled row,
are left out and counted. Scored platform-style: macro-F1 over every state in the
manifest, a state with no windows and no predictions counting 0.

Reports pooled and per recording, next to the test number (<out>/test.json) if present.
Writes <out>/delivery/scores.json.

    python score.py --roles roles --out out
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_manifest  # noqa: E402
from deliver import to_ms  # noqa: E402


def scores(y, p, states):
    """Per-state F1 and macro-F1 over `states` (absent from both y and p -> 0), plus the confusion."""
    y, p = np.asarray(y, dtype=object), np.asarray(p, dtype=object)
    f1 = {}
    for s in states:
        tp = int(((y == s) & (p == s)).sum())
        fp = int(((y != s) & (p == s)).sum())
        fn = int(((y == s) & (p != s)).sum())
        f1[s] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    cm = [[int(((y == a) & (p == b)).sum()) for b in states] for a in states]
    return {"macro_f1": round(float(np.mean([f1[s] for s in states])), 4), "f1": {s: round(v, 4) for s, v in f1.items()},
            "confusion": cm, "windows": int(len(y))}


def pairs(pred, labels):
    """(true, predicted, skipped) for one file: pred has finish_timestamp / predicted_state / invalid;
    labels has timestamp / label, row for row with the delivery file."""
    by_ms = dict(zip((labels["timestamp"].astype(float) * 1000).round().astype(np.int64), labels["label"]))
    skipped = Counter()
    if "invalid" in pred:
        bad = pred["invalid"].astype(str).str.lower() == "true"
        skipped["invalid"] = int(bad.sum())
        pred = pred[~bad]
    truth = [by_ms.get(to_ms(v)) for v in pred["finish_timestamp"]]
    keep = np.array([t is not None for t in truth], dtype=bool)
    skipped["no label at finish time"] = int((~keep).sum())
    return (np.array(truth, dtype=object)[keep], pred["predicted_state"].to_numpy(dtype=object)[keep], skipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    manifest = load_manifest(args.roles)
    states = manifest["states"]
    by_file = {os.path.basename(f["file"]): f for f in manifest["delivery"]["files"]}
    paths = sorted(p for p in glob.glob(os.path.join(args.out, "delivery", "*.csv")) if os.path.basename(p) in by_file)
    if not paths:
        sys.exit(f"no predictions in {args.out}/delivery/: run deliver.py first")
    missing = sorted(set(by_file) - {os.path.basename(p) for p in paths})

    ys, ps, recs, skipped = [], [], [], Counter()
    for p in paths:
        f = by_file[os.path.basename(p)]
        labels = pd.read_csv(os.path.join(args.roles, f.get("labels") or f["file"].replace("delivery/", "delivery_labels/", 1)),
                             dtype=str)
        y, pr, sk = pairs(pd.read_csv(p, dtype=str), labels)
        ys.append(y)
        ps.append(pr)
        recs.append(np.array([f["recording"]] * len(y), dtype=object))
        skipped.update(sk)
    y, p, rec = np.concatenate(ys), np.concatenate(ps), np.concatenate(recs)
    out = {"files": len(paths), "files_missing": missing, "skipped": dict(skipped),
           "unknown_predictions": sorted(set(p) - set(states)), "all": scores(y, p, states),
           "per_recording": {r: scores(y[rec == r], p[rec == r], states) for r in sorted(set(rec))}}

    print(f"delivery: {len(set(rec))} recordings, {len(paths)} files, {len(y):,} windows scored; left out {dict(skipped)}")
    if missing:
        print(f"  WARNING: {len(missing)} delivery file(s) have no predictions: {missing}")
    if out["unknown_predictions"]:
        print(f"  WARNING: predictions outside the manifest's states: {out['unknown_predictions']}")
    for name, r in [("all", out["all"])] + list(out["per_recording"].items()):
        print(f"  {name:<40} macro-F1 {r['macro_f1']:.4f}  " + "  ".join(f"{s} {r['f1'][s]:.2f}" for s in states)
              + f"  windows {r['windows']:,}")
    test = os.path.join(args.out, "test.json")
    if os.path.exists(test):
        t = json.load(open(test))
        print(f"next to the test: macro-F1 {t['macro_f1']:.4f}")
        out["test_macro_f1"] = t["macro_f1"]
    path = os.path.join(args.out, "delivery", "scores.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
