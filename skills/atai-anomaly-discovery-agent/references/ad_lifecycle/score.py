#!/usr/bin/env python3
"""Path 2, step 4: score the delivery against the held-back labels. Offline, no API key.

Each output window is paired with the held-back label of its last row (the sidecar row
whose timestamp equals `finish_timestamp`); windows the platform marked invalid (across a
boundary between joined recordings) are left out and counted. Then, at two thresholds —
the trial's own (what the agent's labels used) and one set to a --far false-alarm rate on
the calibration file's scores (known-normal data, delivered for this) — it reports:

  per incident type (the sidecar's `type`, else each recording):
    window detection   the share of post-onset windows over the threshold
    detected           recordings with a sustained crossing: --sustain consecutive windows
                       over the threshold at or after onset (one spike does not count)
    delay              hours from onset to that first sustained crossing
    AUROC              the type's anomaly windows against every normal window: threshold-free
  overall:
    the false-alarm rate on normal windows, and the mean window detection over types

    python score.py --roles roles --out out
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_manifest  # noqa: E402


def pair(roles, out, f):
    """The output joined to its sidecar on the window's last row, invalid windows dropped."""
    o = pd.read_csv(os.path.join(out, "delivery", os.path.basename(f["file"])))
    s = pd.read_csv(os.path.join(roles, f["labels"]), dtype={"type": str}).fillna({"type": ""})
    o["ms"] = (o.finish_timestamp * 1000).round().astype(np.int64)
    s["ms"] = (s.timestamp * 1000).round().astype(np.int64)
    invalid = o.invalid.astype(str).str.lower() == "true"
    j = o[~invalid].merge(s[["ms", "label", "recording", "type"]], on="ms", how="left")
    return j, int(invalid.sum()), int(j.label.isna().sum())


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sustained(over, n):
    """Index of the first window that completes n consecutive crossings, or None."""
    run = 0
    for i, v in enumerate(over):
        run = run + 1 if v else 0
        if run >= n:
            return i
    return None


def evaluate(windows, threshold, sustain, step_ms):
    normal = windows[windows.label == "normal"]
    rows = {}
    for (rec, typ), g in windows[windows.label.notna()].groupby(["recording", "type"], sort=False):
        if not (g.label == "anomaly").any():
            continue
        g = g.sort_values("ms")
        onset = g.ms[g.label == "anomaly"].min()
        post = g[g.ms >= onset]
        hit = sustained((post.anomaly_score > threshold).to_numpy(), sustain)
        delay = (post.ms.iloc[hit] - onset) / 3.6e6 if hit is not None else None
        r = rows.setdefault(typ or rec, {"windows": 0, "flagged": 0, "recordings": 0, "detected": 0, "delays": [],
                                         "scores": []})
        r["windows"] += len(post)
        r["flagged"] += int((post.anomaly_score > threshold).sum())
        r["recordings"] += 1
        r["detected"] += hit is not None
        if delay is not None:
            r["delays"].append(float(delay))
        r["scores"].append(post.anomaly_score.to_numpy())
    out = {}
    for k, r in rows.items():
        out[k] = {"window_detection": r["flagged"] / r["windows"], "detected": r["detected"],
                  "recordings": r["recordings"],
                  "median_delay_h": float(np.median(r["delays"])) if r["delays"] else None,
                  "auroc": auroc(np.concatenate(r["scores"]), normal.anomaly_score.to_numpy())}
    fa = int((normal.anomaly_score > threshold).sum())
    return {"threshold": threshold, "by_type": out,
            "false_alarm_rate": fa / max(len(normal), 1), "false_alarms": fa, "normal_windows": len(normal),
            "mean_window_detection": float(np.mean([v["window_detection"] for v in out.values()])) if out else None,
            "macro_auroc": float(np.nanmean([v["auroc"] for v in out.values()])) if out else None}


def show(name, r):
    print(f"\n=== {name}: threshold {r['threshold']:.3f}, sustain {SUSTAIN} ===")
    print(f"  {'type':<14} {'window det':>10} {'detected':>9} {'delay h':>8} {'AUROC':>7}")
    for k, v in sorted(r["by_type"].items()):
        d = "—" if v["median_delay_h"] is None else f"{v['median_delay_h']:.2f}"
        print(f"  {k:<14} {v['window_detection']:>10.1%} {v['detected']:>4}/{v['recordings']:<4} {d:>8} {v['auroc']:>7.3f}")
    print(f"  false alarms on normal windows: {r['false_alarm_rate']:.2%} ({r['false_alarms']:,} / {r['normal_windows']:,})")
    print(f"  mean window detection {r['mean_window_detection']:.1%}   macro AUROC {r['macro_auroc']:.3f}")


SUSTAIN = 3


def main(argv=None):
    global SUSTAIN
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    ap.add_argument("--far", type=float, default=0.01, help="target false-alarm rate for the calibrated threshold")
    ap.add_argument("--sustain", type=int, default=3, help="consecutive windows that make a detection")
    a = ap.parse_args(argv)
    SUSTAIN = a.sustain
    m = load_manifest(a.roles)
    test = json.load(open(os.path.join(a.out, "test_state.json")))
    trial_thr = float(test["blueprint"]["values"]["threshold"])
    scored, calib, invalid, unmatched = [], [], 0, 0
    for f in m["delivery"]:
        j, inv, um = pair(a.roles, a.out, f)
        invalid, unmatched = invalid + inv, unmatched + um
        (calib if f.get("calibration") else scored).append(j)
    windows = pd.concat(scored, ignore_index=True)
    print(f"delivery: {len(scored)} files, {len(windows):,} windows scored; left out: {invalid:,} invalid, "
          f"{unmatched:,} with no label at finish time")
    results = {"trial": evaluate(windows, trial_thr, a.sustain, m["step_ms"])}
    if calib:
        c = pd.concat(calib, ignore_index=True).anomaly_score.to_numpy()
        thr = float(np.quantile(c, 1 - a.far))
        print(f"calibration: {len(c):,} known-normal windows; {(c > trial_thr).mean():.2%} over the trial's "
              f"threshold; threshold for {a.far:.0%} false alarms = {thr:.3f}")
        results["calibrated"] = evaluate(windows, thr, a.sustain, m["step_ms"])
    show("the trial's threshold", results["trial"])
    if "calibrated" in results:
        show(f"calibrated to {a.far:.0%} false alarms on known-normal data", results["calibrated"])
    json.dump(results, open(os.path.join(a.out, "delivery", "scores.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
