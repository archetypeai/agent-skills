#!/usr/bin/env python3
"""Path 2, step 4: score the delivered outputs against the held-back labels. Offline, no API key.

Each window is scored three ways, because the event is rare and accuracy says nothing:

  window      the design rule: a window's truth is the class holding the majority of its
              rows (a tie goes to the event); precision / recall of event vs background,
              whichever event.
  incident    each incident detected or missed: detected when a window overlapping it is
              predicted as its class; the delay is from onset to the end of the first
              such window, in rows.
  false alarm an event predicted on a window with no event row. On a tail file of new
              normal operation, also per day.

Windows the platform marked invalid are left out, and so are windows that do not lie
wholly inside the label file.

    python score.py --roles roles --out out
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_manifest  # noqa: E402


def window_truth(labels, lo, hi, background):
    """The design rule: the majority class of labels[lo:hi], a tie going to an event."""
    c = Counter(labels[lo:hi])
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    events = [k for k in tied if k != background]
    return events[0] if events else background


def incidents_of(labels, background):
    """[(start, end, class)] runs of one event class."""
    out, start = [], None
    for i, lab in enumerate(list(labels) + [background]):
        if start is not None and lab != labels[start]:
            out.append((start, i, labels[start]))
            start = None
        if start is None and lab != background and i < len(labels):
            start = i
    return out


def score_windows(labels, windows, background):
    """labels: per row; windows: [(lo, hi, predicted)] half-open row spans. Returns counts,
    false alarms and per-incident detection."""
    r = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "false_alarms": 0, "normal_windows": 0, "windows": len(windows)}
    events = [lab != background for lab in labels]
    pre = [0]
    for e in events:
        pre.append(pre[-1] + e)
    for lo, hi, pred in windows:
        truth = window_truth(labels, lo, hi, background)
        hit, real = pred != background, truth != background
        r[("t" if hit == real else "f") + ("p" if hit else "n")] += 1
        if pre[hi] - pre[lo] == 0:
            r["normal_windows"] += 1
            r["false_alarms"] += hit
    r["incidents"] = []
    for a, b, cls in incidents_of(labels, background):
        hits = [hi for lo, hi, pred in windows if pred == cls and lo < b and a < hi]
        r["incidents"].append({"rows": [a, b], "length": b - a, "class": cls, "detected": bool(hits),
                               "delay_rows": (min(hits) - 1 - a) if hits else None})
    return r


def read_output(path):
    """Rows of a delivered output, without the header repeated at chunk boundaries."""
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("finish_timestamp") != "finish_timestamp"]


def score_output(out_path, labels_path, window, background):
    with open(labels_path, newline="") as fh:
        lab = list(csv.DictReader(fh))
    labels = [r["label"] for r in lab]
    index = {int(float(r["timestamp"])): i for i, r in enumerate(lab)}
    windows, invalid, outside = [], 0, 0
    for row in read_output(out_path):
        if str(row.get("invalid", "")).lower() in ("true", "1") or row["predicted_state"] == "INVALID_STATE":
            invalid += 1
            continue
        end = index.get(int(float(row["finish_timestamp"])))
        if end is None or end < window - 1:
            outside += 1
            continue
        windows.append((end - window + 1, end + 1, row["predicted_state"]))
    r = score_windows(labels, windows, background)
    r.update({"invalid": invalid, "outside": outside})
    return r


def prf(r):
    p = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0
    rec = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0.0
    return p, rec, 2 * p * rec / (p + rec) if p + rec else 0.0


def report(name, r, step_s=None):
    p, rec, f = prf(r)
    print(f"{name}: {r['windows']:,} windows scored ({r.get('invalid', 0)} invalid)")
    if r["tp"] + r["fn"]:
        print(f"  event  precision {p:.4f}  recall {rec:.4f}  F1 {f:.4f}   tp {r['tp']} fp {r['fp']} fn {r['fn']} tn {r['tn']}")
    fa = r["false_alarms"] / r["normal_windows"] if r["normal_windows"] else 0.0
    line = f"  false alarms {r['false_alarms']:,} / {r['normal_windows']:,} = {fa:.4f}"
    if step_s and r["normal_windows"]:
        line += f"  ({r['false_alarms'] / (r['normal_windows'] * step_s / 86400):.1f} per day of normal operation)"
    print(line)
    for i in r["incidents"]:
        when = f"detected, +{i['delay_rows']} rows after onset" if i["detected"] else "MISSED"
        print(f"  {i['class']} incident, {i['length']:,} rows: {when}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roles", default="roles")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    m = load_manifest(a.roles)
    window = json.load(open(os.path.join(a.out, "test_state.json")))["blueprint"]["values"]["window_size"]
    total = Counter()
    results, incidents = {}, []
    for f in m["delivery"]:
        out = os.path.join(a.out, "delivery", os.path.basename(f["file"]))
        if not os.path.exists(out):
            sys.exit(f"no {out}: run deliver.py first")
        r = score_output(out, os.path.join(a.roles, f["labels"]), window, m["background"])
        results[f["file"]] = r
        report(f["file"], r, None if "incident" in f else m["step_s"])
        total.update({k: r[k] for k in ("tp", "fn", "fp", "tn", "false_alarms", "normal_windows", "windows")})
        incidents += r["incidents"]
    p, rec, f1 = prf(total)
    caught = [i for i in incidents if i["detected"]]
    print(f"\npooled over {len(results)} files: event F1 {f1:.4f} (precision {p:.4f}, recall {rec:.4f}); "
          f"false alarms {total['false_alarms']:,} / {total['normal_windows']:,}; "
          f"incidents detected {len(caught)} of {len(incidents)}")
    json.dump({"window_size": window, "files": results, "pooled": dict(total)},
              open(os.path.join(a.out, "evaluation.json"), "w"), indent=1)
    print(f"wrote {a.out}/evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
