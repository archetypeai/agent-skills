#!/usr/bin/env python3
"""Step 2: can the sensor tell your states apart? Check before you commit to a state list.

A state that isn't visible in the signal caps every model's macro-F1: it is one n-th
of the score and nothing can recover it. (In the LARCO washing-machine example,
"heating" — the heater switching on during wash — was a separate label, but the drum
moves the same either way: held-out recordings told it from wash at 0.62 balanced
accuracy, where 0.5 is chance. It was folded into wash.)

This is a quick, model-free check, not a benchmark: simple per-window features
(per-channel log RMS + 16-band log power spectrum), a kNN classifier, and
leave-one-recording-out scoring over the library and validation recordings, with the
training classes balanced so no state wins by count. Per state it reports the recall
on held-out recordings and where the misses go. A state near chance, or one whose
windows mostly land in one other state, is a candidate to merge or drop — or to
measure with another sensor.

    python check_states.py --index recordings.csv --prepared prepared --window 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osm_common import load_prepared, read_index  # noqa: E402


def features(w):
    level = np.log(np.sqrt(((w - w.mean(0)) ** 2).mean(0)) + 1e-6)
    z = (w - w.mean(0)) / np.maximum(w.std(0), 1e-9)
    p = np.abs(np.fft.rfft(z, axis=0)) ** 2
    edges = np.unique(np.geomspace(1, p.shape[0] - 1, 17).astype(int))
    fft = np.concatenate([np.log1p(p[edges[i]:edges[i + 1]].sum(0)) for i in range(len(edges) - 1)])
    return np.concatenate([level, fft])


def windows(df, channels, window):
    """Features and label of every non-overlapping, single-state window within a segment."""
    x, lab, seg = df[channels].to_numpy(np.float64), df.label.astype(str).to_numpy(), df.segment.to_numpy()
    F, y = [], []
    for s in range(0, len(x) - window + 1, window):
        if seg[s] == seg[s + window - 1] and (lab[s:s + window] == lab[s]).all():
            F.append(features(x[s:s + window]))
            y.append(lab[s])
    return np.array(F), np.array(y)


def knn(Xtr, ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    out = []
    block = max(1, int(2.5e8 / (8 * len(A) * A.shape[1])))      # keeps the distance work near 250 MB
    for i in range(0, len(B), block):
        d = np.abs(B[i:i + block, None, :] - A[None]).sum(2)
        nn = np.argsort(d, axis=1)[:, :k]
        vals = ytr[nn]
        out += [max(set(r), key=list(r).count) for r in vals]
    return np.array(out)


def check(idx, prepared, window, k, seed=0):
    rows = idx[idx.get("role", "").isin(["library", "validation"])] if "role" in idx and idx.role.str.len().any() else idx
    data = {}
    for rec in rows.recording:
        df = load_prepared(prepared, rec)
        channels = [c for c in df.columns if c not in ("timestamp", "label", "segment")]
        data[rec] = windows(df, channels, window)
    states = sorted(str(s) for s in set(np.concatenate([y for _, y in data.values() if len(y)])))
    rng = np.random.default_rng(seed)
    ys, ps = [], []
    for rec, (Fte, yte) in data.items():
        if not len(yte):
            continue
        tr = [(F, y) for r, (F, y) in data.items() if r != rec and len(y)]
        Ftr, ytr = np.concatenate([f for f, _ in tr]), np.concatenate([y for _, y in tr])
        have = [s for s in states if (ytr == s).any()]
        n = min((ytr == s).sum() for s in have)
        pick = np.concatenate([rng.choice(np.flatnonzero(ytr == s), n, replace=False) for s in have])
        ys.append(yte)
        ps.append(knn(Ftr[pick], ytr[pick], Fte, min(k, len(pick))))
    y, p = np.concatenate(ys), np.concatenate(ps)
    report = {}
    for s in states:
        m = y == s
        if not m.any():
            continue
        misses = {t: int(((p == t) & m).sum()) for t in states if t != s and ((p == t) & m).any()}
        worst = max(misses, key=misses.get) if misses else None
        report[s] = {"windows": int(m.sum()), "recall": round(float((p[m] == s).mean()), 3),
                     "chance": round(1 / len(states), 3), "misses": misses,
                     "mostly_confused_with": worst,
                     "share_to_worst": round(misses[worst] / m.sum(), 3) if worst else 0.0}
    return {"window": window, "k": k, "recordings": len(data), "states": states,
            "balanced_accuracy": round(float(np.mean([r["recall"] for r in report.values()])), 3), "per_state": report}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--window", type=int, default=512, help="rows per window (use the window you expect to train with)")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--warn-recall", type=float, default=0.5, help="flag states whose held-out recall is below this")
    ap.add_argument("--out", help="write the report as JSON")
    a = ap.parse_args(argv)
    rep = check(read_index(a.index), a.prepared, a.window, a.k)
    print(f"leave-one-recording-out over {rep['recordings']} recordings, window {a.window}, kNN k={a.k}, "
          f"classes balanced; chance = {1 / len(rep['states']):.2f}")
    flagged = []
    for s, r in rep["per_state"].items():
        flag = r["recall"] < a.warn_recall or r["share_to_worst"] > 0.4
        flagged += [s] if flag else []
        print(f"  {'WARN' if flag else 'ok  '}  {s:<16} recall {r['recall']:.2f}  ({r['windows']:,} windows)"
              + (f"  → mostly {r['mostly_confused_with']} ({r['share_to_worst']:.0%})" if r["mostly_confused_with"] else ""))
    print(f"balanced accuracy {rep['balanced_accuracy']:.2f}")
    if flagged:
        print(f"look into: {flagged}. A state the sensor can't see caps every model (merge it, drop it or add a\n"
              f"sensor). But a low recall can also mean too few recordings, or recordings that differ\n"
              f"(another unit, another load): check which recordings the misses come from before deciding.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    main()
