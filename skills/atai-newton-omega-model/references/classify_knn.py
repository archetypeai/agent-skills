"""
Machine-state classification on Omega embeddings — client-side KNN, held-out.

The managed `machine-state-classification` batch pipeline, done over Direct
Query: embed short windows with Omega, then KNN against a small n-shot library
of labelled windows. No batch job, no lens session.

This is a genuine held-out evaluation:
  * Library (n-shot): contiguous windows from the labelled shot files
    `bearing_healthy.csv` / `bearing_degraded.csv`.
  * Test: windows from `bearing_inference.csv` (sensors only, NO label column —
    so leakage is structurally impossible) whose timestamps are DISJOINT from
    the shot files, scored against ground truth in `bearing_labels.csv`.

Normalization follows the data-prep / omega-1-4-preflight convention: fit ONE
per-channel scaler (mean/std) on the n-shot pool, apply it to every window, and
call /query with `normalize_input=false`. Windows that would span a timestamp
gap are skipped (Omega reads a window as a contiguous series).

The shipped `bearing_inference.csv` holds ~1000 non-overlapping windows. The
default run embeds all of them — that's ~1000 independent /query calls, ~6–7 min
at 8-way parallel. Dial it down for a quick check, or point at your own data:

    # full shipped eval (~1000 windows, several minutes)
    python classify_knn.py

    # quick check (50 windows, ~30s)
    python classify_knn.py --max-windows 50

    # your own data
    python classify_knn.py --inference my.csv --labels my_labels.csv --workers 8

Embeds are independent /query calls, so `--workers` fans them out concurrently.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from _common import MODEL, WINDOW, banner, client, read_series, read_series_and_time, window_at

SAMPLE = Path(__file__).parent / "sample_data"
HEALTHY_SHOT = SAMPLE / "bearing_healthy.csv"
DEGRADED_SHOT = SAMPLE / "bearing_degraded.csv"
DEFAULT_INFERENCE = SAMPLE / "bearing_inference.csv"  # sensors only — NO label column
DEFAULT_LABELS = SAMPLE / "bearing_labels.csv"        # ground truth, used only for scoring

LIB_STRIDE = 256  # overlapping windows are fine for the n-shot library
K = 3


def fit_scaler(series: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a channel-first series (the StandardScaler)."""
    a = np.asarray(series, dtype=float)
    return a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True) + 1e-9


def _embed_resilient(scaled_window, retries: int = 4):
    """One /query embed with retry on rate-limit/5xx. Returns the L2 joint feature, or None on failure."""
    endpoint, headers = client()
    body = {
        "query": "",
        "model": MODEL,
        "normalize_input": False,
        "events": [{"type": "data.numeric_array", "event_data": {"contents": scaled_window}}],
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{endpoint}/query", headers=headers, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.ok:
            resp = r.json().get("response", {})
            emb = resp.get("response") if isinstance(resp, dict) else resp
            feat = np.concatenate([np.asarray(ch, dtype=float) for ch in emb])
            return feat / (np.linalg.norm(feat) + 1e-9)
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1))
            continue
        return None
    return None


def feature(window, mean, std):
    return _embed_resilient(((np.asarray(window, dtype=float) - mean) / std).tolist())


def features_parallel(series, starts, mean, std, workers):
    """Embed many windows concurrently. Returns {start: feature} (skips failures)."""
    out: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(lambda s: (s, feature(window_at(series, s, WINDOW), mean, std)), starts)
        for s, feat in results:
            if feat is not None:
                out[s] = feat
    return out


def contiguous_starts(ts: list[int], window: int = WINDOW) -> list[int]:
    """Non-overlapping window starts that do NOT span a timestamp gap (ts step != 1)."""
    n = len(ts)
    starts: list[int] = []
    s = 0
    while s + window <= n:
        if ts[s + window - 1] - ts[s] == window - 1:
            starts.append(s)
            s += window
        else:
            gap = next((i for i in range(s + 1, s + window) if ts[i] - ts[i - 1] != 1), None)
            s = gap if gap is not None else s + 1
    return starts


def even_subsample(items: list, k: int) -> list:
    """Evenly spread `k` items across the list (to cover the whole timeline)."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(int(i) for i in idx))]


def load_labels(path: Path, needed: set[int] | None = None) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, []) or []
        label_idx = next((i for i, h in enumerate(header) if h.strip().lower() == "label"), len(header) - 1)
        for row in reader:
            if len(row) <= label_idx:
                continue
            ts = int(float(row[0]))
            if needed is None or ts in needed:
                labels[ts] = row[label_idx].strip()
    return labels


def knn_classify(feat, lib_feats, lib_labels, k=K):
    dists = np.linalg.norm(lib_feats - feat, axis=1)
    votes = [lib_labels[i] for i in np.argsort(dists)[:k]]
    return max(set(votes), key=votes.count)


def metrics(truths: list[str], preds: list[str], positive: str = "degraded") -> dict:
    """Binary classification metrics with `positive` as the positive class."""
    tp = sum(t == positive and p == positive for t, p in zip(truths, preds))
    fp = sum(t != positive and p == positive for t, p in zip(truths, preds))
    fn = sum(t == positive and p != positive for t, p in zip(truths, preds))
    tn = sum(t != positive and p != positive for t, p in zip(truths, preds))
    n = len(truths)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / n if n else 0.0,
            "precision": precision, "recall": recall, "f1": f1}


def print_report(truths: list[str], preds: list[str], positive: str = "degraded") -> None:
    m = metrics(truths, preds, positive)
    n = len(truths)
    print(f"\nEvaluated {n} held-out windows "
          f"({truths.count('healthy')} healthy, {truths.count('degraded')} degraded)")
    print(f"Confusion matrix (positive = {positive!r}): "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  accuracy : {m['accuracy']:.3f}  ({m['tp'] + m['tn']}/{n})")
    print(f"  precision: {m['precision']:.3f}")
    print(f"  recall   : {m['recall']:.3f}")
    print(f"  f1       : {m['f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out machine-state eval over Omega embeddings.")
    parser.add_argument("--inference", default=str(DEFAULT_INFERENCE), help="test sensor CSV")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="ground-truth labels CSV (timestamp,label)")
    parser.add_argument("--max-windows", type=int, default=1000, help="cap on test windows (default 1000)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent /query embeds")
    args = parser.parse_args()

    healthy = read_series(HEALTHY_SHOT)
    degraded = read_series(DEGRADED_SHOT)
    banner(f"Held-out machine-state eval — Omega embeddings + {K}-NN vs ground truth")

    # Global per-channel scaler fit on the n-shot pool only (no test leakage).
    mean, std = fit_scaler(np.concatenate([np.asarray(healthy), np.asarray(degraded)], axis=1).tolist())

    print("Building n-shot library from shot files (scaled, normalize_input=false)...")
    lib_starts_h = list(range(0, len(healthy[0]) - WINDOW + 1, LIB_STRIDE))
    lib_starts_d = list(range(0, len(degraded[0]) - WINDOW + 1, LIB_STRIDE))
    lib = [(feature(window_at(healthy, s, WINDOW), mean, std), "healthy") for s in lib_starts_h]
    lib += [(feature(window_at(degraded, s, WINDOW), mean, std), "degraded") for s in lib_starts_d]
    lib = [(f, lbl) for f, lbl in lib if f is not None]
    lib_feats = np.vstack([f for f, _ in lib])
    lib_labels = [lbl for _, lbl in lib]
    print(f"Library: {lib_feats.shape[0]} windows x {lib_feats.shape[1]} dims\n")

    print(f"Loading test series {Path(args.inference).name} ...")
    ts, inference = read_series_and_time(args.inference)
    all_starts = contiguous_starts(ts)
    starts = even_subsample(all_starts, args.max_windows)
    print(f"{len(all_starts)} non-overlapping contiguous windows available; "
          f"evaluating {len(starts)} (max-windows={args.max_windows}, {args.workers}-way parallel)")
    truth_by_ts = load_labels(Path(args.labels), needed={ts[s] for s in starts})

    t0 = time.time()
    feats = features_parallel(inference, starts, mean, std, args.workers)
    truths, preds = [], []
    for s in starts:
        if s not in feats or ts[s] not in truth_by_ts:
            continue
        truths.append(truth_by_ts[ts[s]])
        preds.append(knn_classify(feats[s], lib_feats, lib_labels))
    print(f"Embedded + classified {len(preds)} windows in {time.time() - t0:.1f}s")
    print_report(truths, preds, positive="degraded")


if __name__ == "__main__":
    main()
