"""
Machine-state classification on Omega embeddings — client-side KNN, held-out.

The managed `machine-state-classification` batch pipeline, done over Direct
Query: embed short windows with Omega, then KNN against a small n-shot library
of labelled windows. No batch job, no lens session.

This script is a genuine held-out evaluation:
  * Library (n-shot): contiguous windows from the labelled shot files
    `bearing_healthy.csv` / `bearing_degraded.csv`.
  * Test: contiguous windows from `bearing_inference_subset.csv` — a subset of
    the inference timeline whose timestamps are DISJOINT from the shot files
    (no leakage) — scored against ground-truth labels in
    `bearing_labels_subset.csv` (carved from `bearing_raw_labeled.csv`).

Normalization follows the data-prep / omega-1-4-preflight convention: fit ONE
per-channel scaler (mean/std) on the n-shot pool, apply it to every window, and
call /query with `normalize_input=false`. (Per-window `normalize_input=true`
would erase cross-window amplitude — the signal that separates states.)
Windows that would span a timestamp gap are skipped (Omega reads a window as a
contiguous series).

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY
    python classify_knn.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from _common import WINDOW, banner, embed, read_series, read_series_and_time, window_at

SAMPLE = Path(__file__).parent / "sample_data"
HEALTHY_SHOT = SAMPLE / "bearing_healthy.csv"
DEGRADED_SHOT = SAMPLE / "bearing_degraded.csv"
INFERENCE = SAMPLE / "bearing_inference_subset.csv"
LABELS = SAMPLE / "bearing_labels_subset.csv"

LIB_STRIDE = 256  # overlapping windows are fine for the n-shot library
K = 3


def fit_scaler(series: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a channel-first series (the StandardScaler)."""
    a = np.asarray(series, dtype=float)
    return a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True) + 1e-9


def scaled_feature(window: list[list[float]], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the global scaler, embed (normalize_input=false), fold channels into one L2 feature."""
    scaled = ((np.asarray(window, dtype=float) - mean) / std).tolist()
    embeddings, _warnings, _ms = embed(scaled, normalize_input=False)
    feat = np.concatenate([np.asarray(ch, dtype=float) for ch in embeddings])
    return feat / (np.linalg.norm(feat) + 1e-9)


def library_windows(series, mean, std, label, stride=LIB_STRIDE):
    n = len(series[0])
    starts = range(0, n - WINDOW + 1, stride)
    return [(scaled_feature(window_at(series, s, WINDOW), mean, std), label) for s in starts]


def contiguous_starts(ts: list[int], window: int = WINDOW) -> list[int]:
    """Non-overlapping window starts that do NOT span a timestamp gap (ts step != 1)."""
    n = len(ts)
    starts: list[int] = []
    s = 0
    while s + window <= n:
        if ts[s + window - 1] - ts[s] == window - 1:
            starts.append(s)
            s += window
        else:  # jump past the first gap inside this window
            gap = next((i for i in range(s + 1, s + window) if ts[i] - ts[i - 1] != 1), None)
            s = gap if gap is not None else s + 1
    return starts


def load_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                labels[int(float(row[0]))] = row[1].strip()
    return labels


def knn_classify(feat, lib_feats, lib_labels, k=K):
    dists = np.linalg.norm(lib_feats - feat, axis=1)
    votes = [lib_labels[i] for i in np.argsort(dists)[:k]]
    return max(set(votes), key=votes.count)


def main() -> None:
    for p in (HEALTHY_SHOT, DEGRADED_SHOT, INFERENCE, LABELS):
        if not p.exists():
            sys.exit(f"Missing sample file: {p}")

    banner(f"Held-out machine-state eval — Omega embeddings + {K}-NN vs ground truth")

    healthy = read_series(HEALTHY_SHOT)
    degraded = read_series(DEGRADED_SHOT)
    # Global per-channel scaler fit on the n-shot pool only (no test leakage).
    mean, std = fit_scaler(np.concatenate([np.asarray(healthy), np.asarray(degraded)], axis=1).tolist())

    print("Building n-shot library from shot files (scaled, normalize_input=false)...")
    library = library_windows(healthy, mean, std, "healthy") + library_windows(degraded, mean, std, "degraded")
    lib_feats = np.vstack([f for f, _ in library])
    lib_labels = [lbl for _, lbl in library]
    print(f"Library: {lib_feats.shape[0]} windows x {lib_feats.shape[1]} dims "
          f"({len(healthy)} channels x 768, concatenated)\n")

    ts, inference = read_series_and_time(INFERENCE)
    truth_by_ts = load_labels(LABELS)
    starts = contiguous_starts(ts)
    print(f"Held-out test: {len(starts)} contiguous windows from bearing_inference_subset.csv")
    print("(timestamps disjoint from the shot files — genuine held-out, no leakage)\n")

    correct = 0
    for s in starts:
        truth = truth_by_ts.get(ts[s], "?")
        pred = knn_classify(scaled_feature(window_at(inference, s, WINDOW), mean, std), lib_feats, lib_labels)
        ok = pred == truth
        correct += ok
        print(f"  ts={ts[s]:>9}  truth={truth:<8} pred={pred:<8} {'OK' if ok else 'MISS'}")
    print(f"\nAccuracy vs ground-truth labels: {correct}/{len(starts)}")


if __name__ == "__main__":
    main()
