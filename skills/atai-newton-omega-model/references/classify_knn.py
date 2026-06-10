"""
Classification on Omega embeddings — client-side KNN, held-out.

The managed batch classification pipeline, done over Direct
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from archetypeai.api_client import ArchetypeAI

from _common import WINDOW, banner, embed, make_client, read_series, read_series_and_time, window_at

SAMPLE = Path(__file__).parent / "sample_data"
HEALTHY_SHOT = SAMPLE / "bearing_healthy.csv"
DEGRADED_SHOT = SAMPLE / "bearing_degraded.csv"
DEFAULT_INFERENCE = SAMPLE / "bearing_inference.csv"  # sensors only — NO label column
DEFAULT_LABELS = SAMPLE / "bearing_labels.csv"        # ground truth, used only for scoring

LIB_STRIDE = 256  # overlapping windows are fine for the n-shot library
K_NEIGHBORS = 3


def fit_scaler(series: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a channel-first series (the StandardScaler)."""
    series_array = np.asarray(series, dtype=float)
    return (
        series_array.mean(axis=1, keepdims=True),
        series_array.std(axis=1, keepdims=True) + 1e-9,
    )


def _embed_resilient(client: ArchetypeAI, scaled_window, retries: int = 4):
    """One /query embed with retry/backoff on transient failures.

    The official client already retries 5xx/429 internally; this adds a
    backoff loop around hard failures. Returns the L2 joint feature, or
    None on persistent failure.
    """
    for attempt in range(retries):
        try:
            embeddings, _, _ = embed(client, scaled_window)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            continue
        if not embeddings:
            return None
        joint_feature = np.concatenate(
            [np.asarray(channel_embedding, dtype=float) for channel_embedding in embeddings]
        )
        return joint_feature / (np.linalg.norm(joint_feature) + 1e-9)
    return None


def feature(client, window, mean, std):
    return _embed_resilient(client, ((np.asarray(window, dtype=float) - mean) / std).tolist())


def _embed_start(client, series, start, mean, std):
    return start, feature(client, window_at(series, start, WINDOW), mean, std)


def features_parallel(client, series, starts, mean, std, workers):
    """Embed many windows concurrently, with a live progress line. Returns {start: feature}."""
    features_by_start: dict[int, np.ndarray] = {}
    total = len(starts)
    failures = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_embed_start, client, series, start, mean, std) for start in starts
        ]
        for done, future in enumerate(as_completed(futures), 1):
            start, joint_feature = future.result()
            if joint_feature is not None:
                features_by_start[start] = joint_feature
            else:
                failures += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                fail_note = f", {failures} failed" if failures else ""
                print(f"\r  embedded {done}/{total} ({100 * done // total}%)  "
                      f"{elapsed:4.0f}s elapsed, ~{eta:4.0f}s left{fail_note}   ",
                      end="", flush=True)
    print()  # finish the progress line
    return features_by_start


def contiguous_starts(timestamps: list[int], window: int = WINDOW) -> list[int]:
    """Non-overlapping window starts that do NOT span a timestamp gap (step != 1)."""
    series_length = len(timestamps)
    starts: list[int] = []
    start = 0
    while start + window <= series_length:
        if timestamps[start + window - 1] - timestamps[start] == window - 1:
            starts.append(start)
            start += window
        else:
            gap_index = next(
                (
                    row_index
                    for row_index in range(start + 1, start + window)
                    if timestamps[row_index] - timestamps[row_index - 1] != 1
                ),
                None,
            )
            start = gap_index if gap_index is not None else start + 1
    return starts


def even_subsample(items: list, target_count: int) -> list:
    """Evenly spread `target_count` items across the list (to cover the whole timeline)."""
    if len(items) <= target_count:
        return items
    spread = np.linspace(0, len(items) - 1, target_count).round().astype(int)
    return [items[index] for index in sorted(set(int(index) for index in spread))]


def load_labels(path: Path, needed: set[int] | None = None) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(path, newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader, []) or []
        label_column = next(
            (
                column_index
                for column_index, column_name in enumerate(header)
                if column_name.strip().lower() == "label"
            ),
            len(header) - 1,
        )
        for row in reader:
            if len(row) <= label_column:
                continue
            timestamp = int(float(row[0]))
            if needed is None or timestamp in needed:
                labels[timestamp] = row[label_column].strip()
    return labels


def knn_classify(joint_feature, lib_feats, lib_labels, k_neighbors=K_NEIGHBORS):
    distances = np.linalg.norm(lib_feats - joint_feature, axis=1)
    votes = [lib_labels[index] for index in np.argsort(distances)[:k_neighbors]]
    return max(set(votes), key=votes.count)


def metrics(truths: list[str], preds: list[str], positive: str = "degraded") -> dict:
    """Binary classification metrics with `positive` as the positive class."""
    true_positives = sum(
        truth == positive and pred == positive for truth, pred in zip(truths, preds)
    )
    false_positives = sum(
        truth != positive and pred == positive for truth, pred in zip(truths, preds)
    )
    false_negatives = sum(
        truth == positive and pred != positive for truth, pred in zip(truths, preds)
    )
    true_negatives = sum(
        truth != positive and pred != positive for truth, pred in zip(truths, preds)
    )
    total = len(truths)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": true_positives, "fp": false_positives,
            "fn": false_negatives, "tn": true_negatives,
            "accuracy": (true_positives + true_negatives) / total if total else 0.0,
            "precision": precision, "recall": recall, "f1": f1}


def print_report(truths: list[str], preds: list[str], positive: str = "degraded") -> None:
    report = metrics(truths, preds, positive)
    total = len(truths)
    print(f"\nEvaluated {total} held-out windows "
          f"({truths.count('healthy')} healthy, {truths.count('degraded')} degraded)")
    print(f"Confusion matrix (positive = {positive!r}): "
          f"TP={report['tp']} FP={report['fp']} FN={report['fn']} TN={report['tn']}")
    print(f"  accuracy : {report['accuracy']:.3f}  ({report['tp'] + report['tn']}/{total})")
    print(f"  precision: {report['precision']:.3f}")
    print(f"  recall   : {report['recall']:.3f}")
    print(f"  f1       : {report['f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out classification eval over Omega embeddings.")
    parser.add_argument("--inference", default=str(DEFAULT_INFERENCE), help="test sensor CSV")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="ground-truth labels CSV (timestamp,label)")
    parser.add_argument("--max-windows", type=int, default=1000, help="cap on test windows (default 1000)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent /query embeds")
    args = parser.parse_args()

    client = make_client()
    healthy = read_series(HEALTHY_SHOT)
    degraded = read_series(DEGRADED_SHOT)
    banner(f"Held-out classification eval — Omega embeddings + {K_NEIGHBORS}-NN vs ground truth")

    # Global per-channel scaler fit on the n-shot pool only (no test leakage).
    mean, std = fit_scaler(np.concatenate([np.asarray(healthy), np.asarray(degraded)], axis=1).tolist())

    print("Building n-shot library from shot files (scaled, normalize_input=false)...")
    healthy_starts = list(range(0, len(healthy[0]) - WINDOW + 1, LIB_STRIDE))
    degraded_starts = list(range(0, len(degraded[0]) - WINDOW + 1, LIB_STRIDE))
    library = [
        (feature(client, window_at(healthy, start, WINDOW), mean, std), "healthy")
        for start in healthy_starts
    ]
    library += [
        (feature(client, window_at(degraded, start, WINDOW), mean, std), "degraded")
        for start in degraded_starts
    ]
    library = [(joint_feature, label) for joint_feature, label in library if joint_feature is not None]
    lib_feats = np.vstack([joint_feature for joint_feature, _ in library])
    lib_labels = [label for _, label in library]
    print(f"Library: {lib_feats.shape[0]} windows x {lib_feats.shape[1]} dims\n")

    print(f"Loading test series {Path(args.inference).name} ...")
    timestamps, inference = read_series_and_time(args.inference)
    all_starts = contiguous_starts(timestamps)
    starts = even_subsample(all_starts, args.max_windows)
    print(f"{len(all_starts)} non-overlapping contiguous windows available; "
          f"evaluating {len(starts)} (max-windows={args.max_windows}, {args.workers}-way parallel)")
    truth_by_timestamp = load_labels(Path(args.labels), needed={timestamps[start] for start in starts})

    start_time = time.time()
    features_by_start = features_parallel(client, inference, starts, mean, std, args.workers)
    truths, preds = [], []
    for start in starts:
        if start not in features_by_start or timestamps[start] not in truth_by_timestamp:
            continue
        truths.append(truth_by_timestamp[timestamps[start]])
        preds.append(knn_classify(features_by_start[start], lib_feats, lib_labels))
    print(f"Embedded + classified {len(preds)} windows in {time.time() - start_time:.1f}s")
    print_report(truths, preds, positive="degraded")


if __name__ == "__main__":
    main()
