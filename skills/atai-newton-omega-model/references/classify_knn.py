"""
Classification on Omega embeddings — client-side KNN, held-out.

The managed batch classification pipeline, done over Model Serving: embed
short windows with Omega 1.5, then KNN against a small n-shot library of
labelled windows. No batch job.

This is a genuine held-out evaluation:
  * Library (n-shot): contiguous windows from the labelled shot files
    `bearing_healthy.csv` / `bearing_degraded.csv`.
  * Test: windows from `bearing_inference.csv` (sensors only, NO label column —
    so leakage is structurally impossible) whose timestamps are DISJOINT from
    the shot files, scored against ground truth in `bearing_labels.csv`.

Each window length gets its own Model Serving session (`window_size` =
`step_size` = the length), used for both the library and the test windows,
and cancelled when that eval finishes. All of a window's channels go in one
request.

Normalization follows the data-prep / omega-1-5-preflight convention: fit ONE
per-channel scaler (mean/std) on the n-shot pool and apply it to every window.
Windows that would span a timestamp gap are skipped (Omega reads a window as
a contiguous series).

The shipped `bearing_inference.csv` holds ~1000 non-overlapping 1024-timestep
windows (more at shorter lengths). The default run embeds up to 1000 of them —
one session query each, a couple of minutes at 8-way parallel:

    # full shipped eval (up to 1000 windows)
    python classify_knn.py

    # quick check (50 windows)
    python classify_knn.py --max-windows 50

    # compare window lengths (one full eval each + a comparison table at the end)
    python classify_knn.py --window-size 16 64 256 1024

    # your own data
    python classify_knn.py --inference my.csv --labels my_labels.csv --workers 8

Queries are independent, so `--workers` sends several to the session at once.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from _common import WINDOW, OmegaSession, banner, make_client, read_series, read_series_and_time, window_at

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


def _embed_resilient(session: OmegaSession, scaled_window, retries: int = 4):
    """One session query with retry/backoff on transient failures.

    The official client already retries 5xx/429 internally, and the session
    reopens itself if it has ended; this adds a backoff loop around hard
    failures. Returns the L2 joint feature, or None on persistent failure.
    """
    for attempt in range(retries):
        try:
            embeddings = session.embed(scaled_window)
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


def feature(session, window, mean, std):
    return _embed_resilient(session, ((np.asarray(window, dtype=float) - mean) / std).tolist())


def _embed_start(session, series, start, mean, std, window_size):
    return start, feature(session, window_at(series, start, window_size), mean, std)


def features_parallel(session, series, starts, mean, std, workers, window_size):
    """Embed many windows concurrently, with a live progress line. Returns {start: feature}."""
    features_by_start: dict[int, np.ndarray] = {}
    total = len(starts)
    failures = 0
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_embed_start, session, series, start, mean, std, window_size) for start in starts
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


def contiguous_starts(timestamps: list[int], window_size: int = WINDOW) -> list[int]:
    """Non-overlapping window starts that do NOT span a timestamp gap (step != 1)."""
    series_length = len(timestamps)
    starts: list[int] = []
    start = 0
    while start + window_size <= series_length:
        if timestamps[start + window_size - 1] - timestamps[start] == window_size - 1:
            starts.append(start)
            start += window_size
        else:
            gap_index = next(
                (
                    row_index
                    for row_index in range(start + 1, start + window_size)
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


def load_labels(
    path: Path, needed: set[int] | None = None, label_column: str = "label"
) -> dict[int, str]:
    """Load {timestamp: label} from a ground-truth CSV.

    `label_column` names the column holding the labels (default "label");
    falls back to the last column if the name is not found.
    """
    labels: dict[int, str] = {}
    with open(path, newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader, []) or []
        label_index = next(
            (
                column_index
                for column_index, column_name in enumerate(header)
                if column_name.strip().lower() == label_column.lower()
            ),
            len(header) - 1,
        )
        for row in reader:
            if len(row) <= label_index:
                continue
            timestamp = int(float(row[0]))
            if needed is None or timestamp in needed:
                labels[timestamp] = row[label_index].strip()
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


def print_report(truths: list[str], preds: list[str], positive: str = "degraded") -> dict:
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
    return report


def run_eval(client, args, window_size: int, healthy, degraded, mean, std) -> dict:
    """One full library-build + held-out eval at a single window length.

    Returns the metrics dict plus run context for the comparison table.
    """
    banner(f"Held-out classification eval — Omega embeddings + {K_NEIGHBORS}-NN vs ground truth "
           f"(window_size={window_size})")

    with OmegaSession(client, window_size=window_size) as session:
        return _run_eval(session, args, window_size, healthy, degraded, mean, std)


def _run_eval(session, args, window_size, healthy, degraded, mean, std) -> dict:
    print(f"Building n-shot library from shot files (session {session.session_id}, "
          f"window_size={window_size})...")
    healthy_starts = list(range(0, len(healthy[0]) - window_size + 1, LIB_STRIDE))
    degraded_starts = list(range(0, len(degraded[0]) - window_size + 1, LIB_STRIDE))
    library = [
        (feature(session, window_at(healthy, start, window_size), mean, std), "healthy")
        for start in healthy_starts
    ]
    library += [
        (feature(session, window_at(degraded, start, window_size), mean, std), "degraded")
        for start in degraded_starts
    ]
    library = [(joint_feature, label) for joint_feature, label in library if joint_feature is not None]
    lib_feats = np.vstack([joint_feature for joint_feature, _ in library])
    lib_labels = [label for _, label in library]
    print(f"Library: {lib_feats.shape[0]} windows x {lib_feats.shape[1]} dims\n")

    print(f"Loading test series {Path(args.inference).name} ...")
    timestamps, inference = read_series_and_time(args.inference)
    all_starts = contiguous_starts(timestamps, window_size)
    starts = even_subsample(all_starts, args.max_windows)
    print(f"{len(all_starts)} non-overlapping contiguous windows available; "
          f"evaluating {len(starts)} (max-windows={args.max_windows}, {args.workers}-way parallel)")
    truth_by_timestamp = load_labels(
        Path(args.labels), needed={timestamps[start] for start in starts}, label_column=args.label_column
    )

    start_time = time.time()
    features_by_start = features_parallel(session, inference, starts, mean, std, args.workers, window_size)
    truths, preds = [], []
    for start in starts:
        if start not in features_by_start or timestamps[start] not in truth_by_timestamp:
            continue
        truths.append(truth_by_timestamp[timestamps[start]])
        preds.append(knn_classify(features_by_start[start], lib_feats, lib_labels))
    print(f"Embedded + classified {len(preds)} windows in {time.time() - start_time:.1f}s")
    report = print_report(truths, preds, positive="degraded")
    print()
    return {"window_size": window_size, "n_windows": len(preds),
            "n_library": lib_feats.shape[0], **report}


def comparison_table(rows: list) -> str:
    """Format per-window-size eval results as an aligned comparison table."""
    header = f"{'window':>7} {'library':>8} {'windows':>8} {'accuracy':>9} {'precision':>10} {'recall':>7} {'f1':>6}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['window_size']:>7} {row['n_library']:>8} {row['n_windows']:>8} "
            f"{row['accuracy']:>9.3f} {row['precision']:>10.3f} {row['recall']:>7.3f} {row['f1']:>6.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out classification eval over Omega embeddings.")
    parser.add_argument("--inference", default=str(DEFAULT_INFERENCE), help="test sensor CSV")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="ground-truth labels CSV (timestamp,label)")
    parser.add_argument("--label-column", default="label", help="name of the label column in --labels (default 'label')")
    parser.add_argument("--max-windows", type=int, default=1000, help="cap on test windows (default 1000)")
    parser.add_argument("--workers", type=int, default=8, help="session queries in flight")
    parser.add_argument("--window-size", type=int, nargs="+", default=[WINDOW],
                        help=f"one or more window lengths to evaluate (each applied to BOTH the\n"
                             f"n-shot library and the test windows — mixing lengths within one eval\n"
                             f"would invalidate KNN distances). With several values, each runs as a\n"
                             f"separate full eval and a comparison table is printed at the end.\n"
                             f"Each length gets its own session. Range 16-{WINDOW} (default {WINDOW})")
    args = parser.parse_args()

    client = make_client()
    healthy = read_series(HEALTHY_SHOT)
    degraded = read_series(DEGRADED_SHOT)

    # Global per-channel scaler fit on the n-shot pool only (no test leakage).
    # Window-length independent, so fit once for all runs.
    mean, std = fit_scaler(np.concatenate([np.asarray(healthy), np.asarray(degraded)], axis=1).tolist())

    rows = [
        run_eval(client, args, window_size, healthy, degraded, mean, std)
        for window_size in args.window_size
    ]

    if len(rows) > 1:
        banner("Window-size comparison")
        print(comparison_table(rows))


if __name__ == "__main__":
    main()
