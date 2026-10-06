"""
Omega senses, Nemotron explains — end to end on the bundled bearing sample.

  1. Fit one per-channel scaler on the n-shot pool (healthy + degraded shot files).
  2. Build a KNN library: 256-step windows from each shot file, embedded with
     Omega (one /query per channel), concatenated and L2-normalised.
  3. Classify each window of `bearing_incoming.csv` (two held-out windows: one
     healthy, one degraded) by KNN vote.
  4. For a non-healthy verdict, compute plain statistics for the window and for
     the healthy pool, and ask Nemotron for a brief: observation, likely cause,
     checks. The reply is parsed and validated before it is printed.

Nemotron gets the verdict and the statistics — never the embeddings.

    python explain_knn.py
    python explain_knn.py --always-brief          # brief every window, healthy too
    python explain_knn.py my_incoming.csv --healthy my_healthy.csv --faulty my_faulty.csv \\
        --context "centrifugal pump, 3 accelerometers + motor current"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (
    banner,
    embed,
    make_client,
    nemotron_brief,
    nemotron_model,
    read_series,
    window_at,
    window_stats,
)

SAMPLE = Path(__file__).parent / "sample_data"
WINDOW = 256  # the Omega skill's window sweep: 256 beats 1024 on this bearing data
K = 3
DEFAULT_CONTEXT = (
    "Rolling-element bearing test rig (NASA IMS run-to-failure), 4 accelerometer "
    "channels, one per bearing, sampled at 20 kHz. Values are acceleration in g."
)


def fit_scaler(series: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(series, dtype=float)
    return array.mean(axis=1, keepdims=True), array.std(axis=1, keepdims=True) + 1e-9


def joint_feature(client, window, mean, std) -> np.ndarray:
    scaled = ((np.asarray(window, dtype=float) - mean) / std).tolist()
    vector = np.concatenate([np.asarray(v, dtype=float) for v in embed(client, scaled)])
    return vector / (np.linalg.norm(vector) + 1e-9)


def knn(feature: np.ndarray, library: list[tuple[np.ndarray, str]], k: int = K) -> tuple[str, dict[str, int]]:
    nearest = sorted(library, key=lambda item: float(np.linalg.norm(item[0] - feature)))[:k]
    votes: dict[str, int] = {}
    for _, label in nearest:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get), votes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("incoming", nargs="?", default=SAMPLE / "bearing_incoming.csv")
    parser.add_argument("--healthy", default=SAMPLE / "bearing_healthy.csv")
    parser.add_argument("--faulty", default=SAMPLE / "bearing_degraded.csv")
    parser.add_argument("--faulty-label", default="degraded")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="one or two sentences about the asset")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--always-brief", action="store_true", help="also brief healthy verdicts")
    args = parser.parse_args()

    client = make_client()
    names, healthy = read_series(args.healthy)
    _, faulty = read_series(args.faulty)
    _, incoming = read_series(args.incoming)

    pool = [h + f for h, f in zip(healthy, faulty)]
    mean, std = fit_scaler(pool)

    banner(f"Library: {args.window}-step windows, Omega {len(names)} channels")
    library: list[tuple[np.ndarray, str]] = []
    for label, series in (("healthy", healthy), (args.faulty_label, faulty)):
        for start in range(0, len(series[0]) - args.window + 1, args.window):
            library.append((joint_feature(client, window_at(series, start, args.window), mean, std), label))
    print(f"{len(library)} library windows ({len(library) * len(names)} /query calls)")

    # Baseline statistics come from the raw (unscaled) healthy pool, so the numbers keep their units.
    baseline = window_stats(names, healthy)

    banner(f"Classify + explain · Nemotron {nemotron_model()}")
    for start in range(0, len(incoming[0]) - args.window + 1, args.window):
        window = window_at(incoming, start, args.window)
        verdict, votes = knn(joint_feature(client, window, mean, std), library)
        print(f"\nwindow rows {start}–{start + args.window - 1}: {verdict}  votes={votes}")
        if verdict == "healthy" and not args.always_brief:
            print("  healthy — no brief requested")
            continue
        started = time.time()
        brief = nemotron_brief(args.context, verdict, votes, window_stats(names, window), baseline)
        print(f"  Nemotron brief ({time.time() - started:.1f} s):")
        print("  " + json.dumps(brief, indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
