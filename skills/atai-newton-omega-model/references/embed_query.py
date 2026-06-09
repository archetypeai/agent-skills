"""
Omega embedding examples — turn a window of sensor readings into vectors,
via the cloud Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over /query.

Three patterns:
  1. Embed one multi-channel window (the basic call + output shape).
  2. Short windows (<1024) are zero-padded — what the warning means.
  3. `normalize_input` — z-norm the input window before encoding.

Data: a subset of the NASA IMS Bearing dataset (4 accelerometer channels). See
sample_data/README.md for attribution.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY
    python embed_query.py
    # or point at your own sensor CSV (timestamp column auto-skipped):
    python embed_query.py /path/to/series.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import EMBED_DIM, WINDOW, banner, embed, read_series, window_at

DEFAULT_CSV = Path(__file__).parent / "sample_data" / "bearing_healthy.csv"


def example_basic(series: list[list[float]]) -> None:
    banner(f"1. Embed one window — {len(series)} channels x {WINDOW} timesteps")
    w = window_at(series, start=0, window=WINDOW)
    embeddings, warnings, ms = embed(w)
    print(f"[{ms} ms]")
    print(f"input  : {len(w)} channels x {len(w[0])} timesteps")
    print(f"output : {len(embeddings)} embeddings x {len(embeddings[0])} dims "
          f"(one {EMBED_DIM}-d vector per channel)")
    print(f"channel 1 embedding[:5]: {[round(x, 4) for x in embeddings[0][:5]]}")
    if warnings:
        print(f"warnings: {warnings}")
    print()


def example_padding(series: list[list[float]]) -> None:
    banner("2. Short window (<1024) is zero-padded server-side")
    short = 256
    w = window_at(series, start=0, window=short)
    embeddings, warnings, ms = embed(w)
    print(f"[{ms} ms] sent {short} timesteps; output still {len(embeddings)} x {len(embeddings[0])}")
    print(f"warnings: {warnings or '(none)'}")
    print("Takeaway: feed WINDOW (1024) timesteps to use the encoder's native "
          "receptive field; shorter inputs work but are padded with zeros.\n")


def example_normalize(series: list[list[float]]) -> None:
    banner("3. normalize_input — PER-WINDOW z-norm (usually NOT what you want)")
    w = window_at(series, start=0, window=WINDOW)
    raw, _, _ = embed(w, normalize_input=False)
    norm, _, _ = embed(w, normalize_input=True)
    delta = sum(abs(a - b) for a, b in zip(raw[0], norm[0])) / len(raw[0])
    print(f"mean |Δ| on channel 1 embedding (raw vs per-window normalized): {delta:.4f}")
    print(
        "`normalize_input=True` z-normalizes EACH window independently — which\n"
        "erases cross-window amplitude (a low-flow and a high-flow window can\n"
        "look identical). For comparing windows (classification, anomaly), fit\n"
        "ONE scaler on your training pool, apply it to every window, and call\n"
        "with normalize_input=False — see classify_knn.py. Reserve\n"
        "normalize_input=True for one-off single-window encodes where only the\n"
        "within-window shape matters.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV),
                        help=f"Sensor CSV (timestamp auto-skipped). Default: {DEFAULT_CSV.name}.")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        sys.exit(f"CSV not found: {args.csv}")
    series = read_series(args.csv)

    example_basic(series)
    example_padding(series)
    example_normalize(series)


if __name__ == "__main__":
    main()
