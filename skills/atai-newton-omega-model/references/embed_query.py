"""
Omega embedding examples — turn a window of sensor readings into vectors,
via the cloud Omega encoder (`OmegaEncoder::omega_embeddings_1_4`) over /query.

Three patterns:
  1. Embed one multi-channel window (the basic call + output shape).
  2. Window lengths — the encoder handles 16 to 1024 timesteps natively
     (padding + mask internally); pick the length that fits your signal.
  3. `normalize_input` — z-norm the input window before encoding.

Data: a subset of the NASA IMS Bearing dataset (4 accelerometer channels). See
sample_data/README.md for attribution.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python embed_query.py
    # or point at your own sensor CSV (timestamp column auto-skipped):
    python embed_query.py /path/to/series.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import EMBED_DIM, MIN_WINDOW, WINDOW, banner, embed, make_client, read_series, window_at

DEFAULT_CSV = Path(__file__).parent / "sample_data" / "bearing_healthy.csv"


def example_basic(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner(f"1. Embed one window — {len(series)} channels x {WINDOW} timesteps")
    window_values = window_at(series, start=0, window=WINDOW)
    embeddings, warnings, elapsed_ms = embed(client, window_values)
    print(f"[{elapsed_ms} ms]")
    print(f"input  : {len(window_values)} channels x {len(window_values[0])} timesteps")
    print(f"output : {len(embeddings)} embeddings x {len(embeddings[0])} dims "
          f"(one {EMBED_DIM}-d vector per channel)")
    print(f"channel 1 embedding[:5]: {[round(value, 4) for value in embeddings[0][:5]]}")
    if warnings:
        print(f"warnings: {warnings}")
    print()


def example_window_lengths(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner(f"2. Window lengths — the encoder is trained for {MIN_WINDOW} to {WINDOW} timesteps")
    for window_length in (MIN_WINDOW, 256, WINDOW):
        window_values = window_at(series, start=0, window=window_length)
        embeddings, warnings, elapsed_ms = embed(client, window_values)
        warning_note = warnings[0] if warnings else "(no warnings)"
        leading_coords = [round(value, 4) for value in embeddings[0][:5]]
        print(f"  len={window_length:>4} [{elapsed_ms} ms] -> {len(embeddings)} x {len(embeddings[0])}  {warning_note}")
        print(f"           channel 1 embedding[:5]: {leading_coords}")
    print(
        f"Takeaway: any length in [{MIN_WINDOW}, {WINDOW}] is handled natively — sub-{WINDOW}\n"
        "windows are padded AND masked internally, so the 'padding with zeros'\n"
        "warning is informational. Pick the window length that fits your signal's\n"
        "dynamics; shorter windows are sometimes more appropriate and more\n"
        f"performant. Below {MIN_WINDOW} is outside the trained range, and inputs longer\n"
        f"than {WINDOW} are truncated to the LAST {WINDOW} points.\n"
    )


def example_normalize(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner("3. normalize_input — PER-WINDOW z-norm (usually NOT what you want)")
    window_values = window_at(series, start=0, window=WINDOW)
    raw, _, _ = embed(client, window_values, normalize_input=False)
    normalized, _, _ = embed(client, window_values, normalize_input=True)
    delta = sum(
        abs(raw_value - normalized_value)
        for raw_value, normalized_value in zip(raw[0], normalized[0])
    ) / len(raw[0])
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

    client = make_client()
    example_basic(client, series)
    example_window_lengths(client, series)
    example_normalize(client, series)


if __name__ == "__main__":
    main()
