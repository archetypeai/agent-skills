"""
Omega embedding examples — turn a window of sensor readings into vectors with
Newton Omega 1.5 on Model Serving (model `omega-embedding`).

Four patterns:
  1. Embed one window without a session (`POST /api/inference/query`, 1024).
  2. Window lengths — open a session with `window_size` 16 to 1024.
  3. What a session does with input that isn't exactly one window.
  4. Channels are encoded independently — splitting them gives the same vectors.

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
import time
from pathlib import Path

from archetypeai.api_client import ArchetypeAI

from _common import (
    EMBED_DIM,
    MIN_WINDOW,
    WINDOW,
    OmegaSession,
    banner,
    embed_once,
    make_client,
    read_series,
    window_at,
)

DEFAULT_CSV = Path(__file__).parent / "sample_data" / "bearing_healthy.csv"


def _max_difference(first: list[float], second: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second))


def example_basic(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner(f"1. Embed one window without a session — {len(series)} channels x {WINDOW} timesteps")
    window_values = window_at(series, start=0, window_size=WINDOW)
    start_time = time.time()
    windows = embed_once(client, window_values)
    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"[{elapsed_ms} ms]")
    print(f"input  : {len(window_values)} channels x {len(window_values[0])} timesteps")
    print(f"output : {len(windows)} window x {len(windows[0])} channels x {len(windows[0][0])} dims "
          f"(one {EMBED_DIM}-d vector per channel)")
    print(f"channel 1 embedding[:5]: {[round(value, 4) for value in windows[0][0][:5]]}")
    print("Without a session the window is always 1024 timesteps.\n")


def example_window_lengths(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner(f"2. Window lengths — open a session with window_size {MIN_WINDOW} to {WINDOW}")
    for window_length in (MIN_WINDOW, 256, WINDOW):
        with OmegaSession(client, window_size=window_length) as session:
            start_time = time.time()
            vectors = session.embed(window_at(series, start=0, window_size=window_length))
            elapsed_ms = int((time.time() - start_time) * 1000)
        leading_coords = [round(value, 4) for value in vectors[0][:5]]
        print(f"  window_size={window_length:>4} [{elapsed_ms} ms] -> {len(vectors)} x {len(vectors[0])}"
              f"  channel 1 embedding[:5]: {leading_coords}")
    print(
        f"Takeaway: windows shorter than {WINDOW} are padded and masked inside the\n"
        "model, so padding is ignored rather than read as signal. Pick the length\n"
        "that fits your signal's dynamics and compare a few on held-out data —\n"
        "see classify_knn.py --window-size.\n"
    )


def example_session_input(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner("3. Input that isn't exactly one window")
    with OmegaSession(client, window_size=256) as session:
        windows = session.embed_all(window_at(series, start=0, window_size=600))
        print(f"  600 timesteps, window_size=256 -> {len(windows)} windows "
              "(back-to-back; the last 88 timesteps are dropped)")
        try:
            session.embed_all(window_at(series, start=0, window_size=200))
        except RuntimeError as error:
            print(f"  200 timesteps, window_size=256 -> {error}")
    print(
        "Takeaway: a session cuts each request into window_size-timestep windows\n"
        "(advancing step_size), returns one row per full window, and drops the\n"
        "rest. Shorter input returns success=false with no message — slice your\n"
        "data to exactly window_size per call.\n"
    )


def example_channels_independent(client: ArchetypeAI, series: list[list[float]]) -> None:
    banner("4. Channels are encoded independently")
    window_values = window_at(series, start=0, window_size=256)
    with OmegaSession(client, window_size=256) as session:
        together = session.embed(window_values)
        apart = [session.embed([channel])[0] for channel in window_values]
    largest = max(_max_difference(a, b) for a, b in zip(together, apart))
    print(f"  all {len(window_values)} channels in one request vs one request per channel: "
          f"largest difference {largest:.4f}")
    print(
        "Takeaway: send all channels in one request (fewer calls). For very wide\n"
        "windows, splitting channels across requests gives the same vectors —\n"
        "_common splits automatically to stay under the 2 MiB request limit.\n"
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
    example_session_input(client, series)
    example_channels_independent(client, series)


if __name__ == "__main__":
    main()
