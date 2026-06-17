"""
Shared helpers for the Omega embedding examples (cloud Omega via /query),
built on the official Archetype AI python client
(https://github.com/archetypeai/python-client).

Omega is a time-series ENCODER: you send a window of sensor readings and get
back a fixed-size embedding vector per channel — no text, no prompt. This is
the cloud counterpart to running the encoder locally; here the model runs on
the Archetype AI platform and you call it over the same /query endpoint the
Newton fusion model uses.

Each script creates one client via `make_client()` and passes it as the
first argument to `embed()`.

Credential lookup (first hit wins): ATAI_API_KEY / ATAI_API_ENDPOINT env vars
(BOTH required — no default endpoint), then a .env found by walking up from
cwd, then a .env next to this file.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from archetypeai.api_client import ArchetypeAI

MODEL = "OmegaEncoder::omega_embeddings_1_4"
MIN_WINDOW = 16  # shortest signal length the encoder is trained on
WINDOW = 1024    # longest; inputs > 1024 are truncated to the LAST 1024 points (with a warning)
EMBED_DIM = 768  # per-channel embedding dimension returned by omega_embeddings_1_4

# Default header names treated as time columns (overridable per call —
# pass your own `time_columns` to read_series / read_series_and_time).
TIME_COLUMNS = {"timestamp", "time", "ts", "datetime", "date"}


def _try_load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except ImportError:
        return
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
    sibling = Path(__file__).parent / ".env"
    if sibling.exists():
        load_dotenv(sibling, override=False)


def make_client() -> ArchetypeAI:
    """Resolve credentials and return an official ArchetypeAI client.

    Both ATAI_API_KEY and ATAI_API_ENDPOINT are required. The endpoint is
    deliberately NOT defaulted — pointing at the wrong deployment should
    fail loudly here, not silently at query time.
    """
    _try_load_dotenv()
    api_key = os.environ.get("ATAI_API_KEY")
    if not api_key:
        sys.exit(
            "ATAI_API_KEY is not set. Either export it, or copy .env.example "
            "to .env (in the cwd or alongside this file) and fill it in."
        )
    api_endpoint = os.environ.get("ATAI_API_ENDPOINT")
    if not api_endpoint:
        sys.exit(
            "ATAI_API_ENDPOINT is not set. Export it (e.g. "
            "https://api.u1.archetypeai.app/v0.5) or add it to your .env — "
            "there is no default."
        )
    api_endpoint = api_endpoint.rstrip("/")
    if not api_endpoint.endswith("/v0.5"):
        api_endpoint = api_endpoint + "/v0.5"
    return ArchetypeAI(api_key, api_endpoint=api_endpoint)


def _embed_channel(
    client: ArchetypeAI, channel_values: list[float], normalize_input: bool
) -> tuple[list[float], list[str]]:
    """Embed ONE channel via /query. Returns (vector, warnings)."""
    body = {
        "query": "",
        "model": MODEL,
        "normalize_input": normalize_input,
        "events": [{"type": "data.numeric_array", "event_data": {"contents": [channel_values]}}],
    }
    payload = client.requests_post(
        f"{client.api_endpoint}/query",
        data_payload=json.dumps(body),
        additional_headers={"Content-Type": "application/json"},
    )
    embeddings, warnings = extract_embeddings(payload)
    # A single-channel response carries the vector flat; normalize the shape.
    if embeddings and isinstance(embeddings[0], (int, float)):
        embeddings = [embeddings]
    return (embeddings[0] if embeddings else []), warnings


def embed(
    client: ArchetypeAI,
    window: list[list[float]],
    *,
    normalize_input: bool = False,
    max_workers: int = 8,
) -> tuple[list[list[float]], list[str], int]:
    """Embed one **channel-first** window with Omega 1.4 via /query.

    `window` is `[channels x timesteps]` — a list of `channels` lists, each a
    per-channel time series. Returns (embeddings, warnings, elapsed_ms), where
    `embeddings` is one ``EMBED_DIM``-length vector per input channel.

    Channels are embedded with ONE API call per channel, fanned out in
    parallel (`max_workers`). This is the recommended pattern: per-channel
    requests cap the request/response payload size (a single request carrying
    many channels eventually exceeds REST payload limits), and the parallel
    fan-out keeps wall-clock close to a single call. Don't mix conventions
    within one downstream model: per-channel and all-channels-in-one-request
    embeddings differ slightly (~2e-2 per coordinate, measured), so a KNN
    library and its inference windows must be built the same way.

    Each channel goes in a `data.numeric_array` event (no `file_ids`, no
    prompt). The encoder handles window lengths of MIN_WINDOW (16) to WINDOW
    (1024) natively — sub-1024 windows are padded AND masked internally, so
    the accompanying "padding with zeros" warning is informational. Lengths
    below 16 are outside the trained range; lengths above 1024 are truncated
    to the last 1024 points. Raises `archetypeai.ApiError` on a 4xx response.
    """
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(window))) as executor:
        per_channel = list(
            executor.map(
                lambda channel_values: _embed_channel(client, channel_values, normalize_input),
                window,
            )
        )
    elapsed_ms = int((time.time() - start_time) * 1000)
    embeddings = [vector for vector, _ in per_channel]
    warnings = [message for _, channel_warnings in per_channel for message in channel_warnings]
    return embeddings, warnings, elapsed_ms


def extract_embeddings(payload: dict[str, Any]) -> tuple[list[list[float]], list[str]]:
    """Pull (embeddings, warning_messages) from a /query response."""
    response_field = payload.get("response")
    if isinstance(response_field, dict):
        return response_field.get("response") or [], response_field.get("warning_messages") or []
    if isinstance(response_field, list):
        return response_field, []
    return [], []


def _is_float(cell: str) -> bool:
    try:
        float(cell)
        return True
    except ValueError:
        return False


def read_series_and_time(
    csv_path: str | Path,
    channels: int | None = None,
    time_columns: set[str] = TIME_COLUMNS,
) -> tuple[list[int], list[list[float]]]:
    """Read a sensor CSV into `(timestamps, [channels x n_rows])`.

    Channels are the numeric columns, dropping time columns by header name
    (`time_columns` — defaults to TIME_COLUMNS, pass your own set to
    customize) even though they parse as numbers, and dropping
    non-numeric columns like `label`. The first time column (or column 0) is
    returned as the integer timestamp list — used for contiguity checks and
    joining ground-truth labels. `channels` caps the count (default: all).
    """
    timestamps: list[int] = []
    rows: list[list[float]] = []
    numeric_columns: list[int] | None = None
    with open(csv_path, newline="") as file_handle:
        reader = csv.reader(file_handle)
        header = next(reader, []) or []
        time_column = next(
            (
                column_index
                for column_index, column_name in enumerate(header)
                if column_name.strip().lower() in time_columns
            ),
            0,
        )
        for row in reader:
            if not row:
                continue
            if numeric_columns is None:
                numeric_columns = [
                    column_index
                    for column_index, cell in enumerate(row)
                    if _is_float(cell)
                    and (header[column_index].strip().lower() if column_index < len(header) else "")
                    not in time_columns
                ]
                if channels:
                    numeric_columns = numeric_columns[:channels]
            timestamps.append(int(float(row[time_column])))
            rows.append([float(row[column_index]) for column_index in numeric_columns])
    if not rows:
        sys.exit(f"No numeric rows found in {csv_path}")
    row_count, channel_count = len(rows), len(rows[0])
    return timestamps, [
        [rows[row_index][channel_index] for row_index in range(row_count)]
        for channel_index in range(channel_count)
    ]


def read_series(
    csv_path: str | Path,
    channels: int | None = None,
    time_columns: set[str] = TIME_COLUMNS,
) -> list[list[float]]:
    """Read a sensor CSV into a **channel-first** matrix `[channels x n_rows]`.

    Convenience wrapper over ``read_series_and_time`` that drops the timestamps.
    """
    return read_series_and_time(csv_path, channels, time_columns)[1]


def window_at(series: list[list[float]], start: int, window_size: int = WINDOW) -> list[list[float]]:
    """Slice a channel-first `[channels x window_size]` window from a series."""
    series_length = len(series[0])
    if start + window_size > series_length:
        sys.exit(f"window [{start}:{start + window_size}] out of range (series has {series_length} rows)")
    return [channel[start : start + window_size] for channel in series]


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
