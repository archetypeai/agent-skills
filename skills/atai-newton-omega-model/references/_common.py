"""
Shared helpers for the Omega embedding examples (Newton Omega 1.5 on Model
Serving), built on the official Archetype AI python client
(https://github.com/archetypeai/python-client).

Omega is a time-series ENCODER: you send a window of sensor readings and get
back a fixed-size embedding vector per channel — no text, no prompt.

Two ways to call it:
  * `OmegaSession` — open a Model Serving session with your `window_size`
    (16-1024) and send windows to it. Use this for anything but 1024.
  * `embed_once()` — one stateless `POST /api/inference/query`, fixed
    1024-timestep window.

Each script creates one client via `make_client()`.

Credential lookup (first hit wins): ATAI_API_KEY / ATAI_API_ENDPOINT env vars
(BOTH required — no default endpoint), then a .env found by walking up from
cwd, then a .env next to this file.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from archetypeai.api_client import ApiError, ArchetypeAI

MODEL = "omega-embedding"  # Newton Omega 1.5 on Model Serving
MIN_WINDOW = 16  # shortest window the encoder is trained on
WINDOW = 1024    # longest, and the window used without a session
EMBED_DIM = 768  # embedding dimension per channel
# Request bodies are capped at 2 MiB (HTTP 413 above it): roughly 95 channels
# x 1024 timesteps at full float precision. Wider windows are split into
# requests of at most this many channels; Omega 1.5 encodes each channel
# independently, so the joined result is the same vector.
MAX_CHANNELS_PER_REQUEST = 64
SESSION_ENDED = {"session_not_running", "not_found"}

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
            "https://platform.u1.archetypeai.app/api) or add it to your .env — "
            "there is no default."
        )
    api_endpoint = re.sub(r"/api(/lens)?$", "", api_endpoint.rstrip("/"))
    api_endpoint = re.sub(r"/v[0-9]+(\.[0-9]+)*$", "", api_endpoint) + "/api"
    return ArchetypeAI(api_key, api_endpoint=api_endpoint)


def _post(client: ArchetypeAI, path: str, body: dict) -> dict:
    return client.requests_post(
        f"{client.api_endpoint}{path}",
        data_payload=json.dumps(body),
        additional_headers={"Content-Type": "application/json"},
    )


def _query_body(channels: list[list[float]]) -> dict:
    return {
        "model": MODEL,
        "query": "",
        "events": [{"type": "data.numeric_array", "event_data": {"contents": channels}}],
    }


def _split_rows(rows: list[list[float]], channel_count: int) -> list[list[list[float]]]:
    """[windows x (channels*768)] -> [windows][channel] -> 768-d vector."""
    return [
        [row[channel * EMBED_DIM:(channel + 1) * EMBED_DIM] for channel in range(channel_count)]
        for row in rows
    ]


def _rows_from(payload: dict, channel_count: int) -> list[list[float]]:
    response = payload.get("response") or {}
    # Input shorter than the window comes back as success=false with an
    # empty response and no message.
    if not response.get("success"):
        raise RuntimeError(
            "Omega returned success=false (is the input shorter than the window?)"
        )
    rows = response.get("response") or []
    if not rows or any(len(row) != channel_count * EMBED_DIM for row in rows):
        raise RuntimeError(f"unexpected embedding shape for {channel_count} channels")
    return rows


def _chunks(window: list[list[float]]):
    for first in range(0, len(window), MAX_CHANNELS_PER_REQUEST):
        yield window[first:first + MAX_CHANNELS_PER_REQUEST]


def embed_once(client: ArchetypeAI, window: list[list[float]]) -> list[list[list[float]]]:
    """Embed without a session: `POST /api/inference/query`, window fixed at 1024.

    `window` is channel-first `[channels x timesteps]`. Input is cut into
    back-to-back 1024-timestep windows (leftover timesteps are dropped), so
    the result is `[windows][channel] -> 768-d vector`. Fewer than 1024
    timesteps raises RuntimeError. For any other window length use
    `OmegaSession`.
    """
    per_chunk = [
        _split_rows(_rows_from(_post(client, "/inference/query", _query_body(chunk)), len(chunk)),
                    len(chunk))
        for chunk in _chunks(window)
    ]
    return [sum((chunk[index] for chunk in per_chunk), []) for index in range(len(per_chunk[0]))]


class OmegaSession:
    """A Model Serving session for Omega with a fixed window length.

    Every request sent to the session is cut into `window_size`-timestep
    windows, advancing `step_size` (default: `window_size`, so windows don't
    overlap); each full window returns one 768-d vector per channel.
    Use it as a context manager so the session is cancelled when you're done
    (each org has a limit on open sessions per model):

        with OmegaSession(client, window_size=256) as session:
            vectors = session.embed(window)   # [channels] -> 768-d

    A session ends when cancelled, when idle long enough, or when its server
    is replaced; queries then fail with HTTP 409 `session_not_running`.
    `embed()` reopens the session and retries once in that case.
    """

    def __init__(self, client: ArchetypeAI, window_size: int = WINDOW, step_size: int | None = None):
        if not MIN_WINDOW <= window_size <= WINDOW:
            raise ValueError(f"window_size must be {MIN_WINDOW}-{WINDOW}, got {window_size}")
        self.client = client
        self.values = {"window_size": window_size, "step_size": step_size or window_size}
        self.session_id: str | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "OmegaSession":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def open(self, replace: str | None = None) -> str:
        """Open the session (or reopen it if `replace` is the current id)."""
        with self._lock:
            if self.session_id is None or self.session_id == replace:
                session = _post(self.client, "/inference/sessions",
                                {"model": MODEL, "values": self.values})
                self.session_id = session["id"]
            return self.session_id

    def close(self) -> None:
        """Cancel the session, freeing its place under the org's limit."""
        with self._lock:
            session_id, self.session_id = self.session_id, None
        if session_id:
            try:
                self.client.requests_post(
                    f"{self.client.api_endpoint}/inference/sessions/{session_id}/cancel",
                    data_payload=b"",
                )
            except Exception:  # noqa: BLE001 — an idle session expires anyway
                pass

    def _query(self, chunk: list[list[float]]) -> list[list[float]]:
        session_id = self.open()
        try:
            payload = _post(self.client, f"/inference/sessions/{session_id}/query", _query_body(chunk))
        except ApiError as error:
            codes = {item.get("code") for item in (error.errors or [])}
            if not codes & SESSION_ENDED:
                raise
            session_id = self.open(replace=session_id)
            payload = _post(self.client, f"/inference/sessions/{session_id}/query", _query_body(chunk))
        return _rows_from(payload, len(chunk))

    def embed_all(self, window: list[list[float]]) -> list[list[list[float]]]:
        """Embed a channel-first series: `[windows][channel] -> 768-d vector`."""
        per_chunk = [_split_rows(self._query(chunk), len(chunk)) for chunk in _chunks(window)]
        return [sum((chunk[index] for chunk in per_chunk), []) for index in range(len(per_chunk[0]))]

    def embed(self, window: list[list[float]]) -> list[list[float]]:
        """Embed ONE `[channels x window_size]` window: one 768-d vector per channel."""
        windows = self.embed_all(window)
        if len(windows) != 1:
            raise ValueError(f"expected one window, got {len(windows)}; slice the input to window_size")
        return windows[0]


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
