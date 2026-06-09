"""
Shared helpers for the Omega embedding examples (cloud Omega via /query).

Omega is a time-series ENCODER: you send a window of sensor readings and get
back a fixed-size embedding vector per channel — no text, no prompt. This is
the cloud counterpart to running the encoder locally; here the model runs on
the Archetype AI platform and you call it over the same /query endpoint the
Newton fusion model uses.

Credential lookup (first hit wins): ATAI_API_KEY env var, then a .env found by
walking up from cwd, then a .env next to this file.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

MODEL = "OmegaEncoder::omega_embeddings_1_4"
DEFAULT_ENDPOINT = "https://api.u1.archetypeai.app/v0.5"
WINDOW = 1024  # the encoder's native window length; shorter inputs are zero-padded (with a warning)
EMBED_DIM = 768  # per-channel embedding dimension returned by omega_embeddings_1_4


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


def client() -> tuple[str, dict[str, str]]:
    """Resolve credentials + endpoint. Returns (endpoint, headers)."""
    _try_load_dotenv()
    api_key = os.environ.get("ATAI_API_KEY")
    if not api_key:
        sys.exit(
            "ATAI_API_KEY is not set. Either export it, or copy .env.example "
            "to .env (in the cwd or alongside this file) and fill it in."
        )
    endpoint = os.environ.get("ATAI_API_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    if not endpoint.endswith("/v0.5"):
        endpoint = endpoint + "/v0.5"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return endpoint, headers


def embed(
    window: list[list[float]],
    *,
    normalize_input: bool = False,
    timeout: int = 60,
) -> tuple[list[list[float]], list[str], int]:
    """Embed one **channel-first** window with Omega 1.4 via /query.

    `window` is `[channels x timesteps]` — a list of `channels` lists, each a
    per-channel time series. Returns (embeddings, warnings, elapsed_ms), where
    `embeddings` is one ``EMBED_DIM``-length vector per input channel.

    The window goes in a `data.numeric_array` event (no `file_ids`, no prompt).
    Windows shorter than WINDOW are zero-padded server-side and a warning is
    returned in `warnings`.
    """
    endpoint, headers = client()
    body = {
        "query": "",
        "model": MODEL,
        "normalize_input": normalize_input,
        "events": [{"type": "data.numeric_array", "event_data": {"contents": window}}],
    }
    t0 = time.time()
    r = requests.post(f"{endpoint}/query", headers=headers, json=body, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)
    if not r.ok:
        sys.exit(f"Embed failed [{r.status_code}]: {r.text[:400]}")
    return (*extract_embeddings(r.json()), elapsed_ms)


def extract_embeddings(payload: dict[str, Any]) -> tuple[list[list[float]], list[str]]:
    """Pull (embeddings, warning_messages) from a /query response."""
    resp = payload.get("response")
    if isinstance(resp, dict):
        return resp.get("response") or [], resp.get("warning_messages") or []
    if isinstance(resp, list):
        return resp, []
    return [], []


def _is_float(cell: str) -> bool:
    try:
        float(cell)
        return True
    except ValueError:
        return False


TIME_COLUMNS = {"timestamp", "time", "ts", "datetime", "date"}


def read_series_and_time(
    csv_path: str | Path, channels: int | None = None
) -> tuple[list[int], list[list[float]]]:
    """Read a sensor CSV into `(timestamps, [channels x n_rows])`.

    Channels are the numeric columns, dropping time columns by header name
    (`timestamp`, `time`, ...) even though they parse as numbers, and dropping
    non-numeric columns like `label`. The first time column (or column 0) is
    returned as the integer timestamp list — used for contiguity checks and
    joining ground-truth labels. `channels` caps the count (default: all).
    """
    ts: list[int] = []
    rows: list[list[float]] = []
    numeric_idx: list[int] | None = None
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, []) or []
        ts_idx = next(
            (i for i, h in enumerate(header) if h.strip().lower() in TIME_COLUMNS), 0
        )
        for row in reader:
            if not row:
                continue
            if numeric_idx is None:
                numeric_idx = [
                    i
                    for i, c in enumerate(row)
                    if _is_float(c) and (header[i].strip().lower() if i < len(header) else "") not in TIME_COLUMNS
                ]
                if channels:
                    numeric_idx = numeric_idx[:channels]
            ts.append(int(float(row[ts_idx])))
            rows.append([float(row[i]) for i in numeric_idx])
    if not rows:
        sys.exit(f"No numeric rows found in {csv_path}")
    n, ch = len(rows), len(rows[0])
    return ts, [[rows[t][c] for t in range(n)] for c in range(ch)]


def read_series(csv_path: str | Path, channels: int | None = None) -> list[list[float]]:
    """Read a sensor CSV into a **channel-first** matrix `[channels x n_rows]`.

    Convenience wrapper over ``read_series_and_time`` that drops the timestamps.
    """
    return read_series_and_time(csv_path, channels)[1]


def window_at(series: list[list[float]], start: int, window: int = WINDOW) -> list[list[float]]:
    """Slice a channel-first `[channels x window]` window from a series."""
    n = len(series[0])
    if start + window > n:
        sys.exit(f"window [{start}:{start + window}] out of range (series has {n} rows)")
    return [channel[start : start + window] for channel in series]


def banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
