"""Shared helpers for the AD data-prep scripts: the index, recordings, timestamps.

The index (`recordings.csv`) lists one recording per row:

    recording   a unique id (used in file names)
    file        path to the recording, relative to the index (CSV or Parquet)
    role        library | validation | test | delivery
    group       optional: recordings that must stay in one role (one asset-day, one run, ...)
    type        optional: what kind of incident the recording holds (a fault type, a failure
                mode); the delivery is scored per type when it is given

A recording file has a timestamp column, the sensor channels, and a per-row `label`:
`normal`, or `anomaly` once the incident has started. Library recordings must be all
`normal`: the detector is fitted on them and on nothing else.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROLES = ("library", "validation", "test", "delivery")
SCORED = ("validation", "test", "delivery")
LABELS = ("normal", "anomaly")


def read_index(path):
    idx = pd.read_csv(path, dtype=str).fillna("")
    for col in ("recording", "file", "role"):
        if col not in idx:
            raise SystemExit(f"{path}: needs a '{col}' column")
    if idx.recording.duplicated().any():
        raise SystemExit(f"{path}: duplicate recording ids: {sorted(idx.recording[idx.recording.duplicated()])}")
    bad = sorted(set(idx.role) - set(ROLES))
    if bad:
        raise SystemExit(f"{path}: unknown or empty role(s) {bad}; use {list(ROLES)}")
    for col in ("group", "type"):
        if col not in idx:
            idx[col] = ""
    idx["path"] = [os.path.join(os.path.dirname(os.path.abspath(path)), f) for f in idx.file]
    return idx


def read_recording(path):
    return pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)


def to_seconds(ts):
    """Epoch seconds (float) from epoch seconds, epoch milliseconds or ISO 8601 / datetimes."""
    s = pd.Series(ts)
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(dtype=float)
        return v / 1000.0 if np.nanmedian(np.abs(v)) > 1e11 else v
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True, format="ISO8601")
    elif s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return (s - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def timestamp_strings(ms):
    """The contract's timestamp format: fractional epoch seconds, exactly 3 decimals."""
    ms = np.asarray(ms, dtype=np.int64)
    return (pd.Series(ms // 1000).astype(str) + "." + pd.Series(ms % 1000).astype(str).str.zfill(3)).to_numpy()


def stem(recording):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in recording)


def load_prepared(prepared_dir, recording):
    return pd.read_parquet(os.path.join(prepared_dir, stem(recording) + ".parquet"))


def channels_of(df):
    return [c for c in df.columns if c not in ("timestamp", "label", "segment")]
