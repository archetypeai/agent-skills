"""Shared by the Label-Free AD data-prep scripts: reading raw CSVs, the grid, frozen stretches, reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_raw(paths: list[str], timestamp_col: str) -> pd.DataFrame:
    """One or more raw CSVs (plain or .gz), concatenated and sorted by time.

    The timestamp may be text (ISO 8601 or similar) or numbers (epoch seconds or
    milliseconds, told apart by size). Unnamed index columns are dropped.
    """
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=[c for c in df.columns if c == "" or str(c).startswith("Unnamed")])
    if timestamp_col not in df.columns:
        raise SystemExit(f"no column {timestamp_col!r}; columns are {list(df.columns)[:12]} (use --timestamp-col)")
    t = df[timestamp_col]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.dropna().abs().median() > 1e11 else "s"
        df[timestamp_col] = pd.to_datetime(t, unit=unit)
    else:
        df[timestamp_col] = pd.to_datetime(t, errors="coerce")
    return df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)


def pick_channels(df: pd.DataFrame, timestamp_col: str, channels: str | None) -> tuple[list[str], list[str]]:
    """The channels to keep, and the numeric ones left out.

    Explicit (comma-separated) wins. Otherwise every numeric column except binary
    ones (0/1 switch or valve states), which say little within a window and are
    constant in most.
    """
    numeric = [c for c in df.columns if c != timestamp_col and pd.api.types.is_numeric_dtype(df[c])]
    if channels:
        keep = [c.strip() for c in channels.split(",") if c.strip()]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise SystemExit(f"channels not in the data: {missing}")
        return keep, [c for c in numeric if c not in keep]
    binary = [c for c in numeric if set(np.unique(df[c].dropna())) <= {0, 1}]
    return [c for c in numeric if c not in binary], binary


def grid(df: pd.DataFrame, timestamp_col: str, channels: list[str], step_s: float, max_fill: int) -> pd.DataFrame:
    """Snap rows to an exact step (the mean of each bin). Holes of up to `max_fill` bins
    take the last value; longer holes are gaps and stay gaps. Nothing is interpolated
    across a gap."""
    binned = df.set_index(timestamp_col)[channels].resample(pd.Timedelta(seconds=step_s)).mean()
    present = binned.notna().all(axis=1)
    run_len = present.groupby((present != present.shift()).cumsum()).transform("size")
    gap = (~present) & (run_len > max_fill)
    return binned.ffill(limit=max_fill)[~gap].dropna()


def frozen_mask(g: pd.DataFrame, step_s: float, min_rows: int) -> np.ndarray:
    """Rows in a run of >= `min_rows` consecutive rows where no channel changes (never across a gap)."""
    t = g.index.to_series()
    piece = (t.diff() != pd.Timedelta(seconds=step_s)).cumsum()
    same = (g.diff().abs().sum(axis=1) == 0) & (piece == piece.shift())
    run = (~same).cumsum()
    return (same.groupby(run).transform("size") >= min_rows).to_numpy()


def epoch_s(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype="int64")


def window_counts(ts: np.ndarray, step_s: float, window: int) -> tuple[int, int]:
    """Windows of `window` rows tiled from row 0 without overlap, as od cuts them; and
    how many of them hold no gap (od marks the others invalid)."""
    dt = np.diff(ts)
    n = len(ts) // window
    valid = sum(1 for i in range(n) if (dt[i * window:(i + 1) * window - 1] == step_s).all())
    return n, valid


class Report:
    """PASS / WARN / FAIL / INFO lines; the exit code is non-zero only on FAIL."""

    def __init__(self, title: str):
        self.rows: list[tuple[str, str, str]] = []
        print(f"=== {title} ===")

    def add(self, level: str, name: str, msg: str) -> None:
        self.rows.append((level, name, msg))
        print(f"  {level:<4}  {name:<26} {msg}")

    def done(self) -> int:
        n = {k: sum(r[0] == k for r in self.rows) for k in ("PASS", "WARN", "FAIL", "INFO")}
        print(f"\nRESULT: {'FAIL' if n['FAIL'] else 'PASS'}  "
              f"({n['PASS']} PASS, {n['WARN']} WARN, {n['FAIL']} FAIL, {n['INFO']} INFO)")
        return 1 if n["FAIL"] else 0


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)
