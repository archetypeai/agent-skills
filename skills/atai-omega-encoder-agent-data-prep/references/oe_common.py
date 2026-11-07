"""Shared by the Omega Encoder agent data-prep scripts: reading tables, the report, the rules."""
from __future__ import annotations

import os

import pandas as pd

ROLES = ("library", "validation", "test", "delivery")
SCORED = ("validation", "test", "delivery")
NORMAL = "normal"


def read_table(path: str) -> pd.DataFrame:
    """A recording, from CSV or Parquet."""
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def divides(rows: int, windows: list[int]) -> list[int]:
    """The candidate windows that do NOT divide `rows` (empty when every one does)."""
    return [w for w in windows if w <= 0 or rows % w]


class Report:
    """PASS / WARN / FAIL / INFO lines; exit status non-zero only on FAIL."""

    def __init__(self, title: str):
        self.levels: list[str] = []
        print(f"== {title}")

    def add(self, level: str, check: str, detail: str) -> None:
        self.levels.append(level)
        print(f"  {level:<4}  {check:<32} {detail}")

    def finish(self) -> int:
        counts = {k: self.levels.count(k) for k in ("PASS", "WARN", "FAIL", "INFO")}
        print("  " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        return 1 if counts["FAIL"] else 0


def here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)
