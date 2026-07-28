#!/usr/bin/env python3
"""Generate a synthetic RED sample set — stdlib only, deterministic.

Writes two pure-class shot files (for the `red-fitting` blueprint, which takes
each file's class from its FILENAME), one inference slice, and a row-aligned
label sidecar for scoring.

The signal is intentionally simple: five z-scored channels of correlated noise,
with the excursion class shifting the mean of three and damping a fourth. It
exercises the API contract and the scoring views, not detector quality — see
this directory's README for the real dataset and its attribution.

Usage:  python3 make_sample_data.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANNELS = [f"sensor_{i:02d}" for i in range(5)]
CADENCE_S = 60           # 1-minute samples, like the pump dataset
START_TS = 1_530_000_000
NORMAL_SHOT_ROWS = 4_096
FAULT_SHOT_ROWS = 512
LEAD_ROWS = 2_048        # normal context either side of the inference excursion
EXCURSION_ROWS = 512
NORMAL, FAULT = "normal", "excursion"


def rows(n: int, fault: bool, rng: random.Random, t0: int):
    """n rows of z-scored channels; `fault` shifts three and damps one."""
    walk = [0.0] * len(CHANNELS)
    for i in range(n):
        out = {"timestamp": t0 + i * CADENCE_S}
        for c, name in enumerate(CHANNELS):
            # slow correlated drift plus noise, so windows are not iid
            walk[c] = 0.97 * walk[c] + rng.gauss(0, 0.22)
            v = walk[c] + 0.35 * math.sin((i + c * 40) / 90)
            if fault:
                if c < 3:
                    v += 2.4                       # mean shift
                elif c == 3:
                    v *= 0.15                      # damped, as if flow stopped
            out[name] = round(v, 6)
        yield out


def write(path: Path, records, labels: list[str] | None = None) -> None:
    records = list(records)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp"] + CHANNELS)
        w.writeheader()
        w.writerows(records)
    print(f"  {path.name}: {len(records):,} rows")
    if labels is not None:
        side = path.with_name(path.stem + "_labels.csv")
        with open(side, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "label"])
            w.writeheader()
            w.writerows({"timestamp": r["timestamp"], "label": l}
                        for r, l in zip(records, labels))
        print(f"  {side.name}: {len(labels):,} rows")


def main() -> None:
    rng = random.Random(20260728)

    # Shot files must be single-class: the fitting runner labels a whole file by
    # its filename, so a normal row inside a fault file would be mislabelled.
    write(HERE / "red_sample_shots_normal.csv",
          rows(NORMAL_SHOT_ROWS, False, rng, START_TS))
    write(HERE / f"red_sample_shots_{FAULT}.csv",
          rows(FAULT_SHOT_ROWS, True, rng, START_TS + NORMAL_SHOT_ROWS * CADENCE_S))

    # Inference slice: normal -> excursion -> normal, so the false-alarm rate is
    # measurable and detection latency is meaningful.
    t = START_TS + 10_000 * CADENCE_S
    slice_rows, labels = [], []
    for n, fault in ((LEAD_ROWS, False), (EXCURSION_ROWS, True), (LEAD_ROWS, False)):
        chunk = list(rows(n, fault, rng, t))
        slice_rows += chunk
        labels += [FAULT if fault else NORMAL] * n
        t += n * CADENCE_S
    # renumber timestamps so the cadence is strictly regular across the joins —
    # the blueprint validates monotonicity and sample-rate consistency
    for i, r in enumerate(slice_rows):
        r["timestamp"] = START_TS + 10_000 * CADENCE_S + i * CADENCE_S
    write(HERE / "red_sample_inference.csv", slice_rows, labels)

    print(f"\nfit with:  --files red_sample_shots_normal.csv "
          f"red_sample_shots_{FAULT}.csv   (classes: {NORMAL}, {FAULT})")


if __name__ == "__main__":
    main()
