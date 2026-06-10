"""
Shared pytest fixtures + path setup.

The vendored scripts under ../references/ are designed to be copied into a
user's project, not installed as a package. Tests mimic that by adding
references/ to sys.path so `import data_preprocessor` works.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REFERENCES_DIR = Path(__file__).parent.parent / "references"
sys.path.insert(0, str(REFERENCES_DIR))


@pytest.fixture
def regular_df():
    """30 minutes of 1-minute samples, two sensors, no NaN, no gaps."""
    ts = pd.date_range("2026-01-01 00:00:00", periods=30, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": np.linspace(0.0, 29.0, 30),
            "sensor_b": np.linspace(100.0, 129.0, 30),
        }
    )


@pytest.fixture
def gap_df():
    """40 rows with a single 10-minute gap mid-stream (rows 15..24 missing
    in timestamp), all sensors non-null where present."""
    ts_part_a = pd.date_range("2026-01-01 00:00:00", periods=15, freq="1min")
    ts_part_b = pd.date_range("2026-01-01 00:25:00", periods=15, freq="1min")
    ts = ts_part_a.append(ts_part_b)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": np.arange(30, dtype=float),
            "sensor_b": np.arange(30, dtype=float) * 2,
        }
    )


@pytest.fixture
def short_nan_df():
    """30 rows of 1-min samples with a 2-row NaN run in sensor_a (rows 10-11),
    short enough to interpolate within a block under gap_threshold_samples=5."""
    ts = pd.date_range("2026-01-01 00:00:00", periods=30, freq="1min")
    sensor_a_values = np.linspace(0.0, 29.0, 30)
    sensor_a_values[10:12] = np.nan
    return pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": sensor_a_values,
            "sensor_b": np.linspace(100.0, 129.0, 30),
        }
    )


@pytest.fixture
def long_nan_df():
    """30 rows with a 7-row NaN run in sensor_a (rows 10-16), which exceeds
    a gap_threshold_samples=5 setting and should split the dataframe into
    two blocks (with the 7 bad rows dropped)."""
    ts = pd.date_range("2026-01-01 00:00:00", periods=30, freq="1min")
    sensor_a_values = np.linspace(0.0, 29.0, 30)
    sensor_a_values[10:17] = np.nan
    return pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": sensor_a_values,
            "sensor_b": np.linspace(100.0, 129.0, 30),
        }
    )


@pytest.fixture
def mixed_nan_df():
    """30 rows with mixed-length NaN runs in two sensors:
      - sensor_a: 2-row NaN at 5-6 (short)
      - sensor_b: 7-row NaN at 20-26 (long → splits block)
    Useful for asserting that ANY sensor's long NaN triggers the split.
    """
    ts = pd.date_range("2026-01-01 00:00:00", periods=30, freq="1min")
    sensor_a_values = np.linspace(0.0, 29.0, 30)
    sensor_a_values[5:7] = np.nan
    sensor_b_values = np.linspace(100.0, 129.0, 30)
    sensor_b_values[20:27] = np.nan
    return pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": sensor_a_values,
            "sensor_b": sensor_b_values,
        }
    )


@pytest.fixture
def embeddings_df():
    """Fake embeddings dataframe mimicking what an Omega embedding step
    would produce: 6 windows × 2 sensors = 12 rows, each with a 4-dim
    embedding vector (small for fast tests; production is 768).
    Labels are consistent per window across sensors.
    """
    rng = np.random.default_rng(seed=42)
    records = []
    for read_idx in range(6):
        # alternate label every 3 windows
        label = "normal" if read_idx < 3 else "fault"
        ts = pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(minutes=read_idx)
        for sensor in ["sensor_a", "sensor_b"]:
            records.append(
                {
                    "sensor": sensor,
                    "read_index": read_idx,
                    "window_size": 16,
                    "embedding": rng.standard_normal(4).astype(np.float32),
                    "timestamp": ts,
                    "machine_state": label,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def small_embeddings_df():
    """Same shape as embeddings_df but smaller — 3 windows × 2 sensors.
    Used for size-edge cases like PCA validation."""
    rng = np.random.default_rng(seed=7)
    records = []
    for read_idx in range(3):
        for sensor in ["sensor_a", "sensor_b"]:
            records.append(
                {
                    "sensor": sensor,
                    "read_index": read_idx,
                    "window_size": 16,
                    "embedding": rng.standard_normal(4).astype(np.float32),
                    "machine_state": "normal",
                }
            )
    return pd.DataFrame(records)
