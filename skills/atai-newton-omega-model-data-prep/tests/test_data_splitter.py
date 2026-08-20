"""Tests for DataSplitter — out-of-time / random train/test partitioning."""

import numpy as np
import pandas as pd
import pytest

from data_splitter import DataSplitter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_init():
    splitter = DataSplitter()
    assert splitter.mode == "oot"
    assert splitter.test_size == 0.3
    assert splitter.timestamp_column == "timestamp"
    assert splitter.random_state == 42


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode 'kfold' is invalid"):
        DataSplitter(mode="kfold")


@pytest.mark.parametrize("bad_size", [0, 1, -0.1, 1.5, 2.0])
def test_invalid_test_size_raises(bad_size):
    with pytest.raises(ValueError, match="test_size must be in"):
        DataSplitter(test_size=bad_size)


@pytest.mark.parametrize("mode", ["oot", "random"])
def test_valid_modes_construct(mode):
    splitter = DataSplitter(mode=mode)
    assert splitter.mode == mode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dataset(row_count=20, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((row_count, n_features))
    y = np.arange(row_count) % 2  # alternating 0/1
    ts = pd.date_range("2026-01-01 00:00:00", periods=row_count, freq="1min")
    metadata = pd.DataFrame({"timestamp": ts, "read_index": np.arange(row_count)})
    return X, y, metadata


# ---------------------------------------------------------------------------
# OOT mode
# ---------------------------------------------------------------------------

def test_oot_train_strictly_before_test_in_time():
    X, y, metadata = _make_dataset(row_count=20)
    splitter = DataSplitter(mode="oot", test_size=0.3)
    _, _, _, _, meta_train, meta_test = splitter.split(X, y, metadata)

    # Every train timestamp must be ≤ every test timestamp.
    max_train_ts = pd.to_datetime(meta_train["timestamp"]).max()
    min_test_ts = pd.to_datetime(meta_test["timestamp"]).min()
    assert max_train_ts <= min_test_ts


def test_oot_test_size_proportions():
    X, y, metadata = _make_dataset(row_count=20)
    X_train, X_test, _, _, _, _ = DataSplitter(
        mode="oot", test_size=0.25
    ).split(X, y, metadata)
    assert len(X_test) == 5
    assert len(X_train) == 15


def test_oot_with_unsorted_input_still_chronological():
    """Even if metadata is not in chronological order, OOT split should
    pick the chronologically-latest rows as test."""
    X, y, metadata = _make_dataset(row_count=20)
    # Shuffle.
    order = np.random.default_rng(0).permutation(20)
    X = X[order]
    y = y[order]
    metadata = metadata.iloc[order].reset_index(drop=True)

    _, _, _, _, meta_train, meta_test = DataSplitter(
        mode="oot", test_size=0.3
    ).split(X, y, metadata)

    max_train_ts = pd.to_datetime(meta_train["timestamp"]).max()
    min_test_ts = pd.to_datetime(meta_test["timestamp"]).min()
    assert max_train_ts <= min_test_ts


def test_oot_string_timestamps_coerced():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20)
    metadata = pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-{day_index + 1:02d}" for day_index in range(20)
            ],
        }
    )
    splitter = DataSplitter(mode="oot", test_size=0.3)
    _, _, _, _, meta_train, meta_test = splitter.split(X, y, metadata)
    max_train_ts = pd.to_datetime(meta_train["timestamp"]).max()
    min_test_ts = pd.to_datetime(meta_test["timestamp"]).min()
    assert max_train_ts <= min_test_ts


def test_oot_missing_timestamp_column_raises():
    X, y, metadata = _make_dataset(row_count=20)
    metadata = metadata.drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="OOT mode requires column 'timestamp'"):
        DataSplitter(mode="oot").split(X, y, metadata)


def test_oot_custom_timestamp_column():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20)
    ts = pd.date_range("2026-01-01", periods=20, freq="1D")
    metadata = pd.DataFrame({"sample_time": ts})

    splitter = DataSplitter(mode="oot", timestamp_column="sample_time", test_size=0.3)
    _, _, _, _, meta_train, meta_test = splitter.split(X, y, metadata)
    max_train_ts = pd.to_datetime(meta_train["sample_time"]).max()
    min_test_ts = pd.to_datetime(meta_test["sample_time"]).min()
    assert max_train_ts <= min_test_ts


def test_oot_preserves_feature_label_alignment():
    """The y_test labels must still match the rows in X_test after splitting."""
    row_count = 20
    X = np.arange(row_count).reshape(-1, 1).astype(float)
    y = X.flatten()  # label = first feature
    ts = pd.date_range("2026-01-01", periods=row_count, freq="1min")
    metadata = pd.DataFrame({"timestamp": ts})

    X_train, X_test, y_train, y_test, _, _ = DataSplitter(
        mode="oot", test_size=0.3
    ).split(X, y, metadata)
    np.testing.assert_array_equal(X_train.flatten(), y_train)
    np.testing.assert_array_equal(X_test.flatten(), y_test)


# ---------------------------------------------------------------------------
# Random mode
# ---------------------------------------------------------------------------

def test_random_deterministic_with_same_seed():
    X, y, metadata = _make_dataset(row_count=20)
    s1 = DataSplitter(mode="random", random_state=123, test_size=0.3)
    s2 = DataSplitter(mode="random", random_state=123, test_size=0.3)
    out1 = s1.split(X, y, metadata)
    out2 = s2.split(X, y, metadata)
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])


def test_random_different_seeds_produce_different_splits():
    X, y, metadata = _make_dataset(row_count=50)
    s1 = DataSplitter(mode="random", random_state=1, test_size=0.3)
    s2 = DataSplitter(mode="random", random_state=999, test_size=0.3)
    X_train_1, _, _, _, _, _ = s1.split(X, y, metadata)
    X_train_2, _, _, _, _, _ = s2.split(X, y, metadata)
    # Extremely unlikely they're identical.
    assert not np.array_equal(X_train_1, X_train_2)


def test_random_proportions():
    X, y, metadata = _make_dataset(row_count=100)
    X_train, X_test, _, _, _, _ = DataSplitter(
        mode="random", test_size=0.2
    ).split(X, y, metadata)
    assert len(X_test) == 20
    assert len(X_train) == 80


def test_random_does_not_require_timestamp_column():
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20)
    metadata = pd.DataFrame({"sample_id": range(20)})
    # No timestamp column at all — random mode shouldn't care.
    splitter = DataSplitter(mode="random", test_size=0.3)
    X_train, X_test, _, _, _, _ = splitter.split(X, y, metadata)
    assert len(X_test) + len(X_train) == 20


# ---------------------------------------------------------------------------
# y handling
# ---------------------------------------------------------------------------

def test_y_none_returns_none_for_both():
    X, _, metadata = _make_dataset(row_count=20)
    _, _, y_train, y_test, _, _ = DataSplitter(mode="oot").split(X, None, metadata)
    assert y_train is None
    assert y_test is None


def test_y_string_labels_preserved():
    row_count = 20
    X = np.arange(row_count).reshape(-1, 1).astype(float)
    y = np.array(["normal" if row_index < 10 else "fault" for row_index in range(row_count)])
    metadata = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=row_count, freq="1min")}
    )
    _, _, y_train, y_test, _, _ = DataSplitter(
        mode="oot", test_size=0.5
    ).split(X, y, metadata)
    # OOT with 50/50: train = first 10 ("normal"), test = last 10 ("fault")
    assert set(y_train) == {"normal"}
    assert set(y_test) == {"fault"}


# ---------------------------------------------------------------------------
# Metadata handling
# ---------------------------------------------------------------------------

def test_metadata_columns_preserved():
    row_count = 20
    X = np.arange(row_count).reshape(-1, 1).astype(float)
    y = np.arange(row_count)
    metadata = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="1min"),
            "read_index": np.arange(row_count),
            "well_id": [f"well_{row_index % 3}" for row_index in range(row_count)],
        }
    )
    _, _, _, _, meta_train, meta_test = DataSplitter(
        mode="oot", test_size=0.3
    ).split(X, y, metadata)
    assert list(meta_train.columns) == ["timestamp", "read_index", "well_id"]
    assert list(meta_test.columns) == ["timestamp", "read_index", "well_id"]


def test_metadata_index_is_reset():
    X, y, metadata = _make_dataset(row_count=20)
    # Pre-set non-default indices.
    metadata.index = range(100, 120)
    _, _, _, _, meta_train, meta_test = DataSplitter(
        mode="oot", test_size=0.3
    ).split(X, y, metadata)
    # Index should be reset to 0..N-1 in each partition.
    assert list(meta_train.index) == list(range(len(meta_train)))
    assert list(meta_test.index) == list(range(len(meta_test)))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_test_size_producing_empty_test_raises():
    """test_size=0.01 on 5 samples → n_test=0 → should raise."""
    X, y, metadata = _make_dataset(row_count=5)
    splitter = DataSplitter(mode="oot", test_size=0.01)
    with pytest.raises(ValueError, match="empty train or test set"):
        splitter.split(X, y, metadata)


def test_minimum_viable_split():
    """N=2, test_size=0.5 → n_train=1, n_test=1. Should not raise."""
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    metadata = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=2, freq="1min")}
    )
    X_train, X_test, y_train, y_test, _, _ = DataSplitter(
        mode="oot", test_size=0.5
    ).split(X, y, metadata)
    assert len(X_train) == 1
    assert len(X_test) == 1
