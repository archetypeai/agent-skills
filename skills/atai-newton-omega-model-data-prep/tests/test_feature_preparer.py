"""Tests for FeaturePreparer — joint-state featurization + normalization + PCA."""

import numpy as np
import pandas as pd
import pytest

from feature_preparer import FeaturePreparer


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_init():
    preparer = FeaturePreparer()
    assert preparer.normalize is None
    assert preparer.reduce_dim is None
    assert preparer.sensor_order is None


def test_invalid_normalize_raises():
    with pytest.raises(ValueError, match="normalize must be None"):
        FeaturePreparer(normalize="minmax")


@pytest.mark.parametrize("normalize", [None, "l2", "standardize"])
def test_valid_normalize_values(normalize):
    preparer = FeaturePreparer(normalize=normalize)
    assert preparer.normalize == normalize


# ---------------------------------------------------------------------------
# Validation — windows must have all sensors
# ---------------------------------------------------------------------------

def test_missing_sensor_in_some_windows_raises(embeddings_df):
    # Drop the sensor_b row for one window → that window now has only 1 sensor.
    broken = embeddings_df.drop(
        embeddings_df[
            (embeddings_df["sensor"] == "sensor_b")
            & (embeddings_df["read_index"] == 2)
        ].index
    ).reset_index(drop=True)

    with pytest.raises(ValueError, match="missing sensors"):
        FeaturePreparer().prepare(broken)


def test_missing_label_column_raises(embeddings_df):
    with pytest.raises(ValueError, match="not found in df_emb"):
        FeaturePreparer().prepare(embeddings_df, label_column="not_a_real_column")


def test_inconsistent_labels_across_sensors_raises(embeddings_df):
    # Flip one sensor's label for one window → that window has inconsistent labels.
    df = embeddings_df.copy()
    mask = (df["sensor"] == "sensor_b") & (df["read_index"] == 0)
    df.loc[mask, "machine_state"] = "fault"
    with pytest.raises(ValueError, match="Inconsistent labels"):
        FeaturePreparer().prepare(df, label_column="machine_state")


# ---------------------------------------------------------------------------
# prepare() — shape and basic structure
# ---------------------------------------------------------------------------

def test_prepare_no_pca_shape(embeddings_df):
    # 6 windows × 2 sensors × 4-dim embedding = (6, 8)
    X, y, metadata = FeaturePreparer().prepare(
        embeddings_df, label_column="machine_state"
    )
    assert X.shape == (6, 8)
    assert y.shape == (6,)
    assert len(metadata) == 6


def test_prepare_with_pca_shape(embeddings_df):
    # PCA needs reduce_dim < D (8) and < N (6); use 3.
    X, _, _ = FeaturePreparer(reduce_dim=3).prepare(
        embeddings_df, label_column="machine_state"
    )
    assert X.shape == (6, 3)


def test_prepare_no_label_returns_y_none(embeddings_df):
    X, y, metadata = FeaturePreparer().prepare(embeddings_df, label_column=None)
    assert y is None
    assert X.shape == (6, 8)
    assert "machine_state" not in metadata.columns


def test_prepare_concatenation_order_alphabetical(embeddings_df):
    """Default sensor_order is alphabetical → sensor_a first, sensor_b second."""
    preparer = FeaturePreparer()
    X, _, _ = preparer.prepare(embeddings_df, label_column="machine_state")

    # For window 0, the first 4 dims should be sensor_a's embedding, next 4 sensor_b's.
    sensor_a_emb_0 = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_a'"
    )["embedding"].iloc[0]
    sensor_b_emb_0 = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_b'"
    )["embedding"].iloc[0]
    np.testing.assert_allclose(X[0, :4], sensor_a_emb_0)
    np.testing.assert_allclose(X[0, 4:], sensor_b_emb_0)


def test_prepare_sensor_order_explicit_overrides_default(embeddings_df):
    """sensor_order=['sensor_b', 'sensor_a'] → b's embedding comes first."""
    preparer = FeaturePreparer(sensor_order=["sensor_b", "sensor_a"])
    X, _, _ = preparer.prepare(embeddings_df, label_column="machine_state")

    sensor_a_emb_0 = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_a'"
    )["embedding"].iloc[0]
    sensor_b_emb_0 = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_b'"
    )["embedding"].iloc[0]
    np.testing.assert_allclose(X[0, :4], sensor_b_emb_0)
    np.testing.assert_allclose(X[0, 4:], sensor_a_emb_0)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_none_preserves_values(embeddings_df):
    X, _, _ = FeaturePreparer(normalize=None).prepare(embeddings_df)

    # Build the expected raw concatenation manually for window 0.
    sensor_a = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_a'"
    )["embedding"].iloc[0]
    sensor_b = embeddings_df.query(
        "read_index == 0 and sensor == 'sensor_b'"
    )["embedding"].iloc[0]
    expected = np.concatenate([sensor_a, sensor_b])
    np.testing.assert_allclose(X[0], expected)


def test_normalize_l2_unit_norm(embeddings_df):
    X, _, _ = FeaturePreparer(normalize="l2").prepare(embeddings_df)
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_normalize_l2_zero_vector_safe():
    """A zero vector after concat shouldn't crash (division by zero)."""
    df = pd.DataFrame(
        [
            {
                "sensor": "sensor_a",
                "read_index": 0,
                "embedding": np.zeros(4, dtype=np.float32),
            },
            {
                "sensor": "sensor_b",
                "read_index": 0,
                "embedding": np.zeros(4, dtype=np.float32),
            },
        ]
    )
    X, _, _ = FeaturePreparer(normalize="l2").prepare(df)
    # Zero vector stays zero (we replaced norms==0 with 1 to avoid NaN).
    np.testing.assert_allclose(X[0], np.zeros(8))


def test_normalize_standardize_zero_mean_unit_std(embeddings_df):
    X, _, _ = FeaturePreparer(normalize="standardize").prepare(embeddings_df)
    # StandardScaler standardizes per-column: each column mean ~0, std ~1.
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-6)
    np.testing.assert_allclose(stds, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def test_pca_dim_equal_to_d_raises(embeddings_df):
    # D = 8 (2 sensors × 4 dims). Requesting reduce_dim=8 is not strictly smaller.
    with pytest.raises(ValueError, match="reduce_dim"):
        FeaturePreparer(reduce_dim=8).prepare(embeddings_df)


def test_pca_dim_greater_than_d_raises(embeddings_df):
    with pytest.raises(ValueError, match="reduce_dim"):
        FeaturePreparer(reduce_dim=100).prepare(embeddings_df)


def test_pca_dim_equal_to_n_raises(small_embeddings_df):
    # N = 3 windows. reduce_dim=3 is not strictly smaller than N.
    with pytest.raises(ValueError, match="smaller than number of samples"):
        FeaturePreparer(reduce_dim=3).prepare(small_embeddings_df)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_y_matches_per_window_labels(embeddings_df):
    _, y, _ = FeaturePreparer().prepare(embeddings_df, label_column="machine_state")
    # Each window's label is the same across its sensors → first 3 windows "normal",
    # last 3 windows "fault" (per fixture).
    assert list(y) == ["normal", "normal", "normal", "fault", "fault", "fault"]


def test_y_dtype_preserved_with_strings(embeddings_df):
    _, y, _ = FeaturePreparer().prepare(embeddings_df, label_column="machine_state")
    assert y.dtype.kind in ("U", "O")


def test_y_dtype_preserved_with_ints():
    df = pd.DataFrame(
        [
            {
                "sensor": "sensor_a",
                "read_index": row_index,
                "embedding": np.zeros(4, dtype=np.float32),
                "label": row_index % 2,
            }
            for row_index in range(4)
        ]
    )
    _, y, _ = FeaturePreparer().prepare(df, label_column="label")
    assert y.dtype.kind == "i"
    assert list(y) == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_includes_read_index(embeddings_df):
    _, _, metadata = FeaturePreparer().prepare(embeddings_df)
    assert "read_index" in metadata.columns
    assert list(metadata["read_index"]) == [0, 1, 2, 3, 4, 5]


def test_metadata_includes_label_column_if_provided(embeddings_df):
    _, _, metadata = FeaturePreparer().prepare(
        embeddings_df, label_column="machine_state"
    )
    assert "machine_state" in metadata.columns


def test_metadata_includes_timestamp_if_present(embeddings_df):
    _, _, metadata = FeaturePreparer().prepare(embeddings_df)
    assert "timestamp" in metadata.columns
    # First window's timestamp should match the first record's.
    expected = embeddings_df["timestamp"].iloc[0]
    assert metadata["timestamp"].iloc[0] == expected


def test_metadata_omits_timestamp_when_absent():
    df = pd.DataFrame(
        [
            {
                "sensor": "sensor_a",
                "read_index": row_index,
                "embedding": np.zeros(4, dtype=np.float32),
                "machine_state": "normal",
            }
            for row_index in range(3)
        ]
    )
    _, _, metadata = FeaturePreparer().prepare(df, label_column="machine_state")
    assert "timestamp" not in metadata.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_sensor_single_window():
    """Minimum viable input — one sensor, one window."""
    df = pd.DataFrame(
        [
            {
                "sensor": "sensor_a",
                "read_index": 0,
                "embedding": np.arange(4, dtype=np.float32),
                "machine_state": "normal",
            }
        ]
    )
    X, y, metadata = FeaturePreparer().prepare(df, label_column="machine_state")
    assert X.shape == (1, 4)
    np.testing.assert_array_equal(X[0], np.arange(4))
    assert list(y) == ["normal"]
    assert list(metadata["read_index"]) == [0]


def test_three_sensors_concat():
    """Verify joint-state concatenation works for >2 sensors."""
    rng = np.random.default_rng(0)
    records = []
    for read_idx in range(4):
        for sensor in ["sensor_a", "sensor_b", "sensor_c"]:
            records.append(
                {
                    "sensor": sensor,
                    "read_index": read_idx,
                    "embedding": rng.standard_normal(4).astype(np.float32),
                    "machine_state": "normal",
                }
            )
    df = pd.DataFrame(records)
    X, _, _ = FeaturePreparer().prepare(df, label_column="machine_state")
    assert X.shape == (4, 12)  # 4 windows × 3 sensors × 4 dims
