"""Tests for DataPreprocessor — diagnostics + gap-aware blocking."""

import numpy as np
import pandas as pd
import pytest

from data_preprocessor import DataPreprocessor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_init():
    prep = DataPreprocessor()
    assert prep.timestamp_col == "timestamp"
    assert prep.sampling_rate_minutes == 1
    assert prep.gap_threshold_samples == 5
    assert prep.drop_sensors == []
    assert prep.imputation_method == "linear"
    assert prep.imputation_kwargs == {}


def test_custom_init():
    prep = DataPreprocessor(
        timestamp_col="ts",
        sampling_rate_minutes=5,
        gap_threshold_samples=10,
        drop_sensors=["sensor_x"],
        imputation_method="spline",
        imputation_kwargs={"order": 3},
    )
    assert prep.timestamp_col == "ts"
    assert prep.sampling_rate_minutes == 5
    assert prep.gap_threshold_samples == 10
    assert prep.drop_sensors == ["sensor_x"]
    assert prep.imputation_method == "spline"
    assert prep.imputation_kwargs == {"order": 3}


# ---------------------------------------------------------------------------
# _max_consecutive_true (run-length encoding helper)
# ---------------------------------------------------------------------------

def test_max_consecutive_all_false():
    mask = pd.Series([False] * 10)
    assert DataPreprocessor._max_consecutive_true(mask) == 0


def test_max_consecutive_all_true():
    mask = pd.Series([True] * 10)
    assert DataPreprocessor._max_consecutive_true(mask) == 10


def test_max_consecutive_mixed():
    # runs of True: 1, 3, 2 → max 3
    mask = pd.Series([True, False, True, True, True, False, True, True, False])
    assert DataPreprocessor._max_consecutive_true(mask) == 3


def test_max_consecutive_true_at_boundaries():
    mask = pd.Series([True, True, False, True])
    assert DataPreprocessor._max_consecutive_true(mask) == 2


# ---------------------------------------------------------------------------
# diagnose() — timestamp section
# ---------------------------------------------------------------------------

def test_diagnose_missing_timestamp_column():
    df = pd.DataFrame({"sensor_a": [1.0, 2.0, 3.0]})
    report = DataPreprocessor(timestamp_col="timestamp").diagnose(df)
    assert report["timestamp"]["present"] is False
    assert report["timestamp"]["n_rows"] == 3


def test_diagnose_datetime_column(regular_df):
    report = DataPreprocessor().diagnose(regular_df)
    ts = report["timestamp"]
    assert ts["present"] is True
    assert ts["is_datetime"] is True
    assert ts["is_sorted"] is True
    assert ts["n_duplicates"] == 0


def test_diagnose_coercible_string_column():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "sensor_a": [1.0, 2.0, 3.0],
        }
    )
    report = DataPreprocessor().diagnose(df)
    assert report["timestamp"]["is_datetime"] is False
    assert report["timestamp"]["coercible_to_datetime"] is True


def test_diagnose_non_coercible_column():
    df = pd.DataFrame(
        {
            "timestamp": ["not-a-date", "still-not", "nope"],
            "sensor_a": [1.0, 2.0, 3.0],
        }
    )
    report = DataPreprocessor().diagnose(df)
    assert report["timestamp"]["is_datetime"] is False
    assert report["timestamp"]["coercible_to_datetime"] is False
    # Should bail before sampling info
    assert "sampling" not in report["timestamp"]


def test_diagnose_detects_unsorted(regular_df):
    shuffled = regular_df.sample(frac=1, random_state=0).reset_index(drop=True)
    report = DataPreprocessor().diagnose(shuffled)
    assert report["timestamp"]["is_sorted"] is False


def test_diagnose_detects_duplicates(regular_df):
    df_with_dup = pd.concat([regular_df, regular_df.iloc[[5]]], ignore_index=True)
    report = DataPreprocessor().diagnose(df_with_dup)
    assert report["timestamp"]["n_duplicates"] == 1


def test_diagnose_regular_sampling(regular_df):
    report = DataPreprocessor(sampling_rate_minutes=1).diagnose(regular_df)
    sampling = report["timestamp"]["sampling"]
    assert sampling["median_delta"] == pd.Timedelta(minutes=1)
    assert sampling["matches_expected"] is True
    # CV should be ~0 for perfectly regular sampling
    assert sampling["regularity_cv"] == pytest.approx(0.0, abs=1e-6)


def test_diagnose_irregular_sampling(gap_df):
    report = DataPreprocessor(sampling_rate_minutes=1).diagnose(gap_df)
    sampling = report["timestamp"]["sampling"]
    # Fixture: part-A ends 00:14, part-B starts 00:25 → 11-minute gap.
    assert sampling["max_delta"] == pd.Timedelta(minutes=11)
    assert sampling["min_delta"] == pd.Timedelta(minutes=1)
    assert sampling["regularity_cv"] > 0.1  # noticeably irregular


def test_diagnose_gap_detection(gap_df):
    report = DataPreprocessor().diagnose(gap_df, gap_multiplier=2.0)
    gaps = report["timestamp"]["gaps"]
    assert gaps["count"] == 1
    top = gaps["top"][0]
    assert top["size"] == pd.Timedelta(minutes=11)


def test_diagnose_top_n_gaps_respects_limit():
    # Build a df with 3 gaps of different sizes.
    blocks = [
        pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min"),
        pd.date_range("2026-01-01 00:08:00", periods=5, freq="1min"),  # 4min gap
        pd.date_range("2026-01-01 00:20:00", periods=5, freq="1min"),  # 8min gap
        pd.date_range("2026-01-01 00:30:00", periods=5, freq="1min"),  # 6min gap
    ]
    ts = blocks[0]
    for block in blocks[1:]:
        ts = ts.append(block)
    df = pd.DataFrame({"timestamp": ts, "sensor_a": np.arange(len(ts), dtype=float)})

    report = DataPreprocessor().diagnose(df, top_n_gaps=2)
    gaps = report["timestamp"]["gaps"]
    assert gaps["count"] == 3
    # Top 2, sorted descending: 8min then 6min
    assert len(gaps["top"]) == 2
    assert gaps["top"][0]["size"] == pd.Timedelta(minutes=8)
    assert gaps["top"][1]["size"] == pd.Timedelta(minutes=6)


def test_diagnose_no_gaps_empty_top(regular_df):
    report = DataPreprocessor().diagnose(regular_df, gap_multiplier=2.0)
    gaps = report["timestamp"]["gaps"]
    assert gaps["count"] == 0
    assert gaps["top"] == []
    assert gaps["total_duration"] == pd.Timedelta(0)


def test_diagnose_single_unique_timestamp():
    """All rows share one timestamp → no deltas → sampling=None."""
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 5,
            "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    report = DataPreprocessor().diagnose(df)
    assert report["timestamp"]["sampling"] is None


# ---------------------------------------------------------------------------
# diagnose() — nulls section
# ---------------------------------------------------------------------------

def test_diagnose_no_nulls(regular_df):
    report = DataPreprocessor().diagnose(regular_df)
    nulls = report["nulls"]
    assert nulls["columns_with_nulls"] == 0
    for col in nulls["per_column"]:
        assert col["n_null"] == 0
        assert col["pct_null"] == 0.0
        assert col["max_consecutive_null"] == 0


def test_diagnose_counts_nulls_per_column(short_nan_df):
    report = DataPreprocessor().diagnose(short_nan_df)
    nulls = report["nulls"]
    assert nulls["columns_with_nulls"] == 1

    sensor_a_entry = next(entry for entry in nulls["per_column"] if entry["column"] == "sensor_a")
    assert sensor_a_entry["n_null"] == 2
    assert sensor_a_entry["pct_null"] == pytest.approx(2 / 30 * 100)
    assert sensor_a_entry["max_consecutive_null"] == 2


def test_diagnose_excludes_timestamp_from_nulls(regular_df):
    report = DataPreprocessor().diagnose(regular_df)
    columns = [entry["column"] for entry in report["nulls"]["per_column"]]
    assert "timestamp" not in columns


def test_diagnose_sorts_nulls_by_pct_desc(mixed_nan_df):
    report = DataPreprocessor().diagnose(mixed_nan_df)
    pcts = [entry["pct_null"] for entry in report["nulls"]["per_column"]]
    assert pcts == sorted(pcts, reverse=True)


def test_diagnose_max_consecutive_null_long_run(long_nan_df):
    report = DataPreprocessor().diagnose(long_nan_df)
    sensor_a_entry = next(
        entry for entry in report["nulls"]["per_column"] if entry["column"] == "sensor_a"
    )
    assert sensor_a_entry["max_consecutive_null"] == 7


def test_diagnose_empty_dataframe():
    df = pd.DataFrame({"timestamp": pd.to_datetime([]), "sensor_a": []})
    report = DataPreprocessor().diagnose(df)
    assert report["timestamp"]["n_rows"] == 0
    assert report["nulls"]["n_rows"] == 0
    assert report["nulls"]["columns_with_nulls"] == 0


# ---------------------------------------------------------------------------
# build() — gap-aware blocking + imputation
# ---------------------------------------------------------------------------

def test_build_no_gaps_single_block(regular_df):
    out = DataPreprocessor().build(regular_df)
    assert len(out) == len(regular_df)
    assert out["block_id"].nunique() == 1
    assert not out["imputed"].any()


def test_build_drops_long_gap_rows(long_nan_df):
    out = DataPreprocessor(gap_threshold_samples=5).build(long_nan_df)
    # 7 NaN rows in sensor_a should be dropped, leaving 23 rows.
    assert len(out) == 23
    # Two distinct blocks.
    assert out["block_id"].nunique() == 2


def test_build_imputes_short_nan_within_block_linear(short_nan_df):
    out = DataPreprocessor(
        gap_threshold_samples=5, imputation_method="linear"
    ).build(short_nan_df)
    # Short NaN run does not split the block.
    assert out["block_id"].nunique() == 1
    # Original NaN rows are now imputed (no NaN remains).
    assert not out["sensor_a"].isna().any()
    # Rows 10-11 should be flagged as imputed.
    imputed_indices = out.index[out["imputed"]].tolist()
    assert 10 in imputed_indices and 11 in imputed_indices
    # Linear interpolation between row 9 (=9.0) and row 12 (=12.0): expect 10.0, 11.0
    assert out.loc[10, "sensor_a"] == pytest.approx(10.0)
    assert out.loc[11, "sensor_a"] == pytest.approx(11.0)


def test_build_imputes_short_nan_ffill():
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    sensor_values = [0.0, 1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = pd.DataFrame({"timestamp": ts, "sensor_a": sensor_values})

    out = DataPreprocessor(
        gap_threshold_samples=5, imputation_method="ffill"
    ).build(df)
    # ffill: NaN at rows 3-4 inherit from row 2 (=2.0)
    assert out.loc[3, "sensor_a"] == 2.0
    assert out.loc[4, "sensor_a"] == 2.0


def test_build_imputes_short_nan_bfill():
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    sensor_values = [0.0, 1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = pd.DataFrame({"timestamp": ts, "sensor_a": sensor_values})

    out = DataPreprocessor(
        gap_threshold_samples=5, imputation_method="bfill"
    ).build(df)
    # bfill: NaN at rows 3-4 inherit from row 5 (=5.0)
    assert out.loc[3, "sensor_a"] == 5.0
    assert out.loc[4, "sensor_a"] == 5.0


def test_build_long_nan_in_any_sensor_splits_block(mixed_nan_df):
    out = DataPreprocessor(gap_threshold_samples=5).build(mixed_nan_df)
    # sensor_b's 7-row NaN should trigger a split, regardless of sensor_a's short run.
    assert out["block_id"].nunique() == 2
    # 7 long-gap rows dropped → 23 remain.
    assert len(out) == 23


def test_build_drops_specified_sensors(short_nan_df):
    out = DataPreprocessor(
        gap_threshold_samples=5, drop_sensors=["sensor_b"]
    ).build(short_nan_df)
    assert "sensor_b" not in out.columns
    # sensor_a NaN imputation still happens.
    assert not out["sensor_a"].isna().any()


def test_build_drop_sensors_ignores_unknown_column():
    """If drop_sensors lists a non-existent column, build() should not raise."""
    ts = pd.date_range("2026-01-01", periods=5, freq="1min")
    df = pd.DataFrame({"timestamp": ts, "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = DataPreprocessor(drop_sensors=["nonexistent"]).build(df)
    assert "sensor_a" in out.columns
    assert len(out) == 5


def test_build_sorts_by_timestamp_first():
    ts = pd.date_range("2026-01-01", periods=5, freq="1min")
    # Shuffle the input order.
    df = pd.DataFrame(
        {"timestamp": ts, "sensor_a": [0.0, 1.0, 2.0, 3.0, 4.0]}
    ).sample(frac=1, random_state=0).reset_index(drop=True)

    out = DataPreprocessor().build(df)
    # After build, rows are sorted by timestamp.
    assert out["timestamp"].is_monotonic_increasing
    assert out["sensor_a"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_build_does_not_mutate_input(short_nan_df):
    original = short_nan_df.copy(deep=True)
    DataPreprocessor().build(short_nan_df)
    pd.testing.assert_frame_equal(short_nan_df, original)


def test_build_assigns_block_ids_correctly(long_nan_df):
    out = DataPreprocessor(gap_threshold_samples=5).build(long_nan_df)
    block_ids = sorted(out["block_id"].unique())
    # Block ids are positive and contiguous.
    assert len(block_ids) == 2
    # The first row of each block keeps its order in time.
    block_starts = out.groupby("block_id")["timestamp"].min().sort_values()
    assert list(block_starts.index) == block_ids


def test_build_single_row():
    df = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-01-01")], "sensor_a": [1.0]}
    )
    out = DataPreprocessor().build(df)
    assert len(out) == 1
    assert out.iloc[0]["sensor_a"] == 1.0
    assert not out.iloc[0]["imputed"]


def test_build_all_constant_sensor():
    """A constant sensor with no NaN should pass through unchanged."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    df = pd.DataFrame({"timestamp": ts, "sensor_a": [5.0] * 10})
    out = DataPreprocessor().build(df)
    assert len(out) == 10
    assert (out["sensor_a"] == 5.0).all()


def test_build_imputed_flag_only_where_original_was_nan(short_nan_df):
    out = DataPreprocessor(gap_threshold_samples=5).build(short_nan_df)
    # Only rows 10-11 of the original sensor_a were NaN.
    expected_imputed_rows = {10, 11}
    actual_imputed_rows = set(out.index[out["imputed"]].tolist())
    assert actual_imputed_rows == expected_imputed_rows


def test_non_sensor_columns_customizable():
    """A custom non-sensor column is excluded from imputation/blocking."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": np.arange(10, dtype=float),
            "status_flag": np.ones(10),  # numeric but not a sensor
        }
    )
    preprocessor = DataPreprocessor(non_sensor_columns=["status_flag"])
    assert preprocessor._get_sensor_cols(df) == ["sensor_a"]


def test_non_sensor_columns_default_excludes_machine_status():
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "sensor_a": np.arange(10, dtype=float),
            "machine_status": np.ones(10),
        }
    )
    assert DataPreprocessor()._get_sensor_cols(df) == ["sensor_a"]
