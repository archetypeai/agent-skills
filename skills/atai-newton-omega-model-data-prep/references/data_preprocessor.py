"""
Builds continuous temporal blocks from a multivariate time series with gaps.

A "block" is a continuous segment where:
  - All selected sensors have data (NaNs are imputed within the block)
  - No gap of NaN exceeds `gap_threshold` consecutive samples in any sensor

Blocks are separated when any sensor has a NaN run longer than the threshold,
because imputing across long gaps would fabricate data.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

# Default non-sensor columns excluded from imputation/blocking (overridable
# via the `non_sensor_columns` constructor argument).
NON_SENSOR_COLUMNS = ['machine_status']


class DataPreprocessor:
    def __init__(
        self,
        timestamp_col: str = 'timestamp',
        sampling_rate_minutes: int = 1,
        gap_threshold_samples: int = 5,
        drop_sensors: Optional[List[str]] = None,
        imputation_method: str = 'linear',
        imputation_kwargs: Optional[Dict[str, Any]] = None,
        non_sensor_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            timestamp_col: name of the timestamp column.
            sampling_rate_minutes: expected interval between samples.
            gap_threshold_samples: max consecutive NaN samples tolerated within
                a block. Longer NaN runs split into a new block.
            drop_sensors: list of sensor columns to drop before processing.
            imputation_method: passed to pandas .interpolate() (e.g. 'linear',
                'time') or the special value 'ffill' / 'bfill'.
            imputation_kwargs: extra keyword args forwarded to .interpolate(),
                e.g. {'order': 3} for spline/polynomial. Ignored for ffill/bfill.
            non_sensor_columns: numeric columns that are NOT sensors (labels,
                status flags, ...) and must be excluded from imputation and
                blocking. Defaults to NON_SENSOR_COLUMNS.
        """
        self.timestamp_col = timestamp_col
        self.sampling_rate_minutes = sampling_rate_minutes
        self.gap_threshold_samples = gap_threshold_samples
        self.drop_sensors = drop_sensors or []
        self.imputation_method = imputation_method
        self.imputation_kwargs = imputation_kwargs or {}
        self.non_sensor_columns = (
            list(non_sensor_columns) if non_sensor_columns is not None else list(NON_SENSOR_COLUMNS)
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def diagnose(
        self,
        df: pd.DataFrame,
        gap_multiplier: float = 2.0,
        top_n_gaps: int = 5,
        verbose: bool = False,
        plot: bool = False,
    ) -> Dict[str, Any]:
        """
        Inspect the dataframe before any processing and return a structured
        report covering the timestamp column and missing values.

        This method does NOT modify the dataframe. It only describes it, so
        you can decide how to clean / impute / drop afterwards.

        Args:
            df: input dataframe.
            gap_multiplier: a delta between consecutive timestamps is flagged
                as a "gap" when it exceeds `gap_multiplier * median_delta`.
            top_n_gaps: how many of the largest gaps to include in the report.
            verbose: if True, also pretty-prints the report to stdout.
            plot: if True, displays two diagnostic charts: a histogram of
                consecutive timestamp deltas, and a missingness map showing
                where NaNs occur in time for each column.

        Returns:
            Dict with two top-level keys: 'timestamp' and 'nulls'.
        """
        report: Dict[str, Any] = {
            'timestamp': self._diagnose_timestamp(df, gap_multiplier, top_n_gaps),
            'nulls': self._diagnose_nulls(df),
        }
        if verbose:
            self._print_report(report)
        if plot:
            self._plot_report(df, report)
        return report

    def _diagnose_timestamp(
        self,
        df: pd.DataFrame,
        gap_multiplier: float,
        top_n_gaps: int,
    ) -> Dict[str, Any]:
        col = self.timestamp_col
        info: Dict[str, Any] = {
            'column': col,
            'present': col in df.columns,
            'n_rows': len(df),
        }
        if not info['present']:
            return info

        ts = df[col]
        info['dtype'] = str(ts.dtype)
        info['is_datetime'] = pd.api.types.is_datetime64_any_dtype(ts)

        # Try to coerce a copy if it's not datetime, so we can still report
        # something useful without mutating the input.
        if not info['is_datetime']:
            try:
                ts = pd.to_datetime(ts, errors='raise')
                info['coercible_to_datetime'] = True
            except (ValueError, TypeError):
                info['coercible_to_datetime'] = False
                return info

        info['timezone'] = str(ts.dt.tz) if ts.dt.tz is not None else None
        info['min'] = ts.min()
        info['max'] = ts.max()
        info['duration'] = ts.max() - ts.min()

        # Order & duplicates
        info['is_sorted'] = ts.is_monotonic_increasing
        info['n_duplicates'] = int(ts.duplicated().sum())

        # Sampling rate from the data (use sorted unique timestamps so
        # duplicates and disorder don't poison the deltas).
        ts_sorted = ts.sort_values().drop_duplicates().reset_index(drop=True)
        deltas = ts_sorted.diff().dropna()

        if deltas.empty:
            info['sampling'] = None
            return info

        median_delta = deltas.median()
        info['sampling'] = {
            'median_delta': median_delta,
            'min_delta': deltas.min(),
            'max_delta': deltas.max(),
            # Coefficient of variation: std/median. Near zero => regular.
            'regularity_cv': float(deltas.std() / median_delta)
                if median_delta.total_seconds() > 0 else None,
            'expected_delta': pd.Timedelta(minutes=self.sampling_rate_minutes),
            'matches_expected': median_delta == pd.Timedelta(minutes=self.sampling_rate_minutes),
        }

        # Gaps: deltas larger than gap_multiplier * median
        gap_threshold = median_delta * gap_multiplier
        gap_mask = deltas > gap_threshold
        gaps = deltas[gap_mask]

        info['gaps'] = {
            'threshold': gap_threshold,
            'count': int(gap_mask.sum()),
            'total_duration': gaps.sum() if not gaps.empty else pd.Timedelta(0),
        }

        if not gaps.empty:
            # For each gap, the "start" is the timestamp right before the jump.
            top = gaps.sort_values(ascending=False).head(top_n_gaps)
            info['gaps']['top'] = [
                {
                    'starts_at': ts_sorted.iloc[gap_index - 1],
                    'ends_at': ts_sorted.iloc[gap_index],
                    'size': delta,
                }
                for gap_index, delta in top.items()
            ]
        else:
            info['gaps']['top'] = []

        return info

    def _diagnose_nulls(self, df: pd.DataFrame) -> Dict[str, Any]:
        n_rows = len(df)
        per_column = []

        for col in df.columns:
            if col == self.timestamp_col:
                continue

            series = df[col]
            is_nan = series.isna()
            n_null = int(is_nan.sum())

            entry: Dict[str, Any] = {
                'column': col,
                'dtype': str(series.dtype),
                'n_null': n_null,
                'pct_null': (n_null / n_rows * 100) if n_rows else 0.0,
                'max_consecutive_null': self._max_consecutive_true(is_nan),
            }
            per_column.append(entry)

        # Sort by pct_null desc so the worst offenders show up first.
        per_column.sort(key=lambda column_stats: column_stats['pct_null'], reverse=True)

        return {
            'n_rows': n_rows,
            'columns_with_nulls': sum(1 for column_stats in per_column if column_stats['n_null'] > 0),
            'per_column': per_column,
        }

    @staticmethod
    def _max_consecutive_true(mask: pd.Series) -> int:
        """Longest run of True values in a boolean Series."""
        if not mask.any():
            return 0
        # Run-length encoding via a cumsum group key.
        run_id = (mask != mask.shift()).cumsum()
        run_sizes = mask.groupby(run_id).sum()  # only True runs contribute
        return int(run_sizes.max())

    @staticmethod
    def _print_report(report: Dict[str, Any]) -> None:
        ts = report['timestamp']
        print('=' * 60)
        print('TIMESTAMP DIAGNOSTICS')
        print('=' * 60)
        print(f"Column           : {ts['column']}")
        print(f"Present          : {ts['present']}")
        if not ts['present']:
            print('  -> column not found, skipping rest.')
        else:
            print(f"Rows             : {ts['n_rows']}")
            print(f"Dtype            : {ts.get('dtype')}")
            print(f"Is datetime      : {ts.get('is_datetime')}")
            if 'coercible_to_datetime' in ts and not ts['is_datetime']:
                print(f"Coercible        : {ts['coercible_to_datetime']}")
            if ts.get('min') is not None:
                print(f"Range            : {ts['min']}  ->  {ts['max']}")
                print(f"Duration         : {ts['duration']}")
                print(f"Timezone         : {ts.get('timezone')}")
                print(f"Sorted           : {ts['is_sorted']}")
                print(f"Duplicates       : {ts['n_duplicates']}")

            sampling = ts.get('sampling')
            if sampling:
                print('-' * 60)
                print('Sampling')
                print(f"  median delta   : {sampling['median_delta']}")
                print(f"  min / max      : {sampling['min_delta']}  /  {sampling['max_delta']}")
                cv = sampling['regularity_cv']
                if cv is not None:
                    label = 'regular' if cv < 0.05 else 'irregular'
                    print(f"  regularity CV  : {cv:.4f}  ({label})")
                print(f"  expected       : {sampling['expected_delta']}  "
                      f"(matches: {sampling['matches_expected']})")

            gaps = ts.get('gaps')
            if gaps:
                print('-' * 60)
                print('Gaps')
                print(f"  threshold      : > {gaps['threshold']}")
                print(f"  count          : {gaps['count']}")
                print(f"  total duration : {gaps['total_duration']}")
                for gap_number, gap_info in enumerate(gaps['top'], 1):
                    print(f"  #{gap_number}: {gap_info['size']}  ({gap_info['starts_at']}  ->  {gap_info['ends_at']})")

        nulls = report['nulls']
        print('=' * 60)
        print('NULL DIAGNOSTICS')
        print('=' * 60)
        print(f"Rows                       : {nulls['n_rows']}")
        print(f"Columns with nulls         : {nulls['columns_with_nulls']}")
        print('-' * 60)
        print(f"{'column':<25}{'dtype':<12}{'n_null':>10}{'pct_null':>10}{'max_run':>10}")
        for column_stats in nulls['per_column']:
            print(
                f"{column_stats['column'][:24]:<25}"
                f"{column_stats['dtype'][:11]:<12}"
                f"{column_stats['n_null']:>10}"
                f"{column_stats['pct_null']:>9.2f}%"
                f"{column_stats['max_consecutive_null']:>10}"
            )
        print('=' * 60)

    def _plot_report(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """
        Two diagnostic plots:
          1. Histogram of consecutive timestamp deltas (log y-scale, since
             gaps are typically rare and would be invisible on linear scale).
          2. Missingness map: rows = columns of the dataframe, x-axis = time,
             black cells = NaN. Reveals whether nulls are clustered (sensor
             outage) or scattered (sporadic noise).
        """
        # Imported lazily so the class works without matplotlib installed
        # for users who never call plot=True.
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ts_info = report['timestamp']
        if not ts_info.get('present') or ts_info.get('sampling') is None:
            print('[plot] timestamp not usable, skipping plots.')
            return

        # --- Plot 1: histogram of deltas ---
        ts = pd.to_datetime(df[self.timestamp_col], errors='coerce')
        ts_sorted = ts.dropna().sort_values().drop_duplicates().reset_index(drop=True)
        deltas = ts_sorted.diff().dropna()
        deltas_seconds = deltas.dt.total_seconds()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(deltas_seconds, bins=50, edgecolor='black', alpha=0.75)
        ax.set_yscale('log')
        ax.set_xlabel('Delta between consecutive timestamps (seconds)')
        ax.set_ylabel('Count (log scale)')
        ax.set_title('Distribution of timestamp deltas')

        # Mark median and expected
        median_s = ts_info['sampling']['median_delta'].total_seconds()
        expected_s = ts_info['sampling']['expected_delta'].total_seconds()
        ax.axvline(median_s, color='green', linestyle='--',
                   label=f'median = {median_s:.0f}s')
        if expected_s != median_s:
            ax.axvline(expected_s, color='orange', linestyle='--',
                       label=f'expected = {expected_s:.0f}s')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # --- Plot 2: missingness map ---
        cols = [column_name for column_name in df.columns if column_name != self.timestamp_col]
        if not cols:
            return

        # Build a 2D boolean array: True where value is NaN.
        # Sort by timestamp so the x-axis is meaningful.
        order = ts.sort_values().index
        df_sorted = df.loc[order, cols]
        nan_matrix = df_sorted.isna().to_numpy().T  # shape: (n_cols, n_rows)

        fig, ax = plt.subplots(figsize=(12, max(2, 0.4 * len(cols) + 1)))
        ax.imshow(
            nan_matrix,
            aspect='auto',
            interpolation='nearest',
            cmap='Greys',
            vmin=0,
            vmax=1,
        )
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
        ax.set_title('Missingness map (black = NaN)')

        # Replace numeric x-axis with timestamp labels at a few positions.
        ts_values = ts.loc[order].reset_index(drop=True)
        value_count = len(ts_values)
        n_ticks = min(6, value_count)
        tick_positions = [
            int(tick_index * (value_count - 1) / (n_ticks - 1)) for tick_index in range(n_ticks)
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [ts_values.iloc[position].strftime('%Y-%m-%d %H:%M') for position in tick_positions],
            rotation=30, ha='right',
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the input DataFrame with two new columns:
          - block_id: integer id of the continuous block (NaN if row dropped)
          - imputed: True if any sensor value in this row was imputed
        Rows that fall in long-gap regions are dropped from the output.
        """
        df = df.copy()
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        # 1. Drop unwanted sensors
        df = df.drop(columns=[column_name for column_name in self.drop_sensors if column_name in df.columns])

        sensor_cols = self._get_sensor_cols(df)

        # 2. Detect "bad" rows: rows inside a long NaN run in ANY sensor
        bad_mask = self._detect_long_gap_rows(df, sensor_cols)

        # 3. Assign block_id based on transitions between good/bad rows
        df['block_id'] = self._assign_block_ids(bad_mask)

        # 4. Drop bad rows (long gaps)
        df = df[~bad_mask].reset_index(drop=True)

        # 5. Track which rows had any NaN before imputation
        df['imputed'] = df[sensor_cols].isna().any(axis=1)

        # 6. Impute remaining short NaN runs within each block
        df[sensor_cols] = (
            df.groupby('block_id')[sensor_cols]
            .transform(self._impute_group)
        )

        return df

    def _get_sensor_cols(self, df: pd.DataFrame) -> List[str]:
        """Numeric columns excluding timestamp and metadata."""
        # block_id / imputed are this class's own output columns.
        exclude = {self.timestamp_col, 'block_id', 'imputed', *self.non_sensor_columns}
        return [
            column_name for column_name in df.columns
            if column_name not in exclude and pd.api.types.is_numeric_dtype(df[column_name])
        ]

    def _detect_long_gap_rows(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.Series:
        """
        For each sensor, find runs of NaN longer than the threshold.
        A row is "bad" if it falls in such a run for ANY sensor.
        """
        bad = pd.Series(False, index=df.index)
        for col in sensor_cols:
            is_nan = df[col].isna()
            # group consecutive identical values (run-length encoding)
            run_id = (is_nan != is_nan.shift()).cumsum()
            run_size = is_nan.groupby(run_id).transform('size')
            long_nan = is_nan & (run_size > self.gap_threshold_samples)
            bad = bad | long_nan
        return bad

    def _assign_block_ids(self, bad_mask: pd.Series) -> pd.Series:
        """
        Block id increments every time we transition from bad->good.
        Bad rows get the same id as the surrounding block but will be dropped.
        """
        # a new block starts when the previous row was bad and current is good
        starts_new_block = (~bad_mask) & (bad_mask.shift(fill_value=True))
        return starts_new_block.cumsum()

    def _impute_group(self, group: pd.DataFrame) -> pd.DataFrame:
        if self.imputation_method == 'ffill':
            return group.ffill().bfill()
        if self.imputation_method == 'bfill':
            return group.bfill().ffill()
        return group.interpolate(
            method=self.imputation_method,
            limit_direction='both',
            **self.imputation_kwargs,
        )