"""
DataSplitter: splits features and labels into train/test, supporting multiple modes:

    - 'oot' (out-of-time): train on the past, test on the future. Realistic
      for time-series, prevents temporal leakage.
    - 'random': random split (does not respect temporal order). Useful as a
      sanity check or when temporal order is irrelevant.

New strategies can be added by extending VALID_MODES and the dispatcher in `split()`.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class DataSplitter:
    """
    Splits (X, y, metadata) into train/test partitions.

    Args:
        mode: 'oot' (out-of-time) or 'random'.
        test_size: fraction of samples for the test set (e.g. 0.3 = 30% test).
        timestamp_column: column in metadata used to order samples when mode='oot'.
        random_state: seed used when mode='random'.

    Example:
        >>> splitter = DataSplitter(mode='oot', test_size=0.3)
        >>> X_train, X_test, y_train, y_test, meta_train, meta_test = splitter.split(X, y, metadata)
    """

    VALID_MODES = ('oot', 'random')

    def __init__(
        self,
        mode: str = 'oot',
        test_size: float = 0.3,
        timestamp_column: str = 'timestamp',
        random_state: int = 42,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode '{mode}' is invalid. Use one of: {self.VALID_MODES}")
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

        self.mode = mode
        self.test_size = test_size
        self.timestamp_column = timestamp_column
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        metadata: pd.DataFrame,
    ) -> Tuple[
        np.ndarray, np.ndarray,
        Optional[np.ndarray], Optional[np.ndarray],
        pd.DataFrame, pd.DataFrame,
    ]:
        """
        Split X, y, and metadata into train/test partitions.

        Args:
            X: feature matrix (N, D).
            y: labels (N,) or None for unsupervised.
            metadata: DataFrame with N rows. Must contain `timestamp_column` if mode='oot'.

        Returns:
            X_train, X_test, y_train, y_test, meta_train, meta_test.
            If y is None, y_train and y_test are also None.
        """
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test

        if n_train == 0 or n_test == 0:
            raise ValueError(
                f"test_size={self.test_size} produced empty train or test set "
                f"(n_train={n_train}, n_test={n_test})."
            )

        # Get train and test indices based on mode
        if self.mode == 'oot':
            train_idx, test_idx = self._oot_indices(metadata, n_train)
        else:  # random
            train_idx, test_idx = self._random_indices(n_samples, n_train)

        # Slice each component
        X_train, X_test = X[train_idx], X[test_idx]

        if y is not None:
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            y_train, y_test = None, None

        meta_train = metadata.iloc[train_idx].reset_index(drop=True)
        meta_test = metadata.iloc[test_idx].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, meta_train, meta_test

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _oot_indices(self, metadata: pd.DataFrame, n_train: int):
        if self.timestamp_column not in metadata.columns:
            raise ValueError(
                f"OOT mode requires column '{self.timestamp_column}' in metadata. "
                f"Available columns: {metadata.columns.tolist()}"
            )

        # Convert to datetime to ensure correct ordering even if it's a string
        timestamps = pd.to_datetime(metadata[self.timestamp_column])
        order = timestamps.argsort().values

        train_idx = order[:n_train]
        test_idx = order[n_train:]
        return train_idx, test_idx

    def _random_indices(self, n_samples: int, n_train: int):
        rng = np.random.default_rng(self.random_state)
        permutation = rng.permutation(n_samples)
        return permutation[:n_train], permutation[n_train:]