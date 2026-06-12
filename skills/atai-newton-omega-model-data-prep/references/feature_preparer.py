"""
FeaturePreparer: transforms an embeddings DataFrame into ready-to-use (X, y, metadata)
arrays for any downstream model (KNN, Isolation Forest, etc).

Always operates in "joint state" mode: embeddings from all sensors are concatenated
into a single vector per window. This means each row in X represents the joint
state of the system at a given window (a.k.a. "philosophy 2").
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeaturePreparer:
    """
    Prepares features for downstream classifiers/detectors.

    Args:
        normalize: 'l2', 'standardize', or None.
        reduce_dim: number of PCA components, or None to skip PCA.
        sensor_order: order of sensors when concatenating (None = alphabetical).
        read_index_column: column identifying the window (default 'read_index').
        sensor_column: column identifying the sensor (default 'sensor').
        embedding_column: column holding the embedding vector (default 'embedding').
        timestamp_column: optional timestamp column to carry into metadata
            (default 'timestamp'; skipped if absent).

    Example:
        >>> preparer = FeaturePreparer(normalize='l2', reduce_dim=50)
        >>> X, y, metadata = preparer.prepare(df_emb, label_column='state')
    """

    def __init__(
        self,
        normalize: Optional[str] = None,
        reduce_dim: Optional[int] = None,
        sensor_order: Optional[List[str]] = None,
        read_index_column: str = 'read_index',
        sensor_column: str = 'sensor',
        embedding_column: str = 'embedding',
        timestamp_column: str = 'timestamp',
    ):
        if normalize not in (None, 'l2', 'standardize'):
            raise ValueError(
                f"normalize must be None, 'l2' or 'standardize' (got '{normalize}')."
            )
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.sensor_order = sensor_order
        self.read_index_column = read_index_column
        self.sensor_column = sensor_column
        self.embedding_column = embedding_column
        self.timestamp_column = timestamp_column

    def prepare(
        self,
        df_emb: pd.DataFrame,
        label_column: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """
        Build (X, y, metadata) from an embeddings DataFrame.

        Args:
            df_emb: DataFrame produced by EmbeddingGenerator.generate().
            label_column: optional name of the label column to extract as y.
                If None, returns y=None (intended for unsupervised tasks).

        Returns:
            X: feature matrix of shape (N_windows, D).
                D = (num_sensors * embedding_dim) before PCA, or `reduce_dim` after PCA.
            y: array of shape (N_windows,) with labels, or None if label_column was None.
            metadata: DataFrame with read_index, label (if any), and timestamp (if available).
        """
        sensor_order = self.sensor_order or sorted(df_emb[self.sensor_column].unique())
        num_sensors = len(sensor_order)

        # --- Validate that every window has all sensors ---
        sensors_per_window = df_emb.groupby(self.read_index_column)[self.sensor_column].nunique()
        bad_windows = sensors_per_window[sensors_per_window != num_sensors].index.tolist()
        if bad_windows:
            raise ValueError(
                f"Some windows are missing sensors. Expected {num_sensors} sensors per window, "
                f"problematic read_indexes: {bad_windows[:5]}..."
            )

        # --- Pivot and concatenate embeddings ---
        pivot = df_emb.pivot(
            index=self.read_index_column, columns=self.sensor_column, values=self.embedding_column
        )
        pivot = pivot[sensor_order]  # enforce consistent column ordering

        X = np.stack([np.concatenate(row) for row in pivot.values])

        # --- Extract labels (if requested) ---
        y = None
        if label_column is not None:
            if label_column not in df_emb.columns:
                raise ValueError(f"label_column '{label_column}' not found in df_emb.")

            # Validate label consistency across sensors of the same window
            label_consistency = df_emb.groupby(self.read_index_column)[label_column].nunique()
            inconsistent = label_consistency[label_consistency > 1].index.tolist()
            if inconsistent:
                raise ValueError(
                    f"Inconsistent labels across sensors of the same window. "
                    f"Problematic read_indexes: {inconsistent[:5]}..."
                )

            y = (
                df_emb.groupby(self.read_index_column)[label_column]
                .first()
                .reindex(pivot.index)
                .values
            )

        # --- Apply normalization ---
        X = self._apply_normalization(X)

        # --- Apply PCA ---
        if self.reduce_dim is not None:
            X = self._apply_pca(X)

        # --- Build metadata DataFrame ---
        metadata = self._build_metadata(df_emb, pivot, label_column, y)

        return X, y, metadata

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _apply_normalization(self, X: np.ndarray) -> np.ndarray:
        if self.normalize == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # avoid division by zero
            return X / norms
        elif self.normalize == 'standardize':
            return StandardScaler().fit_transform(X)
        return X

    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        if self.reduce_dim >= X.shape[1]:
            raise ValueError(
                f"reduce_dim ({self.reduce_dim}) must be smaller than current D ({X.shape[1]})."
            )
        if self.reduce_dim >= X.shape[0]:
            raise ValueError(
                f"reduce_dim ({self.reduce_dim}) must be smaller than number of samples "
                f"({X.shape[0]})."
            )
        # 'full' SVD is more numerically stable than randomized for L2-normalized embeddings
        pca = PCA(n_components=self.reduce_dim, svd_solver='full', random_state=42)
        X_reduced = pca.fit_transform(X)
        print(f"PCA: cumulative explained variance = {pca.explained_variance_ratio_.sum():.2%}")
        return X_reduced

    def _build_metadata(
        self,
        df_emb: pd.DataFrame,
        pivot: pd.DataFrame,
        label_column: Optional[str],
        y: Optional[np.ndarray],
    ) -> pd.DataFrame:
        metadata_dict = {self.read_index_column: pivot.index.values}

        if label_column is not None and y is not None:
            metadata_dict[label_column] = y

        if self.timestamp_column in df_emb.columns:
            timestamps = (
                df_emb.groupby(self.read_index_column)[self.timestamp_column]
                .first()
                .reindex(pivot.index)
                .values
            )
            metadata_dict[self.timestamp_column] = timestamps

        return pd.DataFrame(metadata_dict)