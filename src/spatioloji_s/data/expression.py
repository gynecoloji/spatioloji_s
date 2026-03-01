"""
expression.py - Efficient expression matrix with automatic sparse handling
"""

import numpy as np
import pandas as pd
from scipy import sparse


class ExpressionMatrix:
    """
    Efficient expression matrix with automatic sparse/dense handling.

    Features:
    - Automatic conversion to sparse when beneficial
    - Fast subsetting by cells and genes
    - Memory-efficient storage
    - Handles DataFrame, ndarray, or sparse matrix input
    """

    def __init__(
        self,
        data: np.ndarray | sparse.spmatrix | pd.DataFrame,
        cell_ids: pd.Index,
        gene_names: pd.Index,
        auto_sparse: bool = True,
        sparse_threshold: float = 0.5,
    ):
        """
        Initialize expression matrix.

        Parameters
        ----------
        data : np.ndarray, sparse matrix, or pd.DataFrame
            Expression data (cells × genes)
        cell_ids : pd.Index
            Cell identifiers
        gene_names : pd.Index
            Gene names
        auto_sparse : bool
            Automatically convert to sparse if beneficial
        sparse_threshold : float
            Sparsity threshold for conversion (0-1)
        """
        # HANDLE DATAFRAME INPUT
        if isinstance(data, pd.DataFrame):
            print("  → Converting DataFrame to array...")
            data = data.values  # Convert to numpy array

        # Validate dimensions
        if data.shape[0] != len(cell_ids):
            raise ValueError(f"Data rows ({data.shape[0]}) != cell_ids ({len(cell_ids)})")
        if data.shape[1] != len(gene_names):
            raise ValueError(f"Data cols ({data.shape[1]}) != gene_names ({len(gene_names)})")

        # Auto-convert to sparse if beneficial
        if auto_sparse and isinstance(data, np.ndarray):
            sparsity = 1 - np.count_nonzero(data) / data.size
            if sparsity > sparse_threshold:
                data = sparse.csr_matrix(data)
                print(f"  → Auto-converted to sparse matrix (sparsity: {sparsity:.1%})")

        self._data = data
        self._cell_ids = cell_ids
        self._gene_names = gene_names
        self._is_sparse = sparse.issparse(data)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (n_cells, n_genes)."""
        return self._data.shape

    @property
    def is_sparse(self) -> bool:
        """Check if using sparse representation."""
        return self._is_sparse

    @property
    def cell_ids(self) -> pd.Index:
        """Get cell IDs."""
        return self._cell_ids

    @property
    def gene_names(self) -> pd.Index:
        """Get gene names."""
        return self._gene_names

    def get_dense(self) -> np.ndarray:
        """Get dense representation."""
        if self._is_sparse:
            return self._data.toarray()
        return self._data

    def get_sparse(self) -> sparse.spmatrix:
        """Get sparse representation."""
        if not self._is_sparse:
            return sparse.csr_matrix(self._data)
        return self._data

    def subset_cells(self, indices: np.ndarray) -> "ExpressionMatrix":
        """
        Efficiently subset cells by integer indices.

        Parameters
        ----------
        indices : np.ndarray
            Integer indices of cells to keep

        Returns
        -------
        ExpressionMatrix
            New expression matrix with subset
        """
        if self._is_sparse:
            subset_data = self._data[indices, :]
        else:
            subset_data = self._data[indices, :]

        return ExpressionMatrix(
            data=subset_data,
            cell_ids=self._cell_ids[indices],
            gene_names=self._gene_names,
            auto_sparse=False,  # Already in correct format
        )

    def subset_genes(self, indices: np.ndarray) -> "ExpressionMatrix":
        """
        Efficiently subset genes by integer indices.

        Parameters
        ----------
        indices : np.ndarray
            Integer indices of genes to keep

        Returns
        -------
        ExpressionMatrix
            New expression matrix with subset
        """
        if self._is_sparse:
            subset_data = self._data[:, indices]
        else:
            subset_data = self._data[:, indices]

        return ExpressionMatrix(
            data=subset_data, cell_ids=self._cell_ids, gene_names=self._gene_names[indices], auto_sparse=False
        )

    def __getitem__(self, key):
        """Support array-like indexing."""
        return self._data[key]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame (for display/export only).

        Warning: This creates a dense representation in memory.
        """
        return pd.DataFrame(self.get_dense(), index=self._cell_ids, columns=self._gene_names)

    def memory_usage_mb(self) -> float:
        """
        Estimate memory usage in MB.

        FIXED: Handle both sparse and dense properly.
        """
        if self._is_sparse:
            # Sparse matrix
            total_bytes = self._data.data.nbytes + self._data.indices.nbytes + self._data.indptr.nbytes
        else:
            # Dense numpy array
            total_bytes = self._data.nbytes

        return total_bytes / (1024 * 1024)
