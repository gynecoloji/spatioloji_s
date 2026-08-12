"""
normalization.py - Normalization methods for spatial transcriptomics data

Big-data design principles applied throughout
=============================================
  Sparse-native          normalize_total and log_transform operate directly on
                         sparse matrix structures (CSR) without materialising a
                         dense copy.  Zeros are never touched.
  Chunked computation    Dense operations are processed in cell_chunk_size ×
                         chunk_genes tiles so peak extra-RAM is bounded to
                         one tile rather than the full matrix.
  Memory-mapped output   scale() and normalize_pearson_residuals() accept an
                         ``out`` argument (file path or pre-allocated array /
                         memmap) so the dense result can live on disk.
  float32 dense layers   Dense output layers use float32 (4 bytes/element vs 8),
                         halving RAM without loss of biological precision.
  No unnecessary copies  In-place ufuncs (np.log1p(..., out=...)), view-based
                         slicing, and astype(..., copy=False) used throughout.

GPU acceleration
================
  ``normalize_total``, ``log_transform``, ``scale``,
  ``scale_by_batch_normalization``, ``normalize_pearson_residuals`` and
  ``normalize_standard_workflow`` accept a ``device`` parameter
  (``'auto' | 'cpu' | 'gpu'``).  When ``'auto'`` and RAPIDS (``cupy``,
  ``cupyx.scipy.sparse``) is importable, the heavy reductions and
  elementwise math run on GPU and the result is materialised back to
  numpy / scipy.sparse before being written to the spatioloji object.
  The GPU path is non-chunked (operates on the whole matrix), so very
  large datasets that exceed GPU RAM should use ``device='cpu'``.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse

from ._gpu import Device, _resolve_device, _to_cupy, _to_numpy, _warn_fallback

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_X(spatioloji_obj, layer: str | None):
    """Return the expression matrix without unnecessary copies.

    For *None* (main expression), returns the sparse matrix if the expression
    is already sparse, or the dense array if it is dense — never converts
    between formats.  For a named *layer*, returns whatever was stored.
    """
    if layer is None:
        if spatioloji_obj.expression.is_sparse:
            return spatioloji_obj.expression.get_sparse()
        return spatioloji_obj.expression.get_dense()
    return spatioloji_obj.get_layer(layer)


def _alloc_out(
    shape: tuple[int, int],
    dtype,
    out,
) -> np.ndarray:
    """Allocate or validate the output array.

    Parameters
    ----------
    shape : tuple[int, int]
        Required shape.
    dtype : numpy dtype
        Required element type.
    out : None | np.ndarray | str | Path
        * None  → allocate a regular in-memory array.
        * str / Path → create a memory-mapped file at that path (mode ``w+``).
        * ndarray / memmap → validate shape/dtype and return as-is.

    Returns
    -------
    np.ndarray
        Pre-allocated (possibly memory-mapped) output array.
    """
    if out is None:
        return np.empty(shape, dtype=dtype)
    if isinstance(out, (str, Path)):
        return np.memmap(str(out), dtype=dtype, mode="w+", shape=shape)
    if isinstance(out, np.ndarray):
        if out.shape != shape:
            raise ValueError(f"`out` shape {out.shape} does not match expected {shape}")
        return out
    raise TypeError(f"Unsupported type for `out`: {type(out)}")


def _n_workers(n_jobs: int) -> int:
    """Resolve *n_jobs* to a concrete positive thread count.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers.  ``1`` → serial.  ``-1`` → all cores.
        Negative values follow sklearn convention: ``os.cpu_count() + 1 + n_jobs``.

    Returns
    -------
    int
        Positive number of worker threads.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _normalize_total_gpu(X, target_sum, exclude_highly_expressed, max_fraction):
    """cuML/cupy path for normalize_total. Returns (X_norm, row_sums) on host."""
    import cupy as cp
    from cupyx.scipy import sparse as cusparse

    is_sparse = sparse.issparse(X)
    X_gpu = _to_cupy(X)

    if is_sparse:
        X_gpu = X_gpu.tocsr()
        X_gpu.eliminate_zeros()
        row_sums = cp.asarray(X_gpu.sum(axis=1)).ravel()
    else:
        row_sums = X_gpu.sum(axis=1)

    if exclude_highly_expressed:
        col_sums = cp.asarray(X_gpu.sum(axis=0)).ravel()
        gene_fractions = col_sums / cp.maximum(col_sums.sum(), 1.0)
        gene_mask = gene_fractions < max_fraction
        if int((~gene_mask).sum()) > 0:
            if is_sparse:
                row_sums = cp.asarray(X_gpu[:, gene_mask].sum(axis=1)).ravel()
            else:
                row_sums = X_gpu[:, gene_mask].sum(axis=1)

    scale_factors = cp.where(row_sums > 0, target_sum / row_sums, 0.0)

    if is_sparse:
        diag = cusparse.diags(scale_factors)
        X_norm_gpu = (diag @ X_gpu).tocsr()
        X_norm_gpu.data = X_norm_gpu.data.astype(cp.float32, copy=False)
    else:
        X_norm_gpu = (X_gpu * scale_factors[:, None]).astype(cp.float32, copy=False)

    return _to_numpy(X_norm_gpu), _to_numpy(row_sums)


def normalize_total(
    spatioloji_obj,
    target_sum: float = 1e4,
    layer: str | None = None,
    output_layer: str = "normalized_counts",
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    inplace: bool = False,
    copy: bool = False,
    cell_chunk_size: int = 50_000,
    n_jobs: int = 1,
    device: Device = "auto",
):
    """
    Normalize counts per cell (library-size normalization).

    Each cell is scaled so that its total counts equal *target_sum*.
    For sparse input the operation is performed entirely on the sparse
    structure — no dense matrix is ever created.  For dense input, cells are
    processed in *cell_chunk_size* batches to bound peak RAM.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data.
    target_sum : float, optional
        Target count sum per cell, by default 1e4.
    layer : str, optional
        Layer to normalize.  None uses the main expression matrix.
    output_layer : str, optional
        Name for the output layer, by default ``'normalized_counts'``.
    exclude_highly_expressed : bool, optional
        Exclude genes that make up more than *max_fraction* of total counts
        from the library-size denominator, by default False.
    max_fraction : float, optional
        Per-gene fraction threshold for exclusion, by default 0.05.
    inplace : bool, optional
        Add the output layer to the object in place, by default False.
    copy : bool, optional
        Return a deep copy with the output layer added, by default False.
    cell_chunk_size : int, optional
        Cells processed per chunk in the dense path, by default 50 000.
        Has no effect on the sparse path.
    n_jobs : int, optional
        Number of parallel worker threads for the dense path, by default 1.
        ``-1`` uses all available CPU cores.  Has no effect on the sparse path.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cupy/cupyx on GPU when
        RAPIDS is importable, otherwise the chunked CPU path.

    Returns
    -------
    spatioloji, sparse matrix, np.ndarray, or None
        Depends on *copy* / *inplace* flags.

    Examples
    --------
    >>> sj.processing.normalize_total(sp, target_sum=1e4, inplace=True)
    >>> sj.processing.normalize_total(sp, target_sum=1e4, device='gpu', inplace=True)
    """
    X = _get_X(spatioloji_obj, layer)
    n_cells, n_genes = X.shape

    print(f"\nNormalizing expression to target sum: {target_sum:,.0f}")

    backend = _resolve_device(device, "normalize_total")

    if backend == "gpu":
        try:
            X_norm, row_sums = _normalize_total_gpu(
                X, target_sum, exclude_highly_expressed, max_fraction,
            )
            print(f"  Counts per cell — mean: {row_sums.mean():.0f}, median: {np.median(row_sums):.0f}")
            print(f"  ✓ Normalized {n_cells:,} cells × {n_genes:,} genes (GPU)")
            if copy:
                import copy as copy_module

                sp_new = copy_module.deepcopy(spatioloji_obj)
                sp_new.add_layer(output_layer, X_norm, overwrite=True)
                return sp_new
            elif inplace:
                spatioloji_obj.add_layer(output_layer, X_norm, overwrite=True)
                return None
            return X_norm
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("normalize_total", exc)

    if sparse.issparse(X):
        # ── Sparse path (O(nnz), no dense materialization) ──────────────────
        X_csr = X.tocsr()
        X_csr.eliminate_zeros()

        row_sums = np.asarray(X_csr.sum(axis=1)).ravel()
        print(f"  Counts per cell — mean: {row_sums.mean():.0f}, median: {np.median(row_sums):.0f}")

        if exclude_highly_expressed:
            col_sums = np.asarray(X_csr.sum(axis=0)).ravel()
            gene_fractions = col_sums / max(float(col_sums.sum()), 1.0)
            gene_mask = gene_fractions < max_fraction
            n_excl = int((~gene_mask).sum())
            if n_excl:
                print(f"  Excluding {n_excl} highly expressed genes (>{max_fraction * 100:.1f}% of total)")
                row_sums = np.asarray(X_csr[:, gene_mask].sum(axis=1)).ravel()

        scale_factors = np.where(row_sums > 0, target_sum / row_sums, 0.0)

        # Row-scale without densifying: multiply broadcasts (n_cells, 1) across rows
        X_norm = X_csr.multiply(scale_factors[:, np.newaxis]).tocsr()
        X_norm.data = X_norm.data.astype(np.float32, copy=False)

    else:
        # ── Dense chunked path ───────────────────────────────────────────────
        # Pass 1 — compute row sums in chunks (O(n_cells) extra RAM)
        row_sums = np.empty(n_cells, dtype=np.float64)
        for s in range(0, n_cells, cell_chunk_size):
            e = min(s + cell_chunk_size, n_cells)
            row_sums[s:e] = X[s:e, :].sum(axis=1)

        print(f"  Counts per cell — mean: {row_sums.mean():.0f}, median: {np.median(row_sums):.0f}")

        if exclude_highly_expressed:
            col_sums = np.zeros(n_genes, dtype=np.float64)
            for s in range(0, n_cells, cell_chunk_size):
                e = min(s + cell_chunk_size, n_cells)
                col_sums += X[s:e, :].sum(axis=0)
            gene_fractions = col_sums / max(float(col_sums.sum()), 1.0)
            gene_mask = gene_fractions < max_fraction
            n_excl = int((~gene_mask).sum())
            if n_excl:
                print(f"  Excluding {n_excl} highly expressed genes (>{max_fraction * 100:.1f}% of total)")
                for s in range(0, n_cells, cell_chunk_size):
                    e = min(s + cell_chunk_size, n_cells)
                    row_sums[s:e] = X[s:e, :][:, gene_mask].sum(axis=1)

        scale_factors = np.where(row_sums > 0, target_sum / row_sums, 0.0).astype(np.float32)

        # Pass 2 — apply scale factors in chunks; output is float32
        X_norm = np.empty((n_cells, n_genes), dtype=np.float32)

        def _norm_chunk(s):
            e = min(s + cell_chunk_size, n_cells)
            chunk = X[s:e, :].astype(np.float32, copy=False)
            X_norm[s:e, :] = chunk * scale_factors[s:e, np.newaxis]

        starts = list(range(0, n_cells, cell_chunk_size))
        if n_jobs == 1 or len(starts) <= 1:
            for s in starts:
                _norm_chunk(s)
        else:
            with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
                list(ex.map(_norm_chunk, starts))

    print(f"  ✓ Normalized {n_cells:,} cells × {n_genes:,} genes")

    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_norm, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_norm, overwrite=True)
        return None
    else:
        return X_norm


def _log_transform_gpu(X, base, offset):
    """cupy path for log_transform. Returns numpy/scipy on host."""
    import cupy as cp

    use_log1p = base is None and offset == 1.0
    is_sparse = sparse.issparse(X)
    X_gpu = _to_cupy(X)

    if is_sparse and use_log1p:
        X_log_gpu = X_gpu.copy()
        # cupyx CSR exposes .data — log1p in place on stored nonzeros
        cp.log1p(X_log_gpu.data, out=X_log_gpu.data)
        X_log_gpu = X_log_gpu.tocsr()
    else:
        if is_sparse:
            X_gpu = X_gpu.toarray()
        X_log_gpu = X_gpu.astype(cp.float32, copy=True)
        if use_log1p:
            cp.log1p(X_log_gpu, out=X_log_gpu)
        elif base is None:
            cp.log(X_log_gpu + cp.float32(offset), out=X_log_gpu)
        else:
            cp.log(X_log_gpu + cp.float32(offset), out=X_log_gpu)
            X_log_gpu /= cp.float32(np.log(base))

    return _to_numpy(X_log_gpu)


def log_transform(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "log_normalized",
    base: float | None = None,
    offset: float = 1.0,
    inplace: bool = False,
    copy: bool = False,
    cell_chunk_size: int = 50_000,
    n_jobs: int = 1,
    device: Device = "auto",
):
    """
    Apply log transformation to expression data.

    For the default log1p case (``base=None``, ``offset=1.0``), the
    transformation is applied directly to the stored non-zero values of a
    sparse matrix — the zeros are untouched and the output is sparse.
    For all other parameter combinations, a chunked dense path is used.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data.
    layer : str, optional
        Layer to transform.  None uses the main expression matrix.
    output_layer : str, optional
        Name for the output layer, by default ``'log_normalized'``.
    base : float, optional
        Logarithm base.  None means natural log (uses ``np.log1p`` internally).
    offset : float, optional
        Offset added before taking the log, by default 1.0.
    inplace : bool, optional
        Add the output layer to the object in place, by default False.
    copy : bool, optional
        Return a deep copy with the output layer added, by default False.
    cell_chunk_size : int, optional
        Cells processed per chunk in the dense path, by default 50 000.
    n_jobs : int, optional
        Number of parallel worker threads for the dense path, by default 1.
        ``-1`` uses all available CPU cores.  Has no effect on the sparse log1p path.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cupy on GPU when RAPIDS
        is importable; sparse log1p stays sparse on the GPU path too.

    Returns
    -------
    spatioloji, sparse matrix, np.ndarray, or None
        Depends on *copy* / *inplace* flags.

    Examples
    --------
    >>> sj.processing.log_transform(sp, layer='normalized_counts', inplace=True)
    >>> sj.processing.log_transform(sp, layer='normalized_counts', device='gpu', inplace=True)
    """
    X = _get_X(spatioloji_obj, layer)
    n_cells, n_genes = X.shape

    use_log1p = base is None and offset == 1.0
    print(f"\nApplying log transformation (base={'e' if base is None else base})")

    backend = _resolve_device(device, "log_transform")
    if backend == "gpu":
        try:
            X_log = _log_transform_gpu(X, base, offset)
            print(f"  ✓ Transformed {n_cells:,} cells × {n_genes:,} genes (GPU)")
            if copy:
                import copy as copy_module

                sp_new = copy_module.deepcopy(spatioloji_obj)
                sp_new.add_layer(output_layer, X_log, overwrite=True)
                return sp_new
            elif inplace:
                spatioloji_obj.add_layer(output_layer, X_log, overwrite=True)
                return None
            return X_log
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("log_transform", exc)

    if sparse.issparse(X) and use_log1p:
        # ── Sparse path: log1p(0)=0, so sparsity structure is preserved ─────
        # Copy only the sparse arrays (data / indices / indptr), not the full dense shape
        X_log = X.copy()
        np.log1p(X_log.data, out=X_log.data)  # in-place on stored nonzeros only
        X_log = X_log.tocsr()

    else:
        # ── Dense chunked path ────────────────────────────────────────────────
        if sparse.issparse(X):
            # Non-log1p transform: zeros become log(offset) ≠ 0 → must densify
            X = X.toarray()
        _log_base = None if base is None else float(np.log(base))
        X_log = np.empty((n_cells, n_genes), dtype=np.float32)

        def _log_chunk(s):
            e = min(s + cell_chunk_size, n_cells)
            chunk = X[s:e, :].astype(np.float32, copy=False)
            if use_log1p:
                np.log1p(chunk, out=chunk)
            elif base is None:
                np.log(chunk + offset, out=chunk)
            else:
                np.log(chunk + offset, out=chunk)
                chunk /= _log_base  # change-of-base; no extra array
            X_log[s:e, :] = chunk

        starts = list(range(0, n_cells, cell_chunk_size))
        if n_jobs == 1 or len(starts) <= 1:
            for s in starts:
                _log_chunk(s)
        else:
            with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
                list(ex.map(_log_chunk, starts))

    print(f"  ✓ Transformed {n_cells:,} cells × {n_genes:,} genes")

    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_log, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_log, overwrite=True)
        return None
    else:
        return X_log


def _scale_gpu(X, method, zero_center, max_value):
    """cupy path for scale. Returns numpy/scipy on host."""
    import cupy as cp
    from cupyx.scipy import sparse as cusparse

    is_sparse = sparse.issparse(X)
    n_cells = X.shape[0]
    X_gpu = _to_cupy(X)

    # Sparse + zero_center=False preserves sparsity
    if is_sparse and not zero_center:
        X_csc = X_gpu.tocsc()
        X_csc.eliminate_zeros()
        n = float(n_cells)

        if method == "standard":
            # float64 accumulators, to match the CPU path — a float32 sum over
            # n_cells nonzeros drifts linearly in n_cells. Cast the matrix
            # rather than passing dtype= to sum(), which cupyx sparse does not
            # accept on every version.
            X_csc64 = X_csc.astype(cp.float64)
            col_sum = cp.asarray(X_csc64.sum(axis=0)).ravel()
            col_sum2 = cp.asarray(X_csc64.power(2).sum(axis=0)).ravel()
            del X_csc64
            gene_mean = col_sum / n
            gene_var = col_sum2 / n - gene_mean**2
            gene_std = cp.sqrt(cp.maximum(gene_var, 0.0)).astype(cp.float32)
            gene_std = cp.where(gene_std == 0, cp.float32(1.0), gene_std)
        else:  # robust — densify per gene-chunk on GPU
            gene_std = cp.empty(X.shape[1], dtype=cp.float32)
            chunk = 512
            for s in range(0, X.shape[1], chunk):
                e = min(s + chunk, X.shape[1])
                col_dense = X_csc[:, s:e].toarray()
                q75 = cp.percentile(col_dense, 75, axis=0)
                q25 = cp.percentile(col_dense, 25, axis=0)
                iqr = (q75 - q25).astype(cp.float32)
                iqr = cp.where(iqr == 0, cp.float32(1.0), iqr)
                gene_std[s:e] = iqr

        inv_scale = cusparse.diags(cp.float32(1.0) / gene_std)
        X_scaled_gpu = (X_csc @ inv_scale).tocsr()
        X_scaled_gpu.data = X_scaled_gpu.data.astype(cp.float32, copy=False)
        if max_value is not None:
            X_scaled_gpu.data = cp.clip(X_scaled_gpu.data, -max_value, max_value)
        return _to_numpy(X_scaled_gpu)

    # Dense path (also where sparse + zero_center=True must densify)
    if is_sparse:
        X_gpu = X_gpu.toarray()
    X_gpu = X_gpu.astype(cp.float32, copy=False)

    if method == "standard":
        # float64 accumulators, cast back to float32 below — same reason as the
        # CPU path. This is a single reduction over n_cells, so the fp64 penalty
        # on cards with reduced double-precision throughput is negligible next
        # to the full-matrix scaling that follows.
        gene_center = X_gpu.mean(axis=0, dtype=cp.float64)
        gene_scale = X_gpu.std(axis=0, ddof=1, dtype=cp.float64)
    else:  # robust
        gene_center = cp.median(X_gpu, axis=0)
        gene_scale = cp.percentile(X_gpu, 75, axis=0) - cp.percentile(X_gpu, 25, axis=0)

    gene_scale = cp.where(gene_scale == 0, cp.float32(1.0), gene_scale).astype(cp.float32)
    gene_center = gene_center.astype(cp.float32)

    if zero_center:
        X_scaled_gpu = (X_gpu - gene_center) / gene_scale
    else:
        X_scaled_gpu = X_gpu / gene_scale
    if max_value is not None:
        cp.clip(X_scaled_gpu, -max_value, max_value, out=X_scaled_gpu)

    return _to_numpy(X_scaled_gpu)


def scale(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "scaled",
    max_value: float | None = 10.0,
    method: Literal["standard", "robust"] = "standard",
    zero_center: bool = True,
    inplace: bool = False,
    copy: bool = False,
    chunk_genes: int = 500,
    cell_chunk_size: int = 50_000,
    out=None,
    n_jobs: int = 1,
    device: Device = "auto",
):
    """
    Scale expression data (z-score normalization per gene).

    Centers and scales each gene to unit variance.  Designed to avoid the
    peak RAM spike that sklearn's ``fit_transform`` produces:

    * Per-gene statistics are computed in a single pass (O(n_genes) RAM).
    * When ``zero_center=False`` and input is sparse, the output is sparse
      (column-wise division by std; zeros remain zero).
    * Otherwise the dense output is written gene-chunk by gene-chunk
      (``chunk_genes``) into a pre-allocated **float32** array, optionally
      backed by a memory-mapped file (``out``).

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data.
    layer : str, optional
        Layer to scale.  None uses the main expression matrix.
    output_layer : str, optional
        Name for the output layer, by default ``'scaled'``.
    max_value : float, optional
        Clip values to ``[-max_value, max_value]``, by default 10.0.
        Set to None to disable clipping.
    method : {'standard', 'robust'}, optional
        Scaling method: ``'standard'`` → (X − mean) / std;
        ``'robust'`` → (X − median) / IQR.
    zero_center : bool, optional
        Subtract per-gene mean (standard) / median (robust), by default True.
        Set to False to preserve sparsity when input is sparse.
    inplace : bool, optional
        Add the output layer to the object in place, by default False.
    copy : bool, optional
        Return a deep copy with the output layer added, by default False.
    chunk_genes : int, optional
        Genes processed per chunk, by default 500.
    cell_chunk_size : int, optional
        Cells processed per chunk when computing robust statistics,
        by default 50 000.  Has no effect for the standard method.
    out : None | str | Path | np.ndarray, optional
        Output destination.  Accepts a file path (creates a memory-mapped
        file), a pre-allocated ndarray / memmap, or None (in-memory).
        Ignored for the sparse output path (``zero_center=False``).
    n_jobs : int, optional
        Number of parallel worker threads for the dense path gene-chunk loops,
        by default 1.  ``-1`` uses all available CPU cores.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cupy on GPU when RAPIDS
        is importable.  The GPU path ignores ``out`` (no memory-mapped
        output on GPU) and ``chunk_genes`` (operates on the whole matrix).

    Returns
    -------
    spatioloji, sparse matrix, np.ndarray, or None
        Depends on *copy* / *inplace* flags.

    Examples
    --------
    >>> # In-memory, standard scaling
    >>> sj.processing.scale(sp, layer='log_normalized', inplace=True)
    >>>
    >>> # Force GPU
    >>> sj.processing.scale(sp, layer='log_normalized', device='gpu', inplace=True)
    """
    if method not in ("standard", "robust"):
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'robust'.")

    X = _get_X(spatioloji_obj, layer)
    n_cells, n_genes = X.shape

    print(f"\nScaling expression data (method: {method})")

    backend = _resolve_device(device, "scale")
    if backend == "gpu":
        try:
            X_scaled = _scale_gpu(X, method, zero_center, max_value)
            print(f"  ✓ Scaled {n_cells:,} cells × {n_genes:,} genes (GPU)")
            if copy:
                import copy as copy_module

                sp_new = copy_module.deepcopy(spatioloji_obj)
                sp_new.add_layer(output_layer, X_scaled, overwrite=True)
                return sp_new
            elif inplace:
                spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
                return None
            return X_scaled
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("scale", exc)

    # ── Sparse path: zero_center=False preserves sparsity ───────────────────
    if sparse.issparse(X) and not zero_center:
        X_csc = X.tocsc()
        X_csc.eliminate_zeros()

        if method == "standard":
            # Compute per-gene std from sparse column sums
            # E[X] and E[X^2] from nnz values only (zeros contribute 0 to both)
            n = float(n_cells)
            # float64 accumulators — see the note in the dense path below. A
            # float32 sum over n_cells nonzeros drifts linearly in n_cells, and
            # the one-pass E[x^2] - E[x]^2 form used here is more sensitive to
            # that drift than the two-pass form.
            col_sum = np.asarray(X_csc.sum(axis=0, dtype=np.float64)).ravel()
            col_sum2 = np.asarray(
                X_csc.power(2).sum(axis=0, dtype=np.float64)
            ).ravel()
            gene_mean = col_sum / n
            gene_var = col_sum2 / n - gene_mean**2
            gene_std = np.sqrt(np.maximum(gene_var, 0.0)).astype(np.float32)
            gene_std[gene_std == 0] = 1.0
        else:  # robust
            gene_std = np.empty(n_genes, dtype=np.float32)
            for s in range(0, n_genes, chunk_genes):
                e = min(s + chunk_genes, n_genes)
                col_dense = np.asarray(X_csc[:, s:e].todense())
                q75 = np.percentile(col_dense, 75, axis=0)
                q25 = np.percentile(col_dense, 25, axis=0)
                iqr = (q75 - q25).astype(np.float32)
                iqr[iqr == 0] = 1.0
                gene_std[s:e] = iqr

        # Column-scale the sparse matrix (scale each gene column)
        inv_scale = sparse.diags(1.0 / gene_std.astype(np.float64))
        X_scaled = (X_csc @ inv_scale).tocsr()
        X_scaled.data = X_scaled.data.astype(np.float32, copy=False)

        if max_value is not None:
            X_scaled.data = np.clip(X_scaled.data, -max_value, max_value)
            print(f"  Clipped stored values to [{-max_value}, {max_value}]")

        print(f"  ✓ Scaled {n_cells:,} cells × {n_genes:,} genes (sparse output)")

    else:
        # ── Dense chunked path ────────────────────────────────────────────────
        # Memory model:
        #   X          → input (already in RAM; may be sparse or dense)
        #   X_scaled   → float32 output, optionally memory-mapped
        #   per-gene statistics → shape (n_genes,), negligible
        #   one tile   → n_cells × chunk_genes × float32 per iteration
        if sparse.issparse(X):
            X = X.toarray()  # must densify: centering fills all zeros

        # Step 1 — per-gene statistics (no n_cells × n_genes temporaries)
        if method == "standard":
            # float64 ACCUMULATOR, float32 result. Reducing a float32
            # (n_cells, n_genes) C-contiguous array along axis 0 is a strided,
            # *sequential* sum — numpy's pairwise summation does not apply — so
            # the rounding error grows linearly in n_cells. Measured before this
            # fix: the per-gene std drifted from scanpy's by a median relative
            # 4.9e-5, and the resulting elementwise error grew 4.85x when the
            # cell count grew 5x (1.45e-3 at 20k cells, 7.03e-3 at 100k),
            # which is the signature of a naive running sum in float32.
            # scanpy accumulates in float64 for exactly this reason
            # (scanpy.preprocessing._utils._get_mean_var), and it applies the
            # same unbiased ddof=1 correction used here.
            gene_center = X.mean(axis=0, dtype=np.float64).astype(np.float32)
            gene_scale = X.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
            gene_scale[gene_scale == 0] = 1.0
        else:  # robust — chunk over genes to bound median sort memory
            gene_center = np.empty(n_genes, dtype=np.float32)
            gene_scale = np.empty(n_genes, dtype=np.float32)

            def _robust_stats_chunk(s):
                e = min(s + chunk_genes, n_genes)
                col_chunk = X[:, s:e]
                gene_center[s:e] = np.median(col_chunk, axis=0)
                q75 = np.percentile(col_chunk, 75, axis=0).astype(np.float32)
                q25 = np.percentile(col_chunk, 25, axis=0).astype(np.float32)
                iqr = q75 - q25
                iqr[iqr == 0] = 1.0
                gene_scale[s:e] = iqr

            gene_starts = list(range(0, n_genes, chunk_genes))
            if n_jobs == 1 or len(gene_starts) <= 1:
                for s in gene_starts:
                    _robust_stats_chunk(s)
            else:
                with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
                    list(ex.map(_robust_stats_chunk, gene_starts))

        # Step 2 — allocate output (memory-mapped if out is a path)
        X_scaled = _alloc_out((n_cells, n_genes), np.float32, out)

        # Step 3 — apply gene-chunk by gene-chunk
        def _scale_chunk(s):
            e = min(s + chunk_genes, n_genes)
            tile = X[:, s:e].astype(np.float32)  # explicit copy for in-place ops
            if zero_center:
                tile -= gene_center[s:e]
            tile /= gene_scale[s:e]
            if max_value is not None:
                np.clip(tile, -max_value, max_value, out=tile)
            X_scaled[:, s:e] = tile

        gene_starts = list(range(0, n_genes, chunk_genes))
        if n_jobs == 1 or len(gene_starts) <= 1:
            for s in gene_starts:
                _scale_chunk(s)
        else:
            with ThreadPoolExecutor(max_workers=_n_workers(n_jobs)) as ex:
                list(ex.map(_scale_chunk, gene_starts))

        if max_value is not None:
            print(f"  Clipped values to [{-max_value}, {max_value}]")
        print(f"  ✓ Scaled {n_cells:,} cells × {n_genes:,} genes")

    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_scaled, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
        return None
    else:
        return X_scaled


def _scale_by_batch_gpu(X, batch, method, zero_center, max_value):
    """cupy path for scale_by_batch_normalization. Returns numpy on host."""
    import cupy as cp

    if sparse.issparse(X):
        X = X.toarray()

    n_cells, n_genes = X.shape
    X_gpu = cp.asarray(X, dtype=cp.float32)
    X_scaled_gpu = cp.empty_like(X_gpu)
    batches = np.unique(batch)

    for batch_id in batches:
        idx_np = np.where(batch == batch_id)[0]
        idx = cp.asarray(idx_np, dtype=cp.int64)
        B = X_gpu[idx, :]
        n_b = int(idx.shape[0])

        if method == "standard":
            g_center = B.mean(axis=0)
            g_scale = B.std(axis=0, ddof=1 if n_b > 1 else 0)
        else:  # robust
            g_center = cp.median(B, axis=0)
            g_scale = cp.percentile(B, 75, axis=0) - cp.percentile(B, 25, axis=0)

        g_scale = cp.where(g_scale == 0, cp.float32(1.0), g_scale).astype(cp.float32)
        g_center = g_center.astype(cp.float32)

        if zero_center:
            tile = (B - g_center) / g_scale
        else:
            tile = B / g_scale
        if max_value is not None:
            cp.clip(tile, -max_value, max_value, out=tile)
        X_scaled_gpu[idx, :] = tile

    return _to_numpy(X_scaled_gpu), batches


def scale_by_batch_normalization(
    spatioloji_obj,
    batch_key: str,
    layer: str | None = "log_normalized",
    output_layer: str = "batch_scaled",
    method: Literal["standard", "robust"] = "standard",
    max_value: float | None = 10.0,
    zero_center: bool = True,
    inplace: bool = False,
    copy: bool = False,
    chunk_genes: int = 500,
    device: Device = "auto",
):
    """
    Scale expression separately within each batch.

    Useful when you want to standardize expression per gene while preserving
    batch-local distributions (e.g., FOV, slide, sample).  Each batch is
    processed with chunked gene-wise statistics to avoid large temporaries.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data.
    batch_key : str
        Column in ``cell_meta`` indicating batches (e.g. ``'fov'``, ``'sample_id'``).
    layer : str, optional
        Layer to scale, by default ``'log_normalized'``.
    output_layer : str, optional
        Name for the output layer, by default ``'batch_scaled'``.
    method : {'standard', 'robust'}, optional
        Scaling method within each batch.
    max_value : float, optional
        Clip values to ``[-max_value, max_value]``, by default 10.0.
    zero_center : bool, optional
        Center data before scaling, by default True.
    inplace : bool, optional
        Add the output layer to the object in place, by default False.
    copy : bool, optional
        Return a deep copy with the output layer added, by default False.
    chunk_genes : int, optional
        Genes processed per chunk per batch, by default 500.
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cupy on GPU when RAPIDS
        is importable.

    Returns
    -------
    spatioloji, np.ndarray, or None
        Depends on *copy* / *inplace* flags.
    """
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    X = _get_X(spatioloji_obj, layer)

    n_cells = X.shape[0]
    n_genes = X.shape[1]
    batch = spatioloji_obj.cell_meta[batch_key].astype(str).values

    print(f"\nBatch-wise scaling (batch_key='{batch_key}', method={method})")

    backend = _resolve_device(device, "scale_by_batch_normalization")
    if backend == "gpu":
        try:
            X_scaled, batches = _scale_by_batch_gpu(X, batch, method, zero_center, max_value)
            print(f"  Found {len(batches)} batches")
            if max_value is not None:
                print(f"  Clipped values to [{-max_value}, {max_value}]")
            print(f"  ✓ Scaled {n_cells:,} cells × {n_genes:,} genes across {len(batches)} batches (GPU)")
            if copy:
                import copy as copy_module

                sp_new = copy_module.deepcopy(spatioloji_obj)
                sp_new.add_layer(output_layer, X_scaled, overwrite=True)
                return sp_new
            elif inplace:
                spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
                return None
            return X_scaled
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("scale_by_batch_normalization", exc)

    if sparse.issparse(X):
        X = X.toarray()  # batch-wise centering; dense output unavoidable

    batches = np.unique(batch)
    print(f"  Found {len(batches)} batches")

    # Pre-allocate float32 output (no sklearn copy)
    X_scaled = np.empty((n_cells, n_genes), dtype=np.float32)

    for batch_id in batches:
        mask = batch == batch_id
        B = X[mask, :]  # view into the batch rows
        n_b = int(mask.sum())

        # Per-gene statistics for this batch
        if method == "standard":
            g_center = B.mean(axis=0).astype(np.float32)
            g_scale = B.std(axis=0, ddof=1 if n_b > 1 else 0).astype(np.float32)
            g_scale[g_scale == 0] = 1.0
        else:  # robust — chunk by gene
            g_center = np.empty(n_genes, dtype=np.float32)
            g_scale = np.empty(n_genes, dtype=np.float32)
            for s in range(0, n_genes, chunk_genes):
                e = min(s + chunk_genes, n_genes)
                col_chunk = B[:, s:e]
                g_center[s:e] = np.median(col_chunk, axis=0)
                q75 = np.percentile(col_chunk, 75, axis=0).astype(np.float32)
                q25 = np.percentile(col_chunk, 25, axis=0).astype(np.float32)
                iqr = q75 - q25
                iqr[iqr == 0] = 1.0
                g_scale[s:e] = iqr

        # Apply per gene-chunk into the pre-allocated output
        for s in range(0, n_genes, chunk_genes):
            e = min(s + chunk_genes, n_genes)
            tile = B[:, s:e].astype(np.float32)
            if zero_center:
                tile -= g_center[s:e]
            tile /= g_scale[s:e]
            if max_value is not None:
                np.clip(tile, -max_value, max_value, out=tile)
            X_scaled[np.ix_(np.where(mask)[0], np.arange(s, e))] = tile

    if max_value is not None:
        print(f"  Clipped values to [{-max_value}, {max_value}]")
    print(f"  ✓ Scaled {n_cells:,} cells × {n_genes:,} genes across {len(batches)} batches")

    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_scaled, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
        return None
    else:
        return X_scaled


def _pearson_residuals_gpu(X, theta, clip_val):
    """cupy path for normalize_pearson_residuals. Returns numpy on host."""
    import cupy as cp

    is_sparse = sparse.issparse(X)
    n_cells, n_genes = X.shape
    X_gpu = _to_cupy(X)

    if is_sparse:
        X_gpu = X_gpu.tocsr()
        X_gpu.eliminate_zeros()
        lib_size = cp.asarray(X_gpu.sum(axis=1)).ravel().astype(cp.float64)
        gene_sums = cp.asarray(X_gpu.sum(axis=0)).ravel().astype(cp.float64)
        X_dense_gpu = X_gpu.toarray().astype(cp.float32)
    else:
        X_dense_gpu = X_gpu.astype(cp.float32, copy=False)
        lib_size = X_dense_gpu.sum(axis=1).astype(cp.float64)
        gene_sums = X_dense_gpu.sum(axis=0).astype(cp.float64)

    mean_lib = float(lib_size.mean())
    gene_means = (gene_sums / n_cells).astype(cp.float32)
    lib_size_f32 = lib_size.astype(cp.float32)
    theta_f32 = cp.float32(theta)

    mu = lib_size_f32[:, None] * gene_means[None, :] / cp.float32(mean_lib)
    denom = cp.sqrt(mu + mu**2 / theta_f32)
    denom = cp.where(denom == 0, cp.float32(1.0), denom)
    residuals_gpu = (X_dense_gpu - mu) / denom
    cp.clip(residuals_gpu, -clip_val, clip_val, out=residuals_gpu)
    return _to_numpy(residuals_gpu)


def normalize_pearson_residuals(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "pearson_residuals",
    theta: float = 100.0,
    clip: float | None = None,
    inplace: bool = False,
    copy: bool = False,
    cell_chunk_size: int = 50_000,
    out=None,
    device: Device = "auto",
):
    """
    Normalize using Pearson residuals (analytic method).

    Normalizes counts under a negative-binomial model without fitting,
    following Lause *et al.* (2021) *Genome Biology*.

    Library sizes and per-gene means are derived from the sparse structure
    in a single pass (O(nnz)).  Residuals are computed and written
    cell-chunk by cell-chunk into a pre-allocated **float32** output
    (optionally memory-mapped) so peak extra-RAM equals one cell chunk.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with raw counts.
    layer : str, optional
        Layer to use.  None uses the main expression matrix.
    output_layer : str, optional
        Name for the output layer, by default ``'pearson_residuals'``.
    theta : float, optional
        Overdispersion parameter for the NB model, by default 100.0.
    clip : float, optional
        Clip residuals to ``[-clip, clip]``.  None uses ``sqrt(n_cells)``.
    inplace : bool, optional
        Add the output layer to the object in place, by default False.
    copy : bool, optional
        Return a deep copy with the output layer added, by default False.
    cell_chunk_size : int, optional
        Cells processed per chunk, by default 50 000.
    out : None | str | Path | np.ndarray, optional
        Output destination — file path (memory-mapped), pre-allocated array,
        or None (regular in-memory float32 array).
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cupy on GPU when RAPIDS
        is importable.  GPU path ignores ``out`` (no memory-mapped output).

    Returns
    -------
    spatioloji, np.ndarray, or None
        Depends on *copy* / *inplace* flags.

    References
    ----------
    Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for
    normalization of single-cell RNA-seq UMI data. *Genome Biol* 22, 258 (2021).

    Examples
    --------
    >>> sj.processing.normalize_pearson_residuals(sp, inplace=True)
    >>> sj.processing.normalize_pearson_residuals(sp, device='gpu', inplace=True)
    """
    X = _get_X(spatioloji_obj, layer)
    n_cells, n_genes = X.shape

    clip_val = clip if clip is not None else math.sqrt(n_cells)
    print(f"\nNormalizing using Pearson residuals (theta={theta})")

    backend = _resolve_device(device, "normalize_pearson_residuals")
    if backend == "gpu":
        try:
            residuals = _pearson_residuals_gpu(X, theta, clip_val)
            print(f"  Clipped residuals to [{-clip_val:.2f}, {clip_val:.2f}]")
            print(f"  ✓ Calculated residuals for {n_cells:,} cells × {n_genes:,} genes (GPU)")
            if copy:
                import copy as copy_module

                sp_new = copy_module.deepcopy(spatioloji_obj)
                sp_new.add_layer(output_layer, residuals, overwrite=True)
                return sp_new
            elif inplace:
                spatioloji_obj.add_layer(output_layer, residuals, overwrite=True)
                return None
            return residuals
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("normalize_pearson_residuals", exc)

    # ── Sparse-native statistics (O(nnz), no dense copy) ────────────────────
    if sparse.issparse(X):
        X_csr = X.tocsr()
        X_csr.eliminate_zeros()
        lib_size = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float64)  # (n_cells,)
        gene_sums = np.asarray(X_csr.sum(axis=0)).ravel().astype(np.float64)  # (n_genes,)
    else:
        # Dense path: compute sums in chunks to avoid a transient copy
        lib_size = np.empty(n_cells, dtype=np.float64)
        gene_sums = np.zeros(n_genes, dtype=np.float64)
        for s in range(0, n_cells, cell_chunk_size):
            e = min(s + cell_chunk_size, n_cells)
            chunk = X[s:e, :]
            lib_size[s:e] = chunk.sum(axis=1)
            gene_sums += chunk.sum(axis=0)

    mean_lib = float(lib_size.mean())
    gene_means = gene_sums / n_cells  # shape (n_genes,)

    # ── Allocate float32 output (optionally memory-mapped) ──────────────────
    residuals = _alloc_out((n_cells, n_genes), np.float32, out)

    # ── Compute residuals cell-chunk by cell-chunk ───────────────────────────
    gene_means_f32 = gene_means.astype(np.float32)
    theta_f32 = np.float32(theta)

    for s in range(0, n_cells, cell_chunk_size):
        e = min(s + cell_chunk_size, n_cells)

        # Dense slice for this batch (the only large transient allocation)
        if sparse.issparse(X):
            X_chunk = X_csr[s:e, :].toarray().astype(np.float32)
        else:
            X_chunk = X[s:e, :].astype(np.float32, copy=False)

        # mu_chunk: shape (n_chunk, n_genes) — broadcast lib_size over genes
        lib_chunk = lib_size[s:e].astype(np.float32)
        mu_chunk = lib_chunk[:, np.newaxis] * gene_means_f32[np.newaxis, :] / np.float32(mean_lib)

        # Pearson residual: (X - mu) / sqrt(mu + mu^2 / theta)
        denom = np.sqrt(mu_chunk + mu_chunk**2 / theta_f32)
        denom[denom == 0] = 1.0
        r_chunk = (X_chunk - mu_chunk) / denom
        np.clip(r_chunk, -clip_val, clip_val, out=r_chunk)

        residuals[s:e, :] = r_chunk

    print(f"  Clipped residuals to [{-clip_val:.2f}, {clip_val:.2f}]")
    print(f"  ✓ Calculated residuals for {n_cells:,} cells × {n_genes:,} genes")

    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, residuals, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, residuals, overwrite=True)
        return None
    else:
        return residuals


# Convenience function for standard workflow
def normalize_standard_workflow(
    spatioloji_obj,
    target_sum: float = 1e4,
    log_transform: bool = True,
    scale: bool = False,
    output_layer: str = "normalized",
    inplace: bool = True,
    cell_chunk_size: int = 50_000,
    device: Device = "auto",
):
    """
    Apply standard normalization workflow.

    Performs: library-size normalization → log1p → (optional) scaling.
    All steps propagate *cell_chunk_size* for chunked / big-data operation.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with raw counts.
    target_sum : float, optional
        Target sum for library-size normalization, by default 1e4.
    log_transform : bool, optional
        Apply log1p transformation, by default True.
    scale : bool, optional
        Apply z-score scaling, by default False.
    output_layer : str, optional
        Name for the final output layer, by default ``'normalized'``.
    inplace : bool, optional
        Modify the spatioloji object in place, by default True.
    cell_chunk_size : int, optional
        Cells processed per chunk at each step, by default 50 000.

    Returns
    -------
    None or np.ndarray
        None if inplace=True, otherwise the normalized matrix.

    Examples
    --------
    >>> sj.processing.normalize_standard_workflow(sp)
    >>> sj.processing.normalize_standard_workflow(sp, scale=True)
    """
    print("\n" + "=" * 70)
    print("Standard Normalization Workflow")
    print("=" * 70)

    print("\nStep 1: Library-size normalization")
    X_norm = normalize_total(
        spatioloji_obj,
        target_sum=target_sum,
        cell_chunk_size=cell_chunk_size,
        device=device,
    )

    if log_transform:
        print("\nStep 2: Log transformation")
        spatioloji_obj.add_layer("_temp_norm", X_norm, overwrite=True)
        X_norm = globals()["log_transform"](
            spatioloji_obj,
            layer="_temp_norm",
            cell_chunk_size=cell_chunk_size,
            device=device,
        )
        spatioloji_obj.remove_layer("_temp_norm")

    if scale:
        print("\nStep 3: Scaling")
        spatioloji_obj.add_layer("_temp_log", X_norm, overwrite=True)
        X_norm = globals()["scale"](spatioloji_obj, layer="_temp_log", device=device)
        spatioloji_obj.remove_layer("_temp_log")

    print("\n" + "=" * 70)
    print("✓ Normalization complete")
    print("=" * 70)

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_norm, overwrite=True)
        return None
    return X_norm
