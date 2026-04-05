"""
_accel.py - Acceleration bridge for CCC scoring.

Tries to import the compiled C++ extension (_scoring_cpp).
Falls back to pure-Python (NumPy) implementations when unavailable.

Usage
-----
>>> from spatioloji_s.ccc._accel import HAS_CPP_ACCEL, score_edges_batch, permutation_test_accel
"""

from __future__ import annotations

import numpy as np

# ── Try C++ import ───────────────────────────────────────────────────────────

HAS_CPP_ACCEL: bool = False
_cpp = None

try:
    from spatioloji_s.ccc import _scoring_cpp as _cpp  # type: ignore[attr-defined]

    HAS_CPP_ACCEL = True
except ImportError:
    pass


# ── Public API ───────────────────────────────────────────────────────────────


def score_edges_batch(
    expr_mat: np.ndarray,
    sender_idx: np.ndarray,
    receiver_idx: np.ndarray,
    weights: np.ndarray,
    ligand_cols: list[list[int]],
    receptor_cols: list[list[int]],
) -> np.ndarray:
    """Batch-score all LR pairs. Returns flat (n_pairs * n_edges,) array.

    Args:
        expr_mat: Expression matrix (n_cells, n_genes), float64 C-contiguous.
        sender_idx: Integer indices into expr_mat rows for senders (n_edges,).
        receiver_idx: Integer indices into expr_mat rows for receivers (n_edges,).
        weights: Edge weights (n_edges,), float64.
        ligand_cols: Per-pair list of column indices for ligand subunits.
        receptor_cols: Per-pair list of column indices for receptor subunits.

    Returns:
        Flat float64 array of shape (n_pairs * n_edges,).
    """
    if HAS_CPP_ACCEL:
        return _cpp.score_edges_batch(
            np.ascontiguousarray(expr_mat, dtype=np.float64),
            np.ascontiguousarray(sender_idx, dtype=np.intp),
            np.ascontiguousarray(receiver_idx, dtype=np.intp),
            np.ascontiguousarray(weights, dtype=np.float64),
            ligand_cols,
            receptor_cols,
        )

    return _score_edges_batch_py(expr_mat, sender_idx, receiver_idx, weights, ligand_cols, receptor_cols)


def permutation_test_accel(
    sender_idx: np.ndarray,
    receiver_idx: np.ndarray,
    type_array: np.ndarray,
    lr_int: np.ndarray,
    score_arr: np.ndarray,
    key_lr: np.ndarray,
    key_st: np.ndarray,
    key_rt: np.ndarray,
    key_valid: np.ndarray,
    obs_sums: np.ndarray,
    n_types: int,
    n_lrs: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Run permutation test. Returns int64 array of perm_counts (n_keys,).

    Args:
        sender_idx: Integer cell indices for sender of each edge (n_edges,).
        receiver_idx: Integer cell indices for receiver of each edge (n_edges,).
        type_array: Integer-encoded cell type labels (n_cells,).
        lr_int: Integer-encoded LR pair per edge (n_edges,).
        score_arr: Edge scores (n_edges,), float64.
        key_lr: LR index per result row (n_keys,).
        key_st: Sender type index per result row (n_keys,).
        key_rt: Receiver type index per result row (n_keys,).
        key_valid: Boolean mask for valid keys (n_keys,).
        obs_sums: Observed sums per key (n_keys,), float64.
        n_types: Number of unique cell types.
        n_lrs: Number of unique LR pairs.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        int64 array of shape (n_keys,) with permutation counts.
    """
    if HAS_CPP_ACCEL:
        return _cpp.permutation_test(
            np.ascontiguousarray(sender_idx, dtype=np.intp),
            np.ascontiguousarray(receiver_idx, dtype=np.intp),
            np.ascontiguousarray(type_array, dtype=np.intp),
            np.ascontiguousarray(lr_int, dtype=np.intp),
            np.ascontiguousarray(score_arr, dtype=np.float64),
            np.ascontiguousarray(key_lr, dtype=np.intp),
            np.ascontiguousarray(key_st, dtype=np.intp),
            np.ascontiguousarray(key_rt, dtype=np.intp),
            np.ascontiguousarray(key_valid, dtype=bool),
            np.ascontiguousarray(obs_sums, dtype=np.float64),
            n_types,
            n_lrs,
            n_permutations,
            seed,
        )

    return _permutation_test_py(
        sender_idx, receiver_idx, type_array, lr_int, score_arr,
        key_lr, key_st, key_rt, key_valid, obs_sums,
        n_types, n_lrs, n_permutations, seed,
    )


# ── Python fallbacks ─────────────────────────────────────────────────────────


def _score_edges_batch_py(
    expr_mat: np.ndarray,
    sender_idx: np.ndarray,
    receiver_idx: np.ndarray,
    weights: np.ndarray,
    ligand_cols: list[list[int]],
    receptor_cols: list[list[int]],
) -> np.ndarray:
    """Pure-Python (NumPy) fallback for score_edges_batch."""
    n_edges = len(sender_idx)
    n_pairs = len(ligand_cols)
    scores = np.empty(n_pairs * n_edges, dtype=np.float64)

    for p in range(n_pairs):
        lcols = ligand_cols[p]
        rcols = receptor_cols[p]
        offset = p * n_edges

        # Ligand values
        if lcols:
            l_vals = expr_mat[sender_idx][:, lcols].min(axis=1) if len(lcols) > 1 else expr_mat[sender_idx, lcols[0]]
        else:
            l_vals = np.zeros(n_edges)

        # Receptor values
        if rcols:
            r_vals = expr_mat[receiver_idx][:, rcols].min(axis=1) if len(rcols) > 1 else expr_mat[receiver_idx, rcols[0]]
        else:
            r_vals = np.zeros(n_edges)

        scores[offset:offset + n_edges] = np.sqrt(np.maximum(l_vals, 0.0) * np.maximum(r_vals, 0.0)) * weights

    return scores


def _permutation_test_py(
    sender_idx: np.ndarray,
    receiver_idx: np.ndarray,
    type_array: np.ndarray,
    lr_int: np.ndarray,
    score_arr: np.ndarray,
    key_lr: np.ndarray,
    key_st: np.ndarray,
    key_rt: np.ndarray,
    key_valid: np.ndarray,
    obs_sums: np.ndarray,
    n_types: int,
    n_lrs: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Pure-Python (NumPy) fallback for permutation_test using accumulator."""
    n_keys = len(key_lr)

    rng = np.random.RandomState(seed)
    perm_counts = np.zeros(n_keys, dtype=np.int64)

    # Accumulator: flat 3D [lr][sender_type][receiver_type]
    acc_stride_lr = n_types * n_types

    shuffled = type_array.copy()

    for _ in range(n_permutations):
        rng.shuffle(shuffled)

        # Accumulate using np.add.at for the 3D accumulator
        flat_idx = lr_int * acc_stride_lr + shuffled[sender_idx] * n_types + shuffled[receiver_idx]
        acc = np.zeros(n_lrs * acc_stride_lr, dtype=np.float64)
        np.add.at(acc, flat_idx, score_arr)

        # Compare each key to observed
        for k in range(n_keys):
            if not key_valid[k]:
                continue
            perm_sum = acc[key_lr[k] * acc_stride_lr + key_st[k] * n_types + key_rt[k]]
            if perm_sum >= obs_sums[k]:
                perm_counts[k] += 1

    return perm_counts
