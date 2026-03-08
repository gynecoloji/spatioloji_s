"""
layer1.py - Layer 1: Discovery via Bivariate Moran's I and Spatial Lag Regression.

Identifies which LR pairs show significant spatial coupling between
cell types, and ranks them by effect size. All methods use polygon
geometry for spatial weights — no centroid-distance assumptions.

Weight matrix convention
------------------------
W[i, j] = weight from sender cell i toward receiver cell j
  → rows = senders, cols = receivers
  → W @ ones = row sums (each sender's total outgoing weight)
  → W.T used for spatial lag: (W.T @ L)[j] = weighted avg of L in j's neighborhood

Outputs (layer1_results dict)
------------------------------
'significant_pairs' : pd.DataFrame  — filtered ranked LR pairs
'W_juxtacrine'      : csr_matrix    — prebuilt weight matrix (reused in Layer 2)
'W_secreted'        : csr_matrix
'W_ecm'             : csr_matrix
'type_masks'        : dict[str, np.ndarray]  — boolean masks per cell type
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.polygon.graph import PolygonSpatialGraph


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import issparse

from .database import LRPair

# ── Weight matrix builder ─────────────────────────────────────────────────────


def build_weight_matrices(
    sp: spatioloji,
    contact_frac_df: pd.DataFrame,
    free_boundary_ser: pd.Series,
    secreted_graph: PolygonSpatialGraph,
    ecm_graph: PolygonSpatialGraph,
    coord_type: str = "global",
    min_weight: float = 1e-3,
) -> dict[str, sparse.csr_matrix]:
    """
    Build all three polygon-geometry weight matrices (juxtacrine, secreted, ecm).

    Each matrix W has shape (n_cells, n_cells):
      W[i, j] = weight from sender i toward receiver j
      Row-normalized so each sender's weights sum to 1.

    Graph sources
    -------------
    juxtacrine : contact_frac_df (polygon-touching pairs only)
                 w[i→j] = fraction of sender i's membrane touching j
    secreted   : secreted_graph (buffer graph, larger radius)
                 w[i→j] = free_boundary(j) × exp(-dist / σ_s)
                 Receiver exposure × distance decay.
    ecm        : ecm_graph (buffer graph, largest radius)
                 w[i→j] = (1 - solidity(j)) × exp(-dist / σ_e)
                 Morphologically irregular receivers capture more ECM signal.

    σ for each type is estimated from the median edge distance of the
    corresponding graph — data-driven, no manual tuning required.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object. Must have morph_solidity in sp.cell_meta.
    contact_frac_df : pd.DataFrame
        Output of boundaries.contact_fraction().
        Columns: cell_a, cell_b, fraction_a, fraction_b.
    free_boundary_ser : pd.Series
        Output of boundaries.free_boundary_fraction(). Indexed by cell_id.
    secreted_graph : PolygonSpatialGraph
        Buffer graph for secreted signals (build_buffer_graph with secreted_radius).
    ecm_graph : PolygonSpatialGraph
        Buffer graph for ECM signals (build_buffer_graph with ecm_radius).
    coord_type : str
        'global' or 'local'.

    Returns
    -------
    dict[str, csr_matrix]
        Keys: 'juxtacrine', 'secreted', 'ecm'. Each row-normalized.
    """
    cell_ids = np.array(sp.cell_index)
    n = len(cell_ids)
    id2idx = {cid: k for k, cid in enumerate(cell_ids)}

    # ── Retrieve morph_solidity ───────────────────────────────────────────────
    if "morph_solidity" not in sp.cell_meta.columns:
        raise ValueError("morph_solidity not found in sp.cell_meta. Run compute_morphology(sp, store=True) first.")
    solidity = pd.Series(sp.cell_meta["morph_solidity"].values, index=sp.cell_index)

    # ── Accumulators ─────────────────────────────────────────────────────────
    rows_j, cols_j, vals_j = [], [], []  # juxtacrine
    rows_s, cols_s, vals_s = [], [], []  # secreted
    rows_e, cols_e, vals_e = [], [], []  # ecm

    # ── Juxtacrine: from contact_frac_df ─────────────────────────────────────
    # σ estimated from median centroid distance of contacting pairs
    coords_arr = sp.get_spatial_coords(coord_type=coord_type)
    coords_df = pd.DataFrame(coords_arr, index=cell_ids, columns=["x", "y"])

    contact_dists = []
    for _, row in contact_frac_df.iterrows():
        ca, cb = row["cell_a"], row["cell_b"]
        if ca not in coords_df.index or cb not in coords_df.index:
            continue
        xi, yi = coords_df.loc[ca, ["x", "y"]]
        xj, yj = coords_df.loc[cb, ["x", "y"]]
        contact_dists.append(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
    sigma_j = float(np.median(contact_dists)) if contact_dists else 1.0
    sigma_j = max(sigma_j, 1e-6)
    print(f"[Layer1] σ_juxtacrine = {sigma_j:.2f}")

    for _, row in contact_frac_df.iterrows():
        ca, cb = row["cell_a"], row["cell_b"]
        ia = id2idx.get(ca)
        ib = id2idx.get(cb)
        if ia is None or ib is None:
            continue
        fa = float(row["fraction_a"])
        fb = float(row["fraction_b"])
        rows_j.extend([ia, ib])
        cols_j.extend([ib, ia])
        vals_j.extend([fa, fb])

    # ── Secreted: from secreted_graph (vectorized) ───────────────────────────
    secreted_edges = secreted_graph.to_edge_list()
    if len(secreted_edges) > 0:
        sigma_s = float(np.median(secreted_edges["distance"].values))
        sigma_s = max(sigma_s, 1e-6)
    else:
        sigma_s = sigma_j
    print(f"[Layer1] σ_secreted  = {sigma_s:.2f}  ({len(secreted_edges)} edges)")

    if len(secreted_edges) > 0:
        ca_arr = secreted_edges["cell_a"].values
        cb_arr = secreted_edges["cell_b"].values
        ia_arr = np.array([id2idx.get(c, -1) for c in ca_arr])
        ib_arr = np.array([id2idx.get(c, -1) for c in cb_arr])
        valid = (ia_arr >= 0) & (ib_arr >= 0)
        ia_arr, ib_arr = ia_arr[valid], ib_arr[valid]
        d_arr = secreted_edges["distance"].values[valid]
        decay = np.exp(-d_arr / sigma_s)
        fb_a = free_boundary_ser.reindex(ca_arr[valid]).fillna(1.0).values
        fb_b = free_boundary_ser.reindex(cb_arr[valid]).fillna(1.0).values
        # keep only edges with meaningful weight (sparsify)
        keep = decay > min_weight
        ia_arr, ib_arr = ia_arr[keep], ib_arr[keep]
        decay, fb_a, fb_b = decay[keep], fb_a[keep], fb_b[keep]
        rows_s = np.concatenate([ia_arr, ib_arr])
        cols_s = np.concatenate([ib_arr, ia_arr])
        vals_s = np.concatenate([fb_b * decay, fb_a * decay])
        print(f"  Secreted: {keep.sum()}/{len(valid)} edges kept (min_weight={min_weight})")

    # ── ECM: from ecm_graph (vectorized) ─────────────────────────────────────
    ecm_edges = ecm_graph.to_edge_list()
    if len(ecm_edges) > 0:
        sigma_e = float(np.median(ecm_edges["distance"].values))
        sigma_e = max(sigma_e, 1e-6)
    else:
        sigma_e = sigma_j
    print(f"[Layer1] σ_ecm       = {sigma_e:.2f}  ({len(ecm_edges)} edges)")

    if len(ecm_edges) > 0:
        ca_arr = ecm_edges["cell_a"].values
        cb_arr = ecm_edges["cell_b"].values
        ia_arr = np.array([id2idx.get(c, -1) for c in ca_arr])
        ib_arr = np.array([id2idx.get(c, -1) for c in cb_arr])
        valid = (ia_arr >= 0) & (ib_arr >= 0)
        ia_arr, ib_arr = ia_arr[valid], ib_arr[valid]
        d_arr = ecm_edges["distance"].values[valid]
        decay = np.exp(-d_arr / sigma_e)
        sol_a = solidity.reindex(ca_arr[valid]).fillna(1.0).values
        sol_b = solidity.reindex(cb_arr[valid]).fillna(1.0).values
        keep = decay > min_weight
        ia_arr, ib_arr = ia_arr[keep], ib_arr[keep]
        decay, sol_a, sol_b = decay[keep], sol_a[keep], sol_b[keep]
        rows_e = np.concatenate([ia_arr, ib_arr])
        cols_e = np.concatenate([ib_arr, ia_arr])
        vals_e = np.concatenate([(1.0 - sol_b) * decay, (1.0 - sol_a) * decay])
        print(f"  ECM:      {keep.sum()}/{len(valid)} edges kept (min_weight={min_weight})")

    # ── Build sparse matrices ─────────────────────────────────────────────────
    def _build(rows, cols, vals) -> sparse.csr_matrix:
        if not len(rows):
            return sparse.csr_matrix((n, n), dtype=np.float32)
        W = sparse.csr_matrix((np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(n, n))
        return _row_normalize(W)

    W = {
        "juxtacrine": _build(rows_j, cols_j, vals_j),
        "secreted": _build(rows_s, cols_s, vals_s),
        "ecm": _build(rows_e, cols_e, vals_e),
    }

    for name, mat in W.items():
        nnz = mat.nnz
        print(f"  W_{name:10s}: {n}×{n}, {nnz} nonzero ({nnz / n:.1f} avg neighbors per sender)")

    return W


def _row_normalize(W: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Row-normalize a sparse matrix so each row sums to 1.
    Rows with all-zero weights are left as zero (isolated senders).
    """
    row_sums = np.array(W.sum(axis=1)).flatten()
    # Avoid divide-by-zero for isolated cells
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums, format="csr")
    return (D_inv @ W).astype(np.float32)


def _get_expr_vector(
    expr: np.ndarray,
    gene2idx: dict[str, int],
    genes: list[str],
) -> np.ndarray:
    """
    Get expression vector for one gene or a complex (mean of subunits).

    Parameters
    ----------
    expr     : (n_cells, n_genes) dense float32 array
    gene2idx : gene name → column index
    genes    : list of gene symbols (length 1 for single, >1 for complex)

    Returns
    -------
    np.ndarray shape (n_cells,)
        Single gene: raw column.
        Complex:     mean across subunits (handles dropout gracefully).
    """
    cols = [expr[:, gene2idx[g]] for g in genes if g in gene2idx]
    if not cols:
        return np.zeros(expr.shape[0], dtype=np.float32)
    if len(cols) == 1:
        return cols[0].astype(np.float32)
    return np.mean(np.stack(cols, axis=1), axis=1).astype(np.float32)


# ── Bivariate Moran's I ───────────────────────────────────────────────────────


def _bivariate_moran(
    z_L: np.ndarray,
    z_R: np.ndarray,
    W: sparse.csr_matrix,
) -> float:
    """
    Compute bivariate Moran's I for one (L, R, W) combination.

    I = (n / S₀) × (z_L @ W @ z_R) / (z_L @ z_L)

    Parameters
    ----------
    z_L : (n,) centered ligand expression   (mean already subtracted)
    z_R : (n,) centered receptor expression
    W   : (n, n) row-normalized weight matrix, sender rows → receiver cols

    Returns
    -------
    float
        I_bivar value. Returns 0.0 if denominator is near zero.
    """
    n = len(z_L)
    S0 = W.sum()  # scalar: Σ_ij w_ij

    if S0 < 1e-10:
        return 0.0

    num = float(z_L @ W @ z_R)  # Σ_ij w_ij * z_Li * z_Rj
    denom = float(z_L @ z_L)  # Σ_i z²_Li

    if abs(denom) < 1e-10:
        return 0.0

    return (n / S0) * (num / denom)


def _process_one_pair(
    pair: LRPair,
    pair_idx: int,
    expr: np.ndarray,
    gene2idx: dict[str, int],
    W_dict: dict[str, sparse.csr_matrix],
    type_masks: dict[str, np.ndarray],
    n: int,
    n_permutations: int,
    seed: int,
) -> list[dict]:
    """
    Compute bivariate Moran's I for one LR pair across all type pairs.
    Designed to run in a joblib worker — no shared state, owns its own RNG.
    COO arrays extracted once per pair and reused across all type pairs.
    """
    # Each worker gets its own RNG seeded by pair index → fully reproducible
    rng = np.random.default_rng(seed + pair_idx)
    W = W_dict[pair.lr_type]
    L_raw = _get_expr_vector(expr, gene2idx, pair.ligand_genes)
    R_raw = _get_expr_vector(expr, gene2idx, pair.receptor_genes)

    # Extract COO arrays once per LR pair, reused across all type-pair combos
    W_coo = W.tocoo()
    ia_coo = W_coo.row.astype(np.intp)
    ib_coo = W_coo.col.astype(np.intp)
    w_data = W_coo.data.astype(np.float64)

    records: list[dict] = []

    for sender_type, s_mask in type_masks.items():
        for receiver_type, r_mask in type_masks.items():
            if s_mask.sum() < 5 or r_mask.sum() < 5:
                continue

            W_sr = _mask_weight_matrix(W, s_mask, r_mask)
            if W_sr.nnz == 0:
                continue

            # Observed I_AB
            z_L = np.where(s_mask, L_raw - L_raw[s_mask].mean(), 0.0)
            z_R = np.where(r_mask, R_raw - R_raw[r_mask].mean(), 0.0)
            I_AB = _bivariate_moran(z_L, z_R, W_sr)

            # Observed I_BA (directionality)
            W_rs = _mask_weight_matrix(W.T, r_mask, s_mask)
            z_L_ba = np.where(r_mask, R_raw - R_raw[r_mask].mean(), 0.0)
            z_R_ba = np.where(s_mask, L_raw - L_raw[s_mask].mean(), 0.0)
            I_BA = _bivariate_moran(z_L_ba, z_R_ba, W_rs)

            # Permutation null — pass precomputed COO to avoid re-extraction
            null_dist = _permutation_moran(
                L_raw,
                R_raw,
                W,
                W_sr,
                s_mask,
                r_mask,
                n_permutations,
                rng,
                ia_coo=ia_coo,
                ib_coo=ib_coo,
                w_data=w_data,
            )

            p_moran = max(float((null_dist >= I_AB).mean()), 1.0 / n_permutations)

            records.append(
                {
                    "lr_name": pair.lr_name,
                    "ligand": pair.ligand,
                    "receptor": pair.receptor,
                    "lr_type": pair.lr_type,
                    "pathway": pair.pathway,
                    "sender_type": sender_type,
                    "receiver_type": receiver_type,
                    "I_bivar": float(I_AB),
                    "I_bivar_BA": float(I_BA),
                    "direction_score": float(I_AB - I_BA),
                    "p_moran": p_moran,
                    "n_sender": int(s_mask.sum()),
                    "n_receiver": int(r_mask.sum()),
                    "n_contacts": int(W_sr.nnz),
                }
            )

    return records


def compute_bivariate_moran(
    sp: spatioloji,
    lr_pairs: list[LRPair],
    W_dict: dict[str, sparse.csr_matrix],
    cell_type_col: str = "cell_type",
    layer: str | None = "log_normalized",
    n_permutations: int = 1000,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute bivariate Moran's I for all LR pairs.

    For each LR pair, runs per cell type pair analyses:
      I(A→B), I(B→A), direction_score, permutation p-value.

    Parallelized over LR pairs via joblib (n_jobs=-1 = all cores).
    Each worker owns its own RNG seeded by seed + pair_idx.

    Parameters
    ----------
    sp : spatioloji
    lr_pairs : list[LRPair]
    W_dict : dict[str, csr_matrix]
    cell_type_col : str
    layer : str | None
    n_permutations : int
    seed : int
    n_jobs : int
        Number of parallel workers. -1 = all cores. Default 1.
    """
    # ── Extract shared arrays once before spawning workers ────────────────────
    if layer is not None and layer in sp.layers:
        expr = sp.layers[layer]
    else:
        expr = sp.expression.to_dense()
    if issparse(expr):
        expr = expr.toarray()
    expr = expr.astype(np.float32)

    gene2idx = {g: i for i, g in enumerate(sp.gene_index)}
    n = len(sp.cell_index)
    cell_types = np.array(sp.cell_meta[cell_type_col].values, dtype=str)
    type_masks = {t: (cell_types == t) for t in sorted(set(cell_types))}

    print(f"\n[Layer1 Moran] {len(lr_pairs)} LR pairs × {len(type_masks)} cell types | n_jobs={n_jobs}")
    print(f"  Permutations: {n_permutations}  |  n_cells: {n}")

    # ── Parallel dispatch — one job per LR pair ───────────────────────────────
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process_one_pair)(
            pair,
            pair_idx,
            expr,
            gene2idx,
            W_dict,
            type_masks,
            n,
            n_permutations,
            seed,
        )
        for pair_idx, pair in enumerate(lr_pairs)
    )

    # ── Flatten list-of-lists ─────────────────────────────────────────────────
    records = [rec for pair_records in results for rec in pair_records]

    if not records:
        print("  ⚠ No valid type pairs found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("I_bivar", ascending=False).reset_index(drop=True)

    n_sig = (df["p_moran"] < 0.05).sum()
    print(f"\n[Layer1 Moran] Done. {n_sig}/{len(df)} type-pair-LR combinations p < 0.05")

    return df


# ── Private helpers for Moran ─────────────────────────────────────────────────


def _mask_weight_matrix(
    W: sparse.csr_matrix,
    s_mask: np.ndarray,
    r_mask: np.ndarray,
) -> sparse.csr_matrix:
    """
    Zero out W rows not in sender mask and cols not in receiver mask.

    Returns a new CSR matrix with the same shape as W but only
    nonzero entries where row ∈ sender_mask AND col ∈ receiver_mask.

    This does NOT re-normalize — the original row weights are kept,
    so S₀ = sum(W_sr) reflects the actual contact density between
    these two cell types.
    """
    # Diagonal selection matrices
    D_s = sparse.diags(s_mask.astype(np.float32), format="csr")
    D_r = sparse.diags(r_mask.astype(np.float32), format="csr")
    return (D_s @ W @ D_r).astype(np.float32)


def _permutation_moran(
    L_raw: np.ndarray,
    R_raw: np.ndarray,
    W_full: sparse.csr_matrix,
    W_sr: sparse.csr_matrix,
    s_mask: np.ndarray,
    r_mask: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    ia_coo: np.ndarray | None = None,
    ib_coo: np.ndarray | None = None,
    w_data: np.ndarray | None = None,
) -> np.ndarray:
    """
    Permutation null for bivariate Moran's I.

    Optimized: avoids rebuilding W_perm as a sparse matrix each iteration.
    Uses the identity:
      z_L @ W_perm @ z_R  =  (z_L * s_bool) @ W_full @ (z_R * r_bool)
    so W_full is reused unchanged; only the z vectors are masked per permutation.
    S0 is computed via boolean indexing on preextracted COO arrays.

    Parameters
    ----------
    ia_coo, ib_coo, w_data : optional precomputed COO arrays from W_full.
        Pass these when calling from _process_one_pair to avoid re-extracting
        them for every type-pair combination within the same LR pair.
    """
    n = len(L_raw)
    n_s = int(s_mask.sum())
    n_r = int(r_mask.sum())

    # Extract COO arrays once if not provided by caller
    if ia_coo is None:
        W_coo = W_full.tocoo()
        ia_coo = W_coo.row.astype(np.intp)
        ib_coo = W_coo.col.astype(np.intp)
        w_data = W_coo.data.astype(np.float64)

    nulls = np.zeros(n_permutations, dtype=np.float64)
    s_bool = np.zeros(n, dtype=bool)  # reused each iteration
    r_bool = np.zeros(n, dtype=bool)

    for k in range(n_permutations):
        # Randomly assign n_s senders and n_r receivers
        perm = rng.permutation(n)
        s_bool[:] = False
        r_bool[:] = False
        s_bool[perm[:n_s]] = True
        r_bool[perm[n_s : n_s + n_r]] = True

        # S0: sum of W_full entries where row∈senders AND col∈receivers
        # No sparse matrix construction — just boolean index into COO arrays
        edge_active = s_bool[ia_coo] & r_bool[ib_coo]
        S0 = w_data[edge_active].sum()
        if S0 < 1e-10:
            continue

        # Masked z vectors — equivalent to D_s @ z_L, D_r @ z_R without matrix ops
        z_L = np.where(s_bool, L_raw - L_raw[s_bool].mean(), 0.0)
        z_R = np.where(r_bool, R_raw - R_raw[r_bool].mean(), 0.0)

        # W_full unchanged — masking absorbed into z vectors
        num = float(z_L @ W_full @ z_R)
        denom = float(z_L @ z_L)
        if abs(denom) < 1e-10:
            continue

        nulls[k] = (n / S0) * (num / denom)

    return nulls


# ── Spatial Lag Regression ────────────────────────────────────────────────────


def compute_spatial_lag(
    sp: spatioloji,
    lr_pairs: list[LRPair],
    W_dict: dict[str, sparse.csr_matrix],
    contact_graph: PolygonSpatialGraph,
    cell_type_col: str = "cell_type",
    layer: str | None = "log_normalized",
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    Spatial lag regression for all LR pairs.

    For each (LR pair, sender_type, receiver_type):
      Fits R_j = ρ₁ × WL_j + β × controls + ε   (Model A: L→R)
      Fits L_i = ρ₂ × WR_i + β × controls + ε   (Model B: R→L)
      where WL_j = (W.T @ L)[j] = lagged ligand at receiver j
            WR_i = (W   @ R)[i] = lagged receptor at sender i

    Estimation by OLS (numpy lstsq) — fast, no extra dependencies.
    Standard errors from residual variance.
    FDR correction (Benjamini-Hochberg) applied across all results.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object (already subset to one FOV).
        Requires in sp.cell_meta:
          'total_counts'         (from QC)
          'free_boundary_fraction' (from boundaries.free_boundary_fraction)
          'morph_circularity'    (from compute_morphology)
    lr_pairs : list[LRPair]
        Output of filter_to_expressed().
    W_dict : dict[str, csr_matrix]
        Output of build_weight_matrices().
    contact_graph : PolygonSpatialGraph
        Contact graph (for computing local degree per cell).
    cell_type_col : str
        Column in sp.cell_meta with cell type labels.
    layer : str | None
        Expression layer. None → raw sp.expression.
    min_cells : int
        Minimum cells in sender or receiver type to attempt regression.
        Default 10 (need enough for stable OLS).

    Returns
    -------
    pd.DataFrame
        One row per (lr_pair, sender_type, receiver_type).
        Columns: lr_name, ligand, receptor, lr_type, pathway,
                 sender_type, receiver_type,
                 rho_LR, rho_SE, rho_z, p_lag, p_lag_fdr,
                 rho_RL, dir_score, R2,
                 sig_type (which W gave higher rho — for Layer 2 auto-config)

    Examples
    --------
    >>> lag_df = compute_spatial_lag(
    ...     sp_fov, lr_expressed, W_dict, contact_graph,
    ...     cell_type_col='cell_type',
    ... )
    """
    # ── Get expression matrix ─────────────────────────────────────────────────
    if layer is not None and layer in sp.layers:
        expr = sp.layers[layer]
    else:
        expr = sp.expression.to_dense()

    if issparse(expr):
        expr = expr.toarray()
    expr = expr.astype(np.float32)

    gene2idx = {g: i for i, g in enumerate(sp.gene_index)}
    cell_ids = np.array(sp.cell_index)
    n = len(cell_ids)

    # ── Cell type labels ──────────────────────────────────────────────────────
    if cell_type_col not in sp.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in sp.cell_meta.")
    cell_types = np.array(sp.cell_meta[cell_type_col].values, dtype=str)
    type_labels = sorted(set(cell_types))
    type_masks: dict[str, np.ndarray] = {t: (cell_types == t) for t in type_labels}

    # ── Build control matrix (n_cells × 4) ───────────────────────────────────
    controls = _build_controls(sp, contact_graph, n)

    print(f"\n[Layer1 SpatialLag] {len(lr_pairs)} LR pairs × {len(type_labels)} cell types")
    print(f"  Controls: {list(controls.columns)}")

    records: list[dict] = []

    for pair_idx, pair in enumerate(lr_pairs):
        # Expression vectors for this LR pair
        L_all = _get_expr_vector(expr, gene2idx, pair.ligand_genes)
        R_all = _get_expr_vector(expr, gene2idx, pair.receptor_genes)

        W = W_dict[pair.lr_type]  # (n, n) row-normalized sender→receiver

        # Spatially lagged ligand and receptor (full n vectors)
        # WL[j] = Σ_i W[i,j] × L_i = weighted avg of L in j's neighborhood
        # WR[i] = Σ_j W[i,j] × R_j = weighted avg of R in i's neighborhood
        WL = np.array((W.T @ L_all), dtype=np.float64).flatten()
        WR = np.array((W @ R_all), dtype=np.float64).flatten()

        for sender_type in type_labels:
            for receiver_type in type_labels:
                s_mask = type_masks[sender_type]
                r_mask = type_masks[receiver_type]

                if s_mask.sum() < min_cells or r_mask.sum() < min_cells:
                    continue

                # ── Model A: L → R ────────────────────────────────────────────
                # Response: receptor at receiver cells
                # Treatment: lagged ligand at receiver cells (from senders)
                y_A = R_all[r_mask].astype(np.float64)
                x_A = WL[r_mask]  # spatially lagged ligand

                res_A = _ols_fit(y_A, x_A, controls.values[r_mask])
                if res_A is None:
                    continue

                rho_LR, rho_se, rho_z, p_lag, R2 = res_A

                # ── Model B: R → L (directionality check) ────────────────────
                # Response: ligand at sender cells
                # Treatment: lagged receptor at sender cells (from receivers)
                y_B = L_all[s_mask].astype(np.float64)
                x_B = WR[s_mask]  # spatially lagged receptor

                res_B = _ols_fit(y_B, x_B, controls.values[s_mask])
                rho_RL = res_B[0] if res_B is not None else 0.0

                dir_score = float(rho_LR - rho_RL)

                records.append(
                    {
                        "lr_name": pair.lr_name,
                        "ligand": pair.ligand,
                        "receptor": pair.receptor,
                        "lr_type": pair.lr_type,
                        "pathway": pair.pathway,
                        "sender_type": sender_type,
                        "receiver_type": receiver_type,
                        "rho_LR": float(rho_LR),
                        "rho_SE": float(rho_se),
                        "rho_z": float(rho_z),
                        "p_lag": float(p_lag),
                        "p_lag_fdr": np.nan,  # filled after BH
                        "rho_RL": float(rho_RL),
                        "dir_score": dir_score,
                        "R2": float(R2),
                        "n_sender": int(s_mask.sum()),
                        "n_receiver": int(r_mask.sum()),
                    }
                )

        if (pair_idx + 1) % 10 == 0 or pair_idx == len(lr_pairs) - 1:
            print(f"  [{pair_idx + 1}/{len(lr_pairs)}] {pair.lr_name} done")

    if not records:
        print("  ⚠ No valid results.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── FDR correction (BH) across all rows ──────────────────────────────────
    df["p_lag_fdr"] = _bh_fdr(df["p_lag"].values)

    # ── Annotate sig_type: which W type gave the highest rho ─────────────────
    # Used by Layer 2 to auto-select cost matrix per LR pair
    df = _annotate_sig_type(df, sp, lr_pairs, expr, gene2idx, W_dict, type_masks, controls)

    df = df.sort_values("rho_LR", ascending=False).reset_index(drop=True)

    n_sig = (df["p_lag_fdr"] < 0.05).sum()
    print(f"\n[Layer1 SpatialLag] Done. {n_sig}/{len(df)} combinations FDR < 0.05")

    return df


# ── Private helpers for Spatial Lag ──────────────────────────────────────────


def _build_controls(
    sp: spatioloji,
    contact_graph: PolygonSpatialGraph,
    n: int,
) -> pd.DataFrame:
    """
    Build control variable matrix from existing sp data.

    All controls are standardized (zero mean, unit variance)
    so that OLS coefficients are on a comparable scale.

    Controls
    --------
    log1p_counts    : log1p(total_counts) — library size
    local_degree    : number of polygon neighbors — density
    circularity     : morph_circularity — cell shape
    free_boundary   : free_boundary_fraction — membrane exposure

    Missing columns are filled with zeros (standardized mean)
    so regression still runs even if some QC steps were skipped.
    """
    ctrl = pd.DataFrame(index=range(n))

    # log1p total counts (library size confounder)
    if "total_counts" in sp.cell_meta.columns:
        ctrl["log1p_counts"] = np.log1p(sp.cell_meta["total_counts"].values.astype(np.float64))
    else:
        ctrl["log1p_counts"] = 0.0

    # Local degree from contact graph (density confounder)
    degree = contact_graph.get_degree()  # np.ndarray (n,), aligned to graph.cell_index
    # Align to sp.cell_index order
    graph_idx = {cid: k for k, cid in enumerate(contact_graph.cell_index)}
    sp_cell_ids = np.array(sp.cell_index)
    degree_aligned = np.array(
        [degree[graph_idx[cid]] if cid in graph_idx else 0 for cid in sp_cell_ids], dtype=np.float64
    )
    ctrl["local_degree"] = degree_aligned

    # Cell morphology (shape confounder)
    if "morph_circularity" in sp.cell_meta.columns:
        ctrl["circularity"] = sp.cell_meta["morph_circularity"].values.astype(np.float64)
    else:
        ctrl["circularity"] = 0.0

    # Membrane exposure (also in W but has direct effect on receptor accessibility)
    if "free_boundary_fraction" in sp.cell_meta.columns:
        ctrl["free_boundary"] = sp.cell_meta["free_boundary_fraction"].values.astype(np.float64)
    else:
        ctrl["free_boundary"] = 0.0

    # Standardize each control: zero mean, unit variance
    for col in ctrl.columns:
        v = ctrl[col].values
        std = v.std()
        if std > 1e-10:
            ctrl[col] = (v - v.mean()) / std
        else:
            ctrl[col] = 0.0

    return ctrl


def _ols_fit(
    y: np.ndarray,
    x_lag: np.ndarray,
    controls: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    """
    OLS regression: y ~ ρ × x_lag + β × controls + intercept.

    Returns (rho, rho_se, rho_z, p_value, R2) or None if rank-deficient.

    Implementation
    --------------
    Uses numpy.linalg.lstsq — no external dependencies.
    ρ is the first coefficient (x_lag column).
    SE computed from residual variance: SE(ρ) = sqrt(σ² × (X'X)⁻¹[0,0])
    p-value from two-sided z-test (large-sample approximation).
    R² = 1 − SS_res / SS_tot.
    """
    n = len(y)
    if n < 5:
        return None

    # Design matrix: [x_lag | controls | intercept]
    intercept = np.ones(n, dtype=np.float64)
    X = np.column_stack([x_lag, controls, intercept])  # (n, n_params)

    if X.shape[1] >= n:
        return None  # underdetermined

    # OLS via lstsq
    try:
        coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < X.shape[1]:
        return None  # rank-deficient — skip

    rho = float(coeffs[0])
    y_hat = X @ coeffs
    resid = y - y_hat

    # Residual variance σ² = SS_res / (n - p)
    n_params = X.shape[1]
    ss_res = float(resid @ resid)
    sigma2 = ss_res / max(n - n_params, 1)

    # SE(ρ) from diagonal of (X'X)⁻¹ × σ²
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        rho_se = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
    except np.linalg.LinAlgError:
        rho_se = np.nan

    # z-statistic and two-sided p-value (large-sample)
    if rho_se > 1e-10:
        rho_z = rho / rho_se
        from scipy.stats import norm as _norm

        p_val = float(2.0 * _norm.sf(abs(rho_z)))
    else:
        rho_z = 0.0
        p_val = 1.0

    # R²
    ss_tot = float(((y - y.mean()) ** 2).sum())
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    return rho, rho_se, rho_z, p_val, R2


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (may contain NaN — NaNs get FDR = NaN).

    Returns
    -------
    np.ndarray
        FDR-adjusted p-values, same length as input.
    """
    n = len(p_values)
    fdr = np.full(n, np.nan)
    valid = ~np.isnan(p_values)
    p_valid = p_values[valid]

    if len(p_valid) == 0:
        return fdr

    # Sort ascending
    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p_valid) + 1)

    # BH adjustment: p_adj[i] = p[i] × n / rank[i]
    p_adj = p_valid[order] * len(p_valid) / ranks[order]

    # Enforce monotonicity (cumulative minimum from right)
    p_adj = np.minimum.accumulate(p_adj[::-1])[::-1]
    p_adj = np.clip(p_adj, 0.0, 1.0)

    # Unsort back
    p_adj_unsorted = np.empty_like(p_adj)
    p_adj_unsorted[order] = p_adj

    fdr[valid] = p_adj_unsorted
    return fdr


def _annotate_sig_type(
    df: pd.DataFrame,
    sp: spatioloji,
    lr_pairs: list[LRPair],
    expr: np.ndarray,
    gene2idx: dict[str, int],
    W_dict: dict[str, sparse.csr_matrix],
    type_masks: dict[str, np.ndarray],
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (lr_pair, sender_type, receiver_type), determine which
    W type (juxtacrine / secreted / ecm) gives the highest ρ_LR.

    This is stored as 'sig_type' and consumed by Layer 2 to
    auto-select the correct cost matrix — no manual configuration.

    Strategy:
      Re-run OLS with each W type and pick the one with max ρ_LR.
      Only re-runs for rows where pair.lr_type was set by the database
      (e.g. 'secreted') — we also check the other types to see if a
      different geometry gives a better fit in THIS specific tissue.
    """
    df = df.copy()
    df["sig_type"] = df["lr_type"]  # default: use the database annotation

    # Build lr_name → LRPair lookup
    pair_lookup = {p.lr_name: p for p in lr_pairs}

    for idx, row in df.iterrows():
        pair = pair_lookup.get(row["lr_name"])
        if pair is None:
            continue

        s_mask = type_masks.get(row["sender_type"])
        r_mask = type_masks.get(row["receiver_type"])
        if s_mask is None or r_mask is None:
            continue
        if r_mask.sum() < 10:
            continue

        L_all = _get_expr_vector(expr, gene2idx, pair.ligand_genes)
        R_all = _get_expr_vector(expr, gene2idx, pair.receptor_genes)
        y_A = R_all[r_mask].astype(np.float64)

        best_rho = row["rho_LR"]
        best_type = row["lr_type"]

        for wtype, W in W_dict.items():
            if wtype == row["lr_type"]:
                continue  # already computed
            WL = np.array((W.T @ L_all), dtype=np.float64).flatten()
            x_A = WL[r_mask]
            res = _ols_fit(y_A, x_A, controls.values[r_mask])
            if res is None:
                continue
            rho_candidate = res[0]
            if rho_candidate > best_rho:
                best_rho = rho_candidate
                best_type = wtype

        df.at[idx, "sig_type"] = best_type

    return df


# ── Section 4: Combine + Filter + Rank + Output layer1_results ───────────────


def run_layer1(
    sp: spatioloji,
    lr_pairs: list[LRPair],
    contact_graph: PolygonSpatialGraph,
    contact_frac_df: pd.DataFrame,
    free_boundary_ser: pd.Series,
    secreted_graph: PolygonSpatialGraph,
    ecm_graph: PolygonSpatialGraph,
    cell_type_col: str = "cell_type",
    layer: str | None = "log_normalized",
    n_permutations: int = 1000,
    alpha_moran: float = 0.05,
    alpha_lag: float = 0.05,
    min_rho: float = 0.05,
    seed: int = 42,
    n_jobs: int = 1,
    min_weight: float = 1e-3,
) -> dict:
    """
    Run full Layer 1 pipeline and return layer1_results dict.

    Executes in order:
      1. build_weight_matrices()    — polygon W for all three types
      2. compute_bivariate_moran()  — spatial pattern detection
      3. compute_spatial_lag()      — effect size estimation
      4. join + filter + rank       — produce significant_pairs table
      5. package layer1_results     — ready for Layer 2 consumption

    Parameters
    ----------
    sp : spatioloji
        spatioloji object (already subset to one FOV).
        Required in sp.cell_meta:
          cell_type_col              (cell type labels)
          'total_counts'             (from QC)
          'free_boundary_fraction'   (from boundaries.free_boundary_fraction)
          'morph_circularity'        (from compute_morphology)
          'morph_solidity'           (from compute_morphology)
    lr_pairs : list[LRPair]
        Output of filter_to_expressed().
    contact_graph : PolygonSpatialGraph
        Contact graph for this FOV.
    contact_frac_df : pd.DataFrame
        Output of boundaries.contact_fraction() — columns:
        cell_a, cell_b, fraction_a, fraction_b.
    free_boundary_ser : pd.Series
        Output of boundaries.free_boundary_fraction() — indexed by cell_id.
    cell_type_col : str
        Column in sp.cell_meta with cell type labels.
    layer : str | None
        Expression layer. None → raw expression.
    n_permutations : int
        Permutations for Moran significance. Default 1000.
    alpha_moran : float
        Moran p-value threshold. Default 0.05.
    alpha_lag : float
        Spatial lag FDR threshold. Default 0.05.
    min_rho : float
        Minimum ρ_LR effect size. Default 0.05.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'significant_pairs' : pd.DataFrame
            Ranked significant LR pairs — primary input for Layer 2.
        'all_moran'         : pd.DataFrame
            Full Moran results (before filtering) — for diagnostics.
        'all_lag'           : pd.DataFrame
            Full spatial lag results (before filtering) — for diagnostics.
        'W_juxtacrine'      : csr_matrix  — reused by Layer 2
        'W_secreted'        : csr_matrix  — reused by Layer 2
        'W_ecm'             : csr_matrix  — reused by Layer 2
        'type_masks'        : dict[str, np.ndarray]  — reused by Layer 2
        'expr'              : np.ndarray  — (n_cells × n_genes) dense
        'gene2idx'          : dict[str, int]

    Examples
    --------
    >>> from spatioloji_s.ccc.layer1 import run_layer1
    >>> layer1_results = run_layer1(
    ...     sp_fov, lr_expressed, contact_graph,
    ...     contact_frac_df, free_boundary_ser,
    ...     cell_type_col='cell_type',
    ...     n_permutations=1000,
    ... )
    >>> sig = layer1_results['significant_pairs']
    >>> print(sig[['lr_name', 'sender_type', 'receiver_type',
    ...             'I_bivar', 'rho_LR', 'layer1_score']].head(10))
    """
    print("\n" + "=" * 60)
    print("LAYER 1: Discovery")
    print("=" * 60)

    # ── Step 1: Build weight matrices ─────────────────────────────────────────
    print("\n[Step 1/4] Building polygon weight matrices...")
    W_dict = build_weight_matrices(
        sp,
        contact_frac_df,
        free_boundary_ser,
        secreted_graph,
        ecm_graph,
        min_weight=min_weight,
    )

    # ── Step 2: Bivariate Moran ───────────────────────────────────────────────
    print("\n[Step 2/4] Computing Bivariate Moran's I...")
    moran_df = compute_bivariate_moran(
        sp,
        lr_pairs,
        W_dict,
        cell_type_col=cell_type_col,
        layer=layer,
        n_permutations=n_permutations,
        seed=seed,
        n_jobs=n_jobs,
    )

    # ── Step 3: Spatial Lag Regression ───────────────────────────────────────
    print("\n[Step 3/4] Computing Spatial Lag Regression...")
    lag_df = compute_spatial_lag(
        sp,
        lr_pairs,
        W_dict,
        contact_graph,
        cell_type_col=cell_type_col,
        layer=layer,
    )

    # ── Step 4: Join + Filter + Rank ─────────────────────────────────────────
    print("\n[Step 4/4] Joining results and filtering...")

    if moran_df.empty or lag_df.empty:
        print("  ⚠ No results to combine.")
        return _empty_layer1_results(W_dict, sp, layer)

    # Join on (lr_name, sender_type, receiver_type) — the natural key
    join_keys = ["lr_name", "sender_type", "receiver_type"]
    combined = pd.merge(
        moran_df,
        lag_df,
        on=join_keys + ["ligand", "receptor", "lr_type", "pathway"],
        how="inner",
        suffixes=("_moran", "_lag"),
    )

    if combined.empty:
        print("  ⚠ No overlapping results after join.")
        return _empty_layer1_results(W_dict, sp, layer)

    # ── Joint significance filter ─────────────────────────────────────────────
    sig_mask = (
        (combined["p_moran"] < alpha_moran)
        & (combined["p_lag_fdr"] < alpha_lag)
        & (combined["rho_LR"] > min_rho)
        & (combined["I_bivar"] > 0.0)  # positive coupling only
    )
    significant = combined[sig_mask].copy()

    # ── Compute layer1_score for ranking ─────────────────────────────────────
    # Product of both effect sizes — pairs with high pattern AND high effect
    # get the highest priority in Layer 2
    significant["layer1_score"] = significant["I_bivar"].abs() * significant["rho_LR"].abs()

    # ── Sort by layer1_score descending ──────────────────────────────────────
    significant = significant.sort_values("layer1_score", ascending=False).reset_index(drop=True)

    # ── Clean up column set for Layer 2 consumption ───────────────────────────
    keep_cols = [
        "lr_name",
        "ligand",
        "receptor",
        "lr_type",
        "pathway",
        "sender_type",
        "receiver_type",
        # Moran
        "I_bivar",
        "I_bivar_BA",
        "direction_score",
        "p_moran",
        # Spatial Lag
        "rho_LR",
        "rho_SE",
        "rho_z",
        "p_lag",
        "p_lag_fdr",
        "rho_RL",
        "dir_score",
        "R2",
        # Auto-config for Layer 2
        "sig_type",
        # Metadata
        "n_sender",
        "n_receiver",
        "n_contacts",
        # Rank score
        "layer1_score",
    ]
    # Only keep columns that exist (n_contacts may not be in lag_df)
    keep_cols = [c for c in keep_cols if c in significant.columns]
    significant = significant[keep_cols]

    # ── Prepare shared objects for Layer 2 ────────────────────────────────────
    # Get expression once here — Layer 2 reuses without re-extracting
    if layer is not None and layer in sp.layers:
        expr = sp.layers[layer]
    else:
        expr = sp.expression.to_dense()
    if issparse(expr):
        expr = expr.toarray()
    expr = expr.astype(np.float32)

    gene2idx = {g: i for i, g in enumerate(sp.gene_index)}

    cell_types = np.array(sp.cell_meta[cell_type_col].values, dtype=str)
    type_masks = {t: (cell_types == t) for t in sorted(set(cell_types))}

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_layer1_summary(combined, significant, alpha_moran, alpha_lag, min_rho)

    return {
        # Primary output — consumed by Layer 2
        "significant_pairs": significant,
        # Full results — for diagnostics and visualization
        "all_moran": moran_df,
        "all_lag": lag_df,
        "all_combined": combined,
        # Pre-built matrices — reused by Layer 2 (no rebuild)
        "W_juxtacrine": W_dict["juxtacrine"],
        "W_secreted": W_dict["secreted"],
        "W_ecm": W_dict["ecm"],
        # Cell type masks — reused by Layer 2
        "type_masks": type_masks,
        # Expression — reused by Layer 2
        "expr": expr,
        "gene2idx": gene2idx,
    }


# ── Diagnostic helpers ────────────────────────────────────────────────────────


def summarize_layer1(layer1_results: dict, top_n: int = 20) -> None:
    """
    Print a readable summary of Layer 1 results.

    Shows top significant pairs ranked by layer1_score,
    and a breakdown by signaling type and cell type pair.

    Parameters
    ----------
    layer1_results : dict
        Output of run_layer1().
    top_n : int
        Number of top pairs to display.
    """
    sig = layer1_results["significant_pairs"]

    if sig.empty:
        print("No significant LR pairs found.")
        return

    print("\n" + "=" * 60)
    print(f"LAYER 1 SUMMARY  —  {len(sig)} significant combinations")
    print("=" * 60)

    # Top pairs by layer1_score
    print(f"\nTop {min(top_n, len(sig))} pairs by layer1_score:")
    display_cols = [
        "lr_name",
        "sender_type",
        "receiver_type",
        "I_bivar",
        "rho_LR",
        "p_moran",
        "p_lag_fdr",
        "sig_type",
        "layer1_score",
    ]
    display_cols = [c for c in display_cols if c in sig.columns]
    print(sig[display_cols].head(top_n).to_string(index=True))

    # Breakdown by lr_type
    print("\nBreakdown by signaling type:")
    for lrt, grp in sig.groupby("lr_type"):
        print(f"  {lrt:12s}: {len(grp):4d} combinations ({grp['lr_name'].nunique()} unique LR pairs)")

    # Breakdown by sig_type (auto-selected geometry)
    if "sig_type" in sig.columns:
        print("\nAuto-selected geometry (sig_type) for Layer 2:")
        for st, grp in sig.groupby("sig_type"):
            print(f"  {st:12s}: {len(grp):4d} combinations")

    # Top cell type pairs
    print("\nTop sender → receiver type pairs:")
    pair_counts = sig.groupby(["sender_type", "receiver_type"]).size().sort_values(ascending=False).head(10)
    for (s, r), cnt in pair_counts.items():
        print(f"  {s} → {r}: {cnt}")

    # Score range
    print(f"\nlayer1_score range: {sig['layer1_score'].min():.4f} – {sig['layer1_score'].max():.4f}")
    print(f"I_bivar range:      {sig['I_bivar'].min():.4f} – {sig['I_bivar'].max():.4f}")
    print(f"rho_LR range:       {sig['rho_LR'].min():.4f} – {sig['rho_LR'].max():.4f}")


def get_top_pairs(
    layer1_results: dict,
    top_k: int = 20,
    lr_type: str | None = None,
    sender_type: str | None = None,
    receiver_type: str | None = None,
) -> pd.DataFrame:
    """
    Get top-K significant pairs, optionally filtered by type.

    Used to limit Layer 2 computation when running many FOVs.

    Parameters
    ----------
    layer1_results : dict
        Output of run_layer1().
    top_k : int
        Maximum number of pairs to return.
    lr_type : str | None
        Filter to 'juxtacrine', 'secreted', or 'ecm'.
    sender_type : str | None
        Filter to specific sender cell type.
    receiver_type : str | None
        Filter to specific receiver cell type.

    Returns
    -------
    pd.DataFrame
        Top-K significant pairs, sorted by layer1_score.

    Examples
    --------
    >>> # Top 20 juxtacrine pairs
    >>> top = get_top_pairs(layer1_results, top_k=20, lr_type='juxtacrine')

    >>> # All Tumor → T_cell interactions
    >>> top = get_top_pairs(layer1_results,
    ...                     sender_type='Tumor',
    ...                     receiver_type='T_cell')
    """
    sig = layer1_results["significant_pairs"].copy()

    if lr_type is not None:
        sig = sig[sig["lr_type"] == lr_type]
    if sender_type is not None:
        sig = sig[sig["sender_type"] == sender_type]
    if receiver_type is not None:
        sig = sig[sig["receiver_type"] == receiver_type]

    return sig.head(top_k).reset_index(drop=True)


# ── Private helpers ───────────────────────────────────────────────────────────


def _print_layer1_summary(
    combined: pd.DataFrame,
    significant: pd.DataFrame,
    alpha_moran: float,
    alpha_lag: float,
    min_rho: float,
) -> None:
    """Print filtering funnel summary."""
    n_total = len(combined)
    n_moran = (combined["p_moran"] < alpha_moran).sum()
    n_lag = (combined["p_lag_fdr"] < alpha_lag).sum()
    n_rho = (combined["rho_LR"] > min_rho).sum()
    n_sig = len(significant)

    print(f"\n{'─' * 50}")
    print("Layer 1 filter funnel:")
    print(f"  Total tested        : {n_total:5d}")
    print(f"  p_moran < {alpha_moran}     : {n_moran:5d}  (Moran significant)")
    print(f"  p_lag_fdr < {alpha_lag}   : {n_lag:5d}  (Spatial lag significant)")
    print(f"  rho_LR > {min_rho}       : {n_rho:5d}  (Minimum effect size)")
    print(f"  ALL filters pass    : {n_sig:5d}  ← Layer 2 input")
    print(f"{'─' * 50}")

    if n_sig > 0:
        print(
            f"\n  layer1_score range: {significant['layer1_score'].min():.4f} – {significant['layer1_score'].max():.4f}"
        )
        print(f"  Unique LR pairs:    {significant['lr_name'].nunique()}")
        print(f"  Unique type pairs:  {significant.groupby(['sender_type', 'receiver_type']).ngroups}")


def _empty_layer1_results(
    W_dict: dict[str, sparse.csr_matrix],
    sp: spatioloji,
    layer: str | None,
) -> dict:
    """Return empty layer1_results with correct structure."""
    return {
        "significant_pairs": pd.DataFrame(),
        "all_moran": pd.DataFrame(),
        "all_lag": pd.DataFrame(),
        "all_combined": pd.DataFrame(),
        "W_juxtacrine": W_dict.get("juxtacrine"),
        "W_secreted": W_dict.get("secreted"),
        "W_ecm": W_dict.get("ecm"),
        "type_masks": {},
        "expr": np.array([]),
        "gene2idx": {},
    }
