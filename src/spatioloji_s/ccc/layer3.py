"""
layer3.py - Layer 3: Communication Pattern Detection.

Two complementary frameworks:

  Framework D — Contrastive Scoring
    For each significant LR pair, asks:
      "Is the observed signal driven by expression, geometry, or both?"

    Two null models (both use same contact edges from Layer 2):
      Null_expr : shuffle L and R vectors, keep geo_weight fixed
                  → destroys biological expression-position coupling
                  → high z_expr means high-L senders are placed next
                    to high-R receivers (expression co-localization)

      Null_geo  : shuffle geo_weight across edges, keep expression fixed
                  → destroys contact geometry coupling
                  → high z_geo means contact_fraction is enriched at
                    high-L/high-R interfaces (active cell polarization)

    2×2 driver classification:
      SYNERGISTIC      : z_expr > 1.96 AND z_geo > 1.96
      EXPRESSION_DRIVEN: z_expr > 1.96 only
      GEOMETRY_DRIVEN  : z_geo  > 1.96 only
      WEAK             : neither (removed from NMF)

  Framework G — NMF Communication Programs
    Build communication tensor X[i,j,lr] from Layer 2 scores,
    weighted by contrastive score (SYNERGISTIC pairs get full weight).
    Flatten to 2D matrix, run NMF to find K communication programs.

    Each program k captures a recurring CCC mode:
      A[:,k]  : sender   loading per cell
      B[:,k]  : receiver loading per cell
      C[:,k]  : LR pair  loading (which signals define this program)

    Polygon regularization: penalize neighboring cells with different
    program loadings → spatially coherent programs.

    K selection: reconstruction error elbow (range 1–15).

Output: layer3_results dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from spatioloji_s.spatial.polygon.graph import PolygonSpatialGraph

import numpy as np
import pandas as pd
from scipy import sparse

from .layer1 import _get_expr_vector

# ─────────────────────────────────────────────────────────────────────────────
# Framework D: Contrastive Scoring
# ─────────────────────────────────────────────────────────────────────────────

DRIVER_TYPES = {
    "SYNERGISTIC": (True, True),
    "EXPRESSION_DRIVEN": (True, False),
    "GEOMETRY_DRIVEN": (False, True),
    "WEAK": (False, False),
}

CONTRASTIVE_WEIGHTS = {
    "SYNERGISTIC": 1.0,
    "EXPRESSION_DRIVEN": 0.6,
    "GEOMETRY_DRIVEN": 0.6,
    "WEAK": 0.1,
}


def _observed_score(
    L: np.ndarray,
    R: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    w: np.ndarray,
) -> float:
    """
    Observed = Σ_{(i,j)∈contacts} sqrt(L_i * R_j) * w[i,j]

    Uses only cells in the contact graph (ia, ib from Layer 2 edge_data).
    """
    if len(ia) == 0:
        return 0.0
    L_i = np.maximum(L[ia], 0.0)
    R_j = np.maximum(R[ib], 0.0)
    return float(np.sum(np.sqrt(L_i * R_j) * w))


def _null_expression(
    L: np.ndarray,
    R: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    w: np.ndarray,
    s_mask: np.ndarray,
    r_mask: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Null_expr: shuffle expression labels within sender and receiver pools.

    Vectorized: pre-maps edge indices to pool positions, generates all
    n_perm shuffles at once as (n_perm, pool_size) matrices, then scores
    all permutations in one matrix op — no Python loop over permutations.
    """
    if len(ia) == 0:
        return np.zeros(n_perm, dtype=np.float64)

    s_indices = np.where(s_mask)[0]  # (n_s,)
    r_indices = np.where(r_mask)[0]  # (n_r,)

    # Map each edge's cell index → position inside its sender/receiver pool
    # e.g. ia[e]=47, s_indices[3]=47 → ia_in_s[e]=3
    s_inv = {s: k for k, s in enumerate(s_indices)}
    r_inv = {r: k for k, r in enumerate(r_indices)}
    ia_in_s = np.array([s_inv[i] for i in ia], dtype=np.intp)  # (n_edges,)
    ib_in_r = np.array([r_inv[j] for j in ib], dtype=np.intp)  # (n_edges,)

    # Pool values at edges (only what the score formula needs)
    L_pool = np.maximum(L[s_indices], 0.0).astype(np.float64)  # (n_s,)
    R_pool = np.maximum(R[r_indices], 0.0).astype(np.float64)  # (n_r,)
    w_f = w.astype(np.float64)  # (n_edges,)

    # Batch shuffle all permutations at once: (n_perm, pool_size)
    L_shuffled = rng.permuted(np.tile(L_pool, (n_perm, 1)), axis=1)
    R_shuffled = rng.permuted(np.tile(R_pool, (n_perm, 1)), axis=1)

    # Gather edge values for every permutation: (n_perm, n_edges)
    L_at_edges = L_shuffled[:, ia_in_s]
    R_at_edges = R_shuffled[:, ib_in_r]

    # Score all permutations in one vectorized call: (n_perm,)
    return (np.sqrt(L_at_edges * R_at_edges) * w_f).sum(axis=1)


def _null_geometry(
    L: np.ndarray,
    R: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    w: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Null_geo: shuffle geo_weight across contact edges, keep expression fixed.

    Vectorized: lrprod is fixed per edge; tiles w into a (n_perm, n_edges)
    matrix, shuffles each row independently with rng.permuted, then scores
    all permutations in one matrix-vector multiply — no Python loop.
    """
    if len(ia) == 0:
        return np.zeros(n_perm, dtype=np.float64)

    L_i = np.maximum(L[ia], 0.0)
    R_j = np.maximum(R[ib], 0.0)
    lrprod = np.sqrt(L_i * R_j).astype(np.float64)  # (n_edges,) fixed

    # (n_perm, n_edges) — each row is one independent shuffle of w
    W_perms = rng.permuted(
        np.tile(w.astype(np.float64), (n_perm, 1)),
        axis=1,
    )
    # Dot each shuffled row against lrprod → (n_perm,)
    return W_perms @ lrprod


def _classify_driver(
    z_expr: float,
    z_geo: float,
    z_thresh: float = 1.96,
) -> str:
    """Classify into one of four driver types based on z-scores."""
    sig_expr = z_expr > z_thresh
    sig_geo = z_geo > z_thresh
    for name, (e, g) in DRIVER_TYPES.items():
        if sig_expr == e and sig_geo == g:
            return name
    return "WEAK"


def compute_contrastive_scores(
    sp: spatioloji,
    layer1_results: dict,
    layer2_results: dict,
    n_permutations: int = 1000,
    z_thresh: float = 1.96,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute contrastive scores for all significant LR pairs.

    For each (lr_name, sender_type, receiver_type):
      1. Compute observed = Σ sqrt(L_i R_j) × geo_weight
      2. Run Null_expr (shuffle expression in sender/receiver cells)
      3. Run Null_geo  (shuffle geo_weight across edges)
      4. Compute z_expr, z_geo, p_expr, p_geo
      5. Classify driver type

    Parameters
    ----------
    sp : spatioloji
    layer1_results : dict   Output of run_layer1().
    layer2_results : dict   Output of run_layer2().
    n_permutations : int    Permutations per null model. Default 1000.
    z_thresh : float        Z-score threshold for significance. Default 1.96.
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: lr_name, ligand, receptor, lr_type, pathway,
                 sender_type, receiver_type,
                 observed, z_expr, p_expr, z_geo, p_geo,
                 contrastive_score, driver_type, contrastive_weight
    """
    rng = np.random.default_rng(seed)
    expr = layer1_results["expr"]
    gene2idx = layer1_results["gene2idx"]
    type_masks = layer1_results["type_masks"]
    sig = layer1_results["significant_pairs"]
    edge_data = layer2_results["edge_data"]
    cell_ids = layer2_results["cell_ids"]

    # Build edge_data lookup: (lr_name, sender_type, receiver_type) → edge_entry
    edge_lookup: dict[tuple, dict] = {(e["lr_name"], e["sender_type"], e["receiver_type"]): e for e in edge_data}

    # Build lr_name → (ligand, receptor, pathway, lr_type)
    lr_meta: dict[str, dict] = {}
    for _, row in sig.iterrows():
        lr_meta[row["lr_name"]] = {
            "ligand": row["ligand"],
            "receptor": row["receptor"],
            "pathway": row["pathway"],
            "lr_type": row["lr_type"],
        }

    records: list[dict] = []
    n_rows = len(sig)
    print(f"\n[Layer3 Contrastive] {n_rows} combinations × 2 null models × {n_permutations} permutations")

    for row_idx, row in sig.iterrows():
        lr_name = row["lr_name"]
        sender_type = row["sender_type"]
        recv_type = row["receiver_type"]

        key = (lr_name, sender_type, recv_type)
        edge = edge_lookup.get(key)
        if edge is None:
            continue  # geo_weight already in m_vals structure

        # Recompute raw geo_weight (w without expression)
        # from Layer 2 edge_data we only stored m_vals = sqrt(L*R)*w
        # We need raw w — re-extract from the geo dict
        geo = layer2_results["geo"]
        sig_type = edge["sig_type"]
        s_mask = type_masks.get(sender_type, np.zeros(len(cell_ids), dtype=bool))
        r_mask = type_masks.get(recv_type, np.zeros(len(cell_ids), dtype=bool))

        edge_ok = s_mask[geo["ia"]] & r_mask[geo["ib"]]
        ia_raw = geo["ia"][edge_ok]
        ib_raw = geo["ib"][edge_ok]

        if sig_type == "juxtacrine":
            w_raw = (geo["fa"][edge_ok] * geo["fb"][edge_ok]).astype(np.float64)
        elif sig_type == "secreted":
            w_raw = (geo["fb_b"][edge_ok] * geo["decay"][edge_ok]).astype(np.float64)
        else:
            w_raw = ((1.0 - geo["sol_b"][edge_ok]) * geo["decay"][edge_ok]).astype(np.float64)

        if len(ia_raw) == 0:
            continue

        # Expression vectors
        meta = lr_meta.get(lr_name, {})
        L = _get_expr_vector(expr, gene2idx, str(meta.get("ligand", "")).split("|"))
        R = _get_expr_vector(expr, gene2idx, str(meta.get("receptor", "")).split("|"))

        L = L.astype(np.float64)
        R = R.astype(np.float64)

        # Observed score
        observed = _observed_score(L, R, ia_raw, ib_raw, w_raw)

        # Null_expr
        null_expr = _null_expression(L, R, ia_raw, ib_raw, w_raw, s_mask, r_mask, n_permutations, rng)

        # Null_geo
        null_geo = _null_geometry(L, R, ia_raw, ib_raw, w_raw, n_permutations, rng)

        # Z-scores
        def _z(obs, null):
            std = null.std()
            if std < 1e-10:
                return 0.0
            return float((obs - null.mean()) / std)

        z_expr = _z(observed, null_expr)
        z_geo = _z(observed, null_geo)

        # One-sided p-values (P(null >= observed))
        p_expr = float((null_expr >= observed).mean())
        p_geo = float((null_geo >= observed).mean())
        p_expr = max(p_expr, 1.0 / n_permutations)
        p_geo = max(p_geo, 1.0 / n_permutations)

        # Driver type
        driver_type = _classify_driver(z_expr, z_geo, z_thresh)
        contrastive_weight = CONTRASTIVE_WEIGHTS[driver_type]

        # Contrastive score = geometric mean of z-scores (clipped at 0)
        contrastive_score = float(np.sqrt(max(z_expr, 0.0) * max(z_geo, 0.0)))

        records.append(
            {
                "lr_name": lr_name,
                "ligand": meta.get("ligand", ""),
                "receptor": meta.get("receptor", ""),
                "lr_type": meta.get("lr_type", ""),
                "pathway": meta.get("pathway", ""),
                "sender_type": sender_type,
                "receiver_type": recv_type,
                "sig_type": sig_type,
                "observed": observed,
                "z_expr": z_expr,
                "p_expr": p_expr,
                "z_geo": z_geo,
                "p_geo": p_geo,
                "contrastive_score": contrastive_score,
                "driver_type": driver_type,
                "contrastive_weight": contrastive_weight,
            }
        )

        if (row_idx + 1) % 20 == 0:
            print(f"  [{row_idx + 1}/{n_rows}] {lr_name} {sender_type}→{recv_type}: {driver_type}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("contrastive_score", ascending=False).reset_index(drop=True)

    # Summary
    for dt in ["SYNERGISTIC", "EXPRESSION_DRIVEN", "GEOMETRY_DRIVEN", "WEAK"]:
        n = (df["driver_type"] == dt).sum()
        print(f"  {dt:20s}: {n:4d}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Framework G: NMF Communication Programs
# ─────────────────────────────────────────────────────────────────────────────


def _build_communication_matrix(
    layer2_results: dict,
    contrastive_df: pd.DataFrame,
    n_cells: int,
    min_weight: float = 0.05,
) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    """
    Build 2D communication matrix X_flat for NMF.

    X_flat shape: (n_contact_pairs, n_lr_pairs)
      rows = unique (i,j) touching pairs
      cols = unique lr_names (WEAK pairs excluded or downweighted)

    X_flat[pair_idx, lr_idx] = combined_strength × contrastive_weight

    Returns
    -------
    X_flat       : np.ndarray (n_pairs, n_lr)
    pair_list    : list of (ia, ib) index tuples (row index)
    lr_names     : list of lr_name strings (column index)
    """
    edge_data = layer2_results["edge_data"]

    # Contrastive weight lookup: (lr_name, sender_type, receiver_type) → weight
    cw_lookup: dict[tuple, float] = {}
    if not contrastive_df.empty:
        for _, row in contrastive_df.iterrows():
            key = (row["lr_name"], row["sender_type"], row["receiver_type"])
            cw_lookup[key] = float(row["contrastive_weight"])

    # Collect all unique (ia, ib) pairs and lr_names
    all_pairs: set[tuple[int, int]] = set()
    lr_set: set[str] = set()

    for e in edge_data:
        for ia_val, ib_val in zip(e["ia"], e["ib"], strict=True):
            all_pairs.add((int(ia_val), int(ib_val)))
        lr_set.add(e["lr_name"])

    pair_list = sorted(all_pairs)
    lr_names = sorted(lr_set)

    if not pair_list or not lr_names:
        return np.zeros((1, 1)), [], []

    pair2idx = {p: k for k, p in enumerate(pair_list)}
    lr2idx = {lr: k for k, lr in enumerate(lr_names)}

    n_pairs = len(pair_list)
    n_lr = len(lr_names)
    X_flat = np.zeros((n_pairs, n_lr), dtype=np.float32)

    for e in edge_data:
        lr_idx = lr2idx[e["lr_name"]]
        cw_key = (e["lr_name"], e["sender_type"], e["receiver_type"])
        cw = cw_lookup.get(cw_key, 0.6)  # default if contrastive not run

        m_vals = e["m_vals"].astype(np.float32)
        ia_arr = e["ia"]
        ib_arr = e["ib"]

        for k_e in range(len(ia_arr)):
            p_key = (int(ia_arr[k_e]), int(ib_arr[k_e]))
            if p_key in pair2idx:
                p_idx = pair2idx[p_key]
                X_flat[p_idx, lr_idx] += float(m_vals[k_e]) * cw

    # Remove near-zero rows
    row_sums = X_flat.sum(axis=1)
    keep_rows = row_sums >= min_weight * row_sums.max()
    X_flat = X_flat[keep_rows]
    pair_list = [pair_list[k] for k in np.where(keep_rows)[0]]

    return X_flat, pair_list, lr_names


def _nmf_als(
    X: np.ndarray,
    K: int,
    n_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    NMF via Multiplicative Updates (Lee & Seung 2001).

    Minimizes ||X - W @ H||²_F with W,H ≥ 0.

    Parameters
    ----------
    X      : (n_samples, n_features) non-negative matrix
    K      : number of components
    n_iter : max iterations
    tol    : relative reconstruction error tolerance

    Returns
    -------
    W      : (n_samples, K) — sample loadings (pair weights)
    H      : (K, n_features) — feature loadings (LR pair weights)
    recon_err : final ||X - WH||_F / ||X||_F
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    eps = 1e-10

    # Initialize W, H with scaled random (Boutsidis & Gallopoulos 2008 style)
    scale = np.sqrt(X.mean() / K)
    W = rng.random((n, K)).astype(np.float32) * scale
    H = rng.random((K, m)).astype(np.float32) * scale

    X_norm = float(np.linalg.norm(X))
    if X_norm < eps:
        return W, H, 0.0

    prev_err = np.inf
    for _ in range(n_iter):
        # Update H: H *= (W' X) / (W' W H + eps)
        WtX = W.T @ X  # (K, m)
        WtW = W.T @ W  # (K, K)
        H *= WtX / (WtW @ H + eps)
        H = np.maximum(H, eps)

        # Update W: W *= (X H') / (W H H' + eps)
        XHt = X @ H.T  # (n, K)
        HHt = H @ H.T  # (K, K)
        W *= XHt / (W @ HHt + eps)
        W = np.maximum(W, eps)

        # Convergence
        recon = np.linalg.norm(X - W @ H) / X_norm
        if abs(prev_err - recon) < tol:
            break
        prev_err = recon

    return W, H, float(prev_err)


def _polygon_laplacian(
    contact_graph: PolygonSpatialGraph,
    sp: spatioloji,
) -> sparse.csr_matrix:
    """
    Compute normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}
    from the polygon contact graph.

    Used for spatial regularization in NMF:
      Tr(A' L A)  penalizes neighboring cells with different loadings.

    Returns (n_cells × n_cells) sparse CSR matrix aligned to sp.cell_index.
    """
    cell_ids = np.array(sp.cell_index)
    n = len(cell_ids)
    graph_ids = np.array(contact_graph.cell_index)
    graph_idx = {cid: k for k, cid in enumerate(graph_ids)}

    # Map adjacency from graph ordering to sp ordering
    A = contact_graph.adjacency.astype(np.float32)
    rows, cols = A.nonzero()

    sp_rows, sp_cols, vals = [], [], []
    id2sp = {gid: sp_idx for sp_idx, gid in enumerate(cell_ids) if gid in graph_idx}

    for r, c in zip(rows, cols, strict=True):
        gr, gc = graph_ids[r], graph_ids[c]
        if gr in id2sp and gc in id2sp:
            sp_rows.append(id2sp[gr])
            sp_cols.append(id2sp[gc])
            vals.append(1.0)

    if not sp_rows:
        return sparse.eye(n, format="csr", dtype=np.float32)

    A_sp = sparse.csr_matrix((vals, (sp_rows, sp_cols)), shape=(n, n), dtype=np.float32)

    # Degree matrix
    degrees = np.array(A_sp.sum(axis=1)).flatten()
    d_inv_sq = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sq = sparse.diags(d_inv_sq, format="csr")

    # Normalized Laplacian
    L = sparse.eye(n, format="csr") - D_inv_sq @ A_sp @ D_inv_sq
    return L.astype(np.float32)


def _regularized_nmf(
    X: np.ndarray,
    pair_list: list[tuple[int, int]],
    K: int,
    L_poly: sparse.csr_matrix,
    lambda_reg: float = 0.1,
    n_iter: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Polygon-regularized NMF.

    Extends standard NMF by deriving per-cell sender (A) and receiver (B)
    loadings from the pair loading matrix W_pairs, then adding spatial
    regularization to keep neighboring cells in similar programs.

    Approach:
      1. Run standard NMF to get W_pairs (n_pairs, K) and H (K, n_lr)
      2. Derive A[i,k] = Σ_{j:(i,j)∈pairs} W_pairs[(i,j), k]  (sender)
         Derive B[j,k] = Σ_{i:(i,j)∈pairs} W_pairs[(i,j), k]  (receiver)
      3. Smooth A, B via one step of graph-Laplacian diffusion:
         A_smooth = A - λ × L_poly @ A
         (penalizes neighboring cells with very different loadings)
      4. Re-normalize columns of A, B to unit max.

    Returns
    -------
    A        : (n_cells, K) sender   loadings — spatially smoothed
    B        : (n_cells, K) receiver loadings — spatially smoothed
    H        : (K, n_lr)   LR pair  loadings
    W_pairs  : (n_pairs, K) raw pair loadings (for diagnostics)
    """
    n_cells = L_poly.shape[0]

    # Standard NMF on X_flat
    W_pairs, H, _ = _nmf_als(X, K, n_iter=n_iter, seed=seed)

    # Derive per-cell sender/receiver loadings from pair loadings
    A = np.zeros((n_cells, K), dtype=np.float32)
    B = np.zeros((n_cells, K), dtype=np.float32)

    for pair_idx, (ia, ib) in enumerate(pair_list):
        if ia < n_cells:
            A[ia] += W_pairs[pair_idx]  # sender loading
        if ib < n_cells:
            B[ib] += W_pairs[pair_idx]  # receiver loading

    # Spatial regularization: one step of Laplacian smoothing
    # A_smooth[i] = A[i] - λ × Σ_j L[i,j] × A[j]
    # = (I - λ L) @ A  — keeps neighbors similar
    if lambda_reg > 0:
        smooth_op = sparse.eye(n_cells, format="csr") - lambda_reg * L_poly
        A = smooth_op @ A
        B = smooth_op @ B
        A = np.maximum(A, 0.0)  # non-negativity after smoothing
        B = np.maximum(B, 0.0)

    # Normalize columns to [0, 1]
    for k in range(K):
        a_max = A[:, k].max()
        b_max = B[:, k].max()
        if a_max > 1e-10:
            A[:, k] /= a_max
        if b_max > 1e-10:
            B[:, k] /= b_max

    return A, B, H, W_pairs


def _select_k(
    X: np.ndarray,
    k_range: range,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction errors for K selection.

    Runs NMF for each K in k_range, returns reconstruction errors.
    User should look for the elbow.

    Returns
    -------
    k_values : np.ndarray of tested K values
    errors   : np.ndarray of reconstruction errors
    """
    k_values = np.array(list(k_range))
    errors = np.zeros(len(k_values), dtype=np.float32)

    print(f"  Selecting K over range {list(k_range)}...")
    for idx, k in enumerate(k_values):
        _, _, err = _nmf_als(X, k, n_iter=n_iter, seed=seed)
        errors[idx] = err
        print(f"    K={k}: recon_err={err:.4f}")

    return k_values, errors


def _elbow_k(k_values: np.ndarray, errors: np.ndarray) -> int:
    """
    Select K at the elbow of the reconstruction error curve.

    Uses maximum curvature (second derivative) method.
    Falls back to K=3 if the curve is too flat.
    """
    if len(k_values) < 3:
        return int(k_values[0])

    # Normalize to [0,1] for stable curvature
    e_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)

    # Second difference as curvature proxy
    curvature = np.abs(np.diff(e_norm, n=2))
    elbow_idx = int(np.argmax(curvature)) + 1  # +1 offset from double diff

    return int(k_values[elbow_idx])


def _interpret_programs(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    lr_names: list[str],
    cell_ids: np.ndarray,
    type_masks: dict[str, np.ndarray],
    K: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Interpret each NMF program: top senders, receivers, LR pairs.

    For each program k:
      top_sender_cells    : cell IDs with highest A[:,k]
      top_receiver_cells  : cell IDs with highest B[:,k]
      top_lr_pairs        : LR names with highest H[k,:]
      top_sender_types    : cell types enriched in A[:,k]
      top_receiver_types  : cell types enriched in B[:,k]

    Returns
    -------
    pd.DataFrame one row per program.
    """
    rows = []
    for k in range(K):
        a_k = A[:, k]
        b_k = B[:, k]
        h_k = H[k, :]

        # Top sender / receiver cells
        top_s_idx = np.argsort(a_k)[::-1][:top_n]
        top_r_idx = np.argsort(b_k)[::-1][:top_n]
        top_lr_idx = np.argsort(h_k)[::-1][:top_n]

        top_senders = list(cell_ids[top_s_idx])
        top_receivers = list(cell_ids[top_r_idx])
        top_lr = [lr_names[i] for i in top_lr_idx if i < len(lr_names)]

        # Cell type composition of top senders / receivers
        sender_type_comp = _type_composition(a_k, type_masks, threshold=0.5)
        receiver_type_comp = _type_composition(b_k, type_masks, threshold=0.5)

        # Dominant sender / receiver type (highest mean loading)
        dom_sender = max(sender_type_comp, key=sender_type_comp.get, default="?")
        dom_receiver = max(receiver_type_comp, key=receiver_type_comp.get, default="?")

        rows.append(
            {
                "program": k,
                "dominant_sender_type": dom_sender,
                "dominant_recv_type": dom_receiver,
                "top_lr_pairs": top_lr,
                "top_sender_cells": top_senders,
                "top_receiver_cells": top_receivers,
                "sender_type_comp": sender_type_comp,
                "receiver_type_comp": receiver_type_comp,
                "mean_sender_loading": float(a_k.mean()),
                "mean_receiver_loading": float(b_k.mean()),
                "n_active_senders": int((a_k > 0.2).sum()),
                "n_active_receivers": int((b_k > 0.2).sum()),
            }
        )

    return pd.DataFrame(rows)


def _type_composition(
    loadings: np.ndarray,
    type_masks: dict[str, np.ndarray],
    threshold: float = 0.2,
) -> dict[str, float]:
    """Mean loading per cell type (only cells above threshold included)."""
    comp = {}
    active = loadings >= threshold
    for t, mask in type_masks.items():
        both = mask & active
        if both.sum() > 0:
            comp[t] = float(loadings[both].mean())
    return comp


# ── Public entry point ────────────────────────────────────────────────────────


def run_layer3(
    sp: spatioloji,
    layer1_results: dict,
    layer2_results: dict,
    contact_graph: PolygonSpatialGraph,
    n_permutations: int = 1000,
    z_thresh: float = 1.96,
    K: int | None = None,
    k_range: range = range(2, 11),
    lambda_reg: float = 0.1,
    n_iter_nmf: int = 200,
    top_n_interpret: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run full Layer 3: Contrastive Scoring + NMF Communication Programs.

    Steps:
      1. Contrastive scoring — driver type per LR pair
      2. Build communication matrix X_flat
      3. Select K (if K=None, use elbow on k_range)
      4. Polygon-regularized NMF
      5. Interpret programs (top senders, receivers, LR pairs)

    Parameters
    ----------
    sp : spatioloji
    layer1_results : dict   Output of run_layer1().
    layer2_results : dict   Output of run_layer2().
    contact_graph : PolygonSpatialGraph
    n_permutations : int    Permutations per null model. Default 1000.
    z_thresh : float        Z-score threshold for driver classification. Default 1.96.
    K : int | None          Number of NMF programs.
                            None → auto-select via elbow on k_range.
    k_range : range         K values to test for elbow. Default range(2, 11).
    lambda_reg : float      Spatial regularization strength. Default 0.1.
    n_iter_nmf : int        NMF iterations. Default 200.
    top_n_interpret : int   Top cells / LR pairs per program. Default 5.
    seed : int

    Returns
    -------
    dict with keys:
        'contrastive'     : pd.DataFrame  — driver types per LR pair
        'k_selection'     : pd.DataFrame  — K vs reconstruction error
        'K'               : int           — selected K
        'A'               : np.ndarray    — (n_cells, K) sender loadings
        'B'               : np.ndarray    — (n_cells, K) receiver loadings
        'H'               : np.ndarray    — (K, n_lr) LR pair loadings
        'W_pairs'         : np.ndarray    — (n_pairs, K) raw pair loadings
        'programs'        : pd.DataFrame  — program interpretation
        'lr_names'        : list[str]     — LR pair column names for H
        'pair_list'       : list[tuple]   — (ia, ib) row index for W_pairs
        'cell_ids'        : np.ndarray    — aligned to A, B rows

    Examples
    --------
    >>> layer3_results = run_layer3(
    ...     sp_fov, layer1_results, layer2_results, contact_graph,
    ...     K=5, lambda_reg=0.1,
    ... )
    >>> programs = layer3_results['programs']
    >>> print(programs[['program', 'dominant_sender_type',
    ...                  'dominant_recv_type', 'top_lr_pairs']])
    """
    print("\n" + "=" * 60)
    print("LAYER 3: Communication Pattern Detection")
    print("=" * 60)

    cell_ids = layer2_results["cell_ids"]
    type_masks = layer1_results["type_masks"]

    # ── Step 1: Contrastive Scoring ───────────────────────────────────────────
    print("\n[Step 1/4] Contrastive scoring...")
    contrastive_df = compute_contrastive_scores(
        sp,
        layer1_results,
        layer2_results,
        n_permutations=n_permutations,
        z_thresh=z_thresh,
        seed=seed,
    )

    # ── Step 2: Build communication matrix ────────────────────────────────────
    print("\n[Step 2/4] Building communication matrix for NMF...")
    X_flat, pair_list, lr_names = _build_communication_matrix(layer2_results, contrastive_df, n_cells=len(cell_ids))
    print(f"  X_flat shape: {X_flat.shape}  ({len(pair_list)} contact pairs × {len(lr_names)} LR pairs)")

    if X_flat.shape[0] < 2 or X_flat.shape[1] < 2:
        print("  ⚠ Communication matrix too small for NMF.")
        return _empty_layer3_results(contrastive_df, cell_ids)

    # ── Step 3: Select K ──────────────────────────────────────────────────────
    print("\n[Step 3/4] Selecting K...")
    if K is None:
        # Clamp k_range to not exceed matrix rank
        max_k = min(X_flat.shape[0], X_flat.shape[1], max(k_range))
        k_range = range(min(k_range), min(max_k + 1, max(k_range) + 1))
        k_vals, k_errs = _select_k(X_flat, k_range, n_iter=50, seed=seed)
        K = _elbow_k(k_vals, k_errs)
        print(f"  Auto-selected K = {K}")
        k_sel_df = pd.DataFrame({"K": k_vals, "recon_error": k_errs})
    else:
        k_vals, k_errs = _select_k(X_flat, range(K, K + 1), n_iter=50, seed=seed)
        k_sel_df = pd.DataFrame({"K": [K], "recon_error": [k_errs[0]]})
        print(f"  Using user-specified K = {K}")

    K = max(K, 1)

    # ── Step 4: Polygon-regularized NMF ──────────────────────────────────────
    print(f"\n[Step 4/4] Running NMF (K={K}, λ={lambda_reg})...")
    L_poly = _polygon_laplacian(contact_graph, sp)

    A, B, H, W_pairs = _regularized_nmf(
        X_flat,
        pair_list,
        K,
        L_poly,
        lambda_reg=lambda_reg,
        n_iter=n_iter_nmf,
        seed=seed,
    )
    print(f"  A: {A.shape}, B: {B.shape}, H: {H.shape}")

    # ── Step 5: Interpret programs ────────────────────────────────────────────
    programs_df = _interpret_programs(
        A,
        B,
        H,
        lr_names,
        cell_ids,
        type_masks,
        K,
        top_n=top_n_interpret,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"Layer 3 summary — {K} programs:")
    for _, prow in programs_df.iterrows():
        print(
            f"  Program {prow['program']}: "
            f"{prow['dominant_sender_type']} → {prow['dominant_recv_type']} | "
            f"top LR: {prow['top_lr_pairs'][:3]}"
        )
    print(f"{'─' * 50}")

    return {
        "contrastive": contrastive_df,
        "k_selection": k_sel_df,
        "K": K,
        "A": A,
        "B": B,
        "H": H,
        "W_pairs": W_pairs,
        "programs": programs_df,
        "lr_names": lr_names,
        "pair_list": pair_list,
        "cell_ids": cell_ids,
    }


def _empty_layer3_results(
    contrastive_df: pd.DataFrame,
    cell_ids: np.ndarray,
) -> dict:
    """Return empty layer3_results with correct structure."""
    return {
        "contrastive": contrastive_df,
        "k_selection": pd.DataFrame(),
        "K": 0,
        "A": np.zeros((len(cell_ids), 0)),
        "B": np.zeros((len(cell_ids), 0)),
        "H": np.zeros((0, 0)),
        "W_pairs": np.zeros((0, 0)),
        "programs": pd.DataFrame(),
        "lr_names": [],
        "pair_list": [],
        "cell_ids": cell_ids,
    }


# ── Convenience accessors ─────────────────────────────────────────────────────


def get_program_cells(
    layer3_results: dict,
    program: int,
    role: str = "both",
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Get cells participating in a specific NMF program.

    Parameters
    ----------
    layer3_results : dict   Output of run_layer3().
    program : int           Program index (0-indexed).
    role : str              'sender', 'receiver', or 'both'.
    threshold : float       Minimum loading to be considered active.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, sender_loading, receiver_loading, dominant_role
    """
    A = layer3_results["A"]
    B = layer3_results["B"]
    cell_ids = layer3_results["cell_ids"]
    K = layer3_results["K"]

    if program >= K:
        raise ValueError(f"program={program} out of range (K={K})")

    a_k = A[:, program]
    b_k = B[:, program]

    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sender_loading": a_k,
            "receiver_loading": b_k,
        }
    )

    df["dominant_role"] = "inactive"
    both_active = (a_k >= threshold) & (b_k >= threshold)
    only_sender = (a_k >= threshold) & ~(b_k >= threshold)
    only_recvr = ~(a_k >= threshold) & (b_k >= threshold)

    df.loc[both_active, "dominant_role"] = "dual"
    df.loc[only_sender, "dominant_role"] = "sender"
    df.loc[only_recvr, "dominant_role"] = "receiver"

    if role == "sender":
        df = df[only_sender | both_active]
    elif role == "receiver":
        df = df[only_recvr | both_active]
    elif role == "both":
        df = df[both_active | only_sender | only_recvr]

    return df.reset_index(drop=True)


def get_program_lr_pairs(
    layer3_results: dict,
    program: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top LR pairs defining a specific NMF program.

    Parameters
    ----------
    layer3_results : dict
    program : int
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: lr_name, loading, rank
    """
    H = layer3_results["H"]
    lr_names = layer3_results["lr_names"]
    K = layer3_results["K"]

    if program >= K:
        raise ValueError(f"program={program} out of range (K={K})")

    h_k = H[program, :]
    top_idx = np.argsort(h_k)[::-1][:top_n]

    return pd.DataFrame(
        {
            "lr_name": [lr_names[i] for i in top_idx],
            "loading": h_k[top_idx],
            "rank": np.arange(1, len(top_idx) + 1),
        }
    )
