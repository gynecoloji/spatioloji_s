"""
layer2.py - Layer 2: Cell-Pair Level CCC Scoring.

Two complementary frameworks per LR pair:

  Framework A — Polygon Optimal Transport (OT)
    Models CCC as a supply-demand problem.
    Senders produce ligand (supply), receivers express receptor (demand).
    Cost matrix encodes polygon geometry — not just distance.
    Solves for transport plan T[i,j] = communication flow i→j.

  Framework C — Message Passing (MP)
    Three-step cascade: generate → aggregate → detect hubs.
    m[i,j] = sqrt(L_i × R_j) × geo_weight(i,j)
    Signal[j] = Σ_i m[i,j]   (total stimulation at receiver j)

Combined score = geometric mean of normalized OT flow and MP signal.
Hub cells = top cells by combined score in sender or receiver role.

Geometry conventions (from Layer 1 / boundaries.py)
-----------------------------------------------------
  fraction_a  = fraction of cell_a membrane touching cell_b  (sender→receiver)
  fraction_b  = fraction of cell_b membrane touching cell_a  (receiver→sender)
  free_boundary_fraction in sp.cell_meta
  morph_solidity in sp.cell_meta
  sigma (distance decay) passed from layer1 or estimated here

Output: layer2_results dict — consumed by Layer 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji


import numpy as np
import pandas as pd

from .database import LRPair
from .layer1 import _get_expr_vector

# ── Cost / weight geometry builders ──────────────────────────────────────────


def _build_geometry_arrays(
    sp: spatioloji,
    contact_frac_df: pd.DataFrame,
    sigma: float,
    coord_type: str = "global",
) -> dict:
    """
    Pre-compute all geometry arrays needed by both OT and MP.

    Returns a dict with:
        cell_a, cell_b   : arrays of cell ID pairs (n_edges,)
        ia, ib           : integer index arrays aligned to sp.cell_index
        fraction_a       : membrane fraction sender→receiver
        fraction_b       : membrane fraction receiver→sender
        distance         : centroid distance between pair (pixels)
        d_norm           : distance / d_max  (in [0,1])
        fb_a, fb_b       : free_boundary_fraction per cell (edge-aligned)
        sol_a, sol_b     : morph_solidity per cell (edge-aligned)
        decay            : exp(-distance / sigma)
    """
    cell_ids = np.array(sp.cell_index)
    id2idx = {cid: k for k, cid in enumerate(cell_ids)}

    # Spatial coords
    if coord_type == "global":
        xs = sp.spatial.x_global
        ys = sp.spatial.y_global
    else:
        xs = sp.spatial.x_local
        ys = sp.spatial.y_local

    coords = pd.DataFrame({"x": xs, "y": ys}, index=cell_ids)

    # Cell-level attributes (aligned to cell_ids)
    fb = pd.Series(
        sp.cell_meta.get("free_boundary_fraction", pd.Series(1.0, index=range(len(cell_ids)))).values,
        index=cell_ids,
    )
    sol = pd.Series(
        sp.cell_meta.get("morph_solidity", pd.Series(1.0, index=range(len(cell_ids)))).values,
        index=cell_ids,
    )

    # Build edge arrays from contact_frac_df
    ca_arr = contact_frac_df["cell_a"].values
    cb_arr = contact_frac_df["cell_b"].values
    fa_arr = contact_frac_df["fraction_a"].values.astype(np.float32)
    fb_arr = contact_frac_df["fraction_b"].values.astype(np.float32)

    ia_arr = np.array([id2idx.get(c, -1) for c in ca_arr])
    ib_arr = np.array([id2idx.get(c, -1) for c in cb_arr])

    # Filter out edges where either cell not in sp.cell_index
    valid = (ia_arr >= 0) & (ib_arr >= 0)
    ca_arr = ca_arr[valid]
    cb_arr = cb_arr[valid]
    ia_arr = ia_arr[valid]
    ib_arr = ib_arr[valid]
    fa_arr = fa_arr[valid]
    fb_arr = fb_arr[valid]

    # Centroid distances
    xa = coords.loc[ca_arr, "x"].values
    ya = coords.loc[ca_arr, "y"].values
    xb = coords.loc[cb_arr, "x"].values
    yb = coords.loc[cb_arr, "y"].values
    dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2).astype(np.float32)

    d_max = dist.max() if dist.max() > 0 else 1.0
    d_norm = dist / d_max
    decay = np.exp(-dist / max(sigma, 1e-6)).astype(np.float32)

    # Cell-level arrays, edge-aligned
    fb_a = np.array([float(fb.get(c, 1.0)) for c in ca_arr], dtype=np.float32)
    fb_b = np.array([float(fb.get(c, 1.0)) for c in cb_arr], dtype=np.float32)
    sol_a = np.array([float(sol.get(c, 1.0)) for c in ca_arr], dtype=np.float32)
    sol_b = np.array([float(sol.get(c, 1.0)) for c in cb_arr], dtype=np.float32)

    return dict(
        ca=ca_arr,
        cb=cb_arr,
        ia=ia_arr,
        ib=ib_arr,
        fa=fa_arr,
        fb=fb_arr,
        dist=dist,
        d_norm=d_norm.astype(np.float32),
        decay=decay,
        fb_a=fb_a,
        fb_b=fb_b,
        sol_a=sol_a,
        sol_b=sol_b,
        n_cells=len(cell_ids),
        d_max=float(d_max),
    )


def _cost_matrix(
    geo: dict,
    sig_type: str,
    s_mask: np.ndarray,
    r_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sparse cost arrays for OT between sender and receiver cells.

    Only edges where cell_a ∈ senders AND cell_b ∈ receivers are kept.
    Returns (ia, ib, cost, edge_mask) — COO format for the contact pairs.

    Cost definitions
    ----------------
    juxtacrine: C = 1 - fraction_a
        → full membrane commitment = cost 0 (ideal juxtacrine contact)
        → no contact membrane = cost 1

    secreted:   C = (1 - free_boundary_b) × d_norm
        → receiver shielding × distance
        → isolated receiver far away = high cost

    ecm:        C = solidity_b × d_norm
        → irregular (low solidity) receivers have more ECM surface
        → compact round cell far away = high cost
    """
    ia, ib = geo["ia"], geo["ib"]

    # Restrict to sender→receiver edges
    edge_ok = s_mask[ia] & r_mask[ib]

    ia_ok = ia[edge_ok]
    ib_ok = ib[edge_ok]

    if sig_type == "juxtacrine":
        cost = (1.0 - geo["fa"][edge_ok]).clip(0.0, 1.0)
    elif sig_type == "secreted":
        cost = ((1.0 - geo["fb_b"][edge_ok]) * geo["d_norm"][edge_ok]).clip(0.0, 1.0)
    else:  # ecm
        cost = (geo["sol_b"][edge_ok] * geo["d_norm"][edge_ok]).clip(0.0, 1.0)

    return ia_ok, ib_ok, cost.astype(np.float32), edge_ok


def _geo_weight(
    geo: dict,
    sig_type: str,
    s_mask: np.ndarray,
    r_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geo_weight for message passing, restricted to sender→receiver edges.

    Weight definitions
    ------------------
    juxtacrine: w = fraction_a × fraction_b  (mutual membrane commitment)
    secreted:   w = free_boundary_b × decay  (receiver exposure × distance decay)
    ecm:        w = (1-solidity_b) × decay   (receiver ECM surface × decay)

    Returns (ia_ok, ib_ok, weight) in COO format.
    """
    ia, ib = geo["ia"], geo["ib"]
    edge_ok = s_mask[ia] & r_mask[ib]

    ia_ok = ia[edge_ok]
    ib_ok = ib[edge_ok]

    if sig_type == "juxtacrine":
        w = (geo["fa"][edge_ok] * geo["fb"][edge_ok]).astype(np.float32)
    elif sig_type == "secreted":
        w = (geo["fb_b"][edge_ok] * geo["decay"][edge_ok]).astype(np.float32)
    else:  # ecm
        w = ((1.0 - geo["sol_b"][edge_ok]) * geo["decay"][edge_ok]).astype(np.float32)

    return ia_ok, ib_ok, w


# ── Sinkhorn OT ───────────────────────────────────────────────────────────────


def _logsumexp_scatter(
    log_vals: np.ndarray,
    idx: np.ndarray,
    n: int,
    max_buf: np.ndarray,
    sexp_buf: np.ndarray,
) -> np.ndarray:
    """
    Vectorized logsumexp scatter: result[k] = log Σ_{e: idx[e]==k} exp(log_vals[e])

    Numerically stable three-step implementation — no Python loop.
    Uses pre-allocated buffers to avoid allocation inside the Sinkhorn loop.

    Parameters
    ----------
    log_vals : (n_edges,) values in log-space to scatter
    idx      : (n_edges,) target cell index per edge
    n        : number of cells
    max_buf, sexp_buf : (n,) pre-allocated scratch arrays
    """
    # Step 1: per-cell max for numerical stability
    max_buf[:] = -np.inf
    np.maximum.at(max_buf, idx, log_vals)

    # Step 2: scatter-sum shifted exponentials
    sexp_buf[:] = 0.0
    np.add.at(sexp_buf, idx, np.exp(log_vals - max_buf[idx]))

    # Step 3: log(sum_exp) + max  (only cells that received at least one edge)
    result = np.full(n, -np.inf, dtype=np.float64)
    has_edges = max_buf > -np.inf
    result[has_edges] = np.log(sexp_buf[has_edges]) + max_buf[has_edges]
    return result


def _sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    cost: np.ndarray,
    n: int,
    reg: float = 0.1,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sinkhorn algorithm for entropy-regularized OT on sparse contact edges.

    Minimizes Σ_{(i,j)∈edges} T[i,j] × C[i,j] - reg × H(T)
    where H(T) = -Σ T log T is the entropy regularizer.

    Vectorized: replaces the Python for-loop logsumexp scatter with
    _logsumexp_scatter (numpy maximum.at + add.at), eliminating all
    Python interpreter overhead inside the iteration loop.
    Pre-allocated scratch buffers avoid per-iteration memory allocation.

    Parameters
    ----------
    a    : (n,) supply — ligand expression at senders, normalized to sum=1
    b    : (n,) demand — receptor expression at receivers, normalized to sum=1
    ia   : edge row indices (senders)
    ib   : edge col indices (receivers)
    cost : edge costs C[i,j]
    n    : total number of cells
    reg  : entropy regularization (larger = smoother transport)
    n_iter, tol : convergence settings

    Returns
    -------
    T_vals         : transport plan values on each edge
    sender_score   : Σ_j T[i,j] per sender cell
    receiver_score : Σ_i T[i,j] per receiver cell
    """
    n_edges = len(ia)
    if n_edges == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    log_K = (-cost / reg).astype(np.float64)  # (n_edges,)

    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(n, dtype=np.float64)

    a_safe = np.where(a > 0, a, 1e-12).astype(np.float64)
    b_safe = np.where(b > 0, b, 1e-12).astype(np.float64)
    log_a = np.log(a_safe)
    log_b = np.log(b_safe)

    # Pre-allocate scratch buffers — reused every iteration, no per-iter alloc
    max_buf = np.empty(n, dtype=np.float64)
    sexp_buf = np.empty(n, dtype=np.float64)

    for _ in range(n_iter):
        log_u_old = log_u.copy()

        # v update: log_v[j] = log_b[j] - logsumexp_i(log_u[i] + log_K[e])
        log_uK = log_u[ia] + log_K
        log_v = log_b - _logsumexp_scatter(log_uK, ib, n, max_buf, sexp_buf)

        # u update: log_u[i] = log_a[i] - logsumexp_j(log_v[j] + log_K[e])
        log_vK = log_v[ib] + log_K
        log_u = log_a - _logsumexp_scatter(log_vK, ia, n, max_buf, sexp_buf)

        if np.max(np.abs(log_u - log_u_old)) < tol:
            break

    # Recover transport plan and marginals
    T_log = log_u[ia] + log_K + log_v[ib]
    T_vals = np.exp(T_log).clip(0.0).astype(np.float32)

    sender_score = np.zeros(n, dtype=np.float32)
    receiver_score = np.zeros(n, dtype=np.float32)
    np.add.at(sender_score, ia, T_vals)
    np.add.at(receiver_score, ib, T_vals)

    return T_vals, sender_score, receiver_score


# ── Message Passing ───────────────────────────────────────────────────────────


def _message_passing(
    L: np.ndarray,
    R: np.ndarray,
    ia: np.ndarray,
    ib: np.ndarray,
    w: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Message passing signal computation.

    m[i,j] = sqrt(L[i] * R[j]) * w[i,j]
    Signal[j]  = Σ_i m[i,j]   (total stimulation received by j)
    Outgoing[i] = Σ_j m[i,j]  (total signal sent by i)

    sqrt(L*R) = geometric mean — penalizes imbalance, natural for log-scale.

    Returns
    -------
    m_vals   : message values per edge
    signal   : Σ_i m[i,j] per receiver cell (length n)
    outgoing : Σ_j m[i,j] per sender cell   (length n)
    """
    if len(ia) == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    L_i = np.maximum(L[ia], 0.0).astype(np.float32)
    R_j = np.maximum(R[ib], 0.0).astype(np.float32)
    m_vals = np.sqrt(L_i * R_j) * w

    signal = np.zeros(n, dtype=np.float32)
    outgoing = np.zeros(n, dtype=np.float32)
    np.add.at(signal, ib, m_vals)
    np.add.at(outgoing, ia, m_vals)

    return m_vals, signal, outgoing


# ── Hub detection ─────────────────────────────────────────────────────────────


def _detect_hubs(
    scores: np.ndarray,
    cell_ids: np.ndarray,
    mask: np.ndarray,
    percentile: float = 90.0,
) -> np.ndarray:
    """
    Return cell IDs in mask whose score exceeds the given percentile.

    Percentile computed only over cells in mask (not all cells).
    Returns empty array if no cells pass.
    """
    masked_scores = scores[mask]
    if masked_scores.sum() < 1e-10:
        return np.array([], dtype=cell_ids.dtype)
    threshold = np.percentile(masked_scores, percentile)
    hub_mask = mask & (scores >= threshold)
    return cell_ids[hub_mask]


# ── Per-pair scoring ──────────────────────────────────────────────────────────


def _score_one_pair(
    pair: LRPair,
    sig_type: str,
    sender_type: str,
    recv_type: str,
    L: np.ndarray,
    R: np.ndarray,
    geo: dict,
    type_masks: dict[str, np.ndarray],
    cell_ids: np.ndarray,
    ot_reg: float,
    ot_iter: int,
    hub_pct: float,
) -> dict | None:
    """Score one (lr_pair, sender_type, receiver_type) combination."""
    n = geo["n_cells"]
    s_mask = type_masks.get(sender_type)
    r_mask = type_masks.get(recv_type)
    if s_mask is None or r_mask is None:
        return None

    # ── OT ────────────────────────────────────────────────────────────────────
    ia_ot, ib_ot, cost, _ = _cost_matrix(geo, sig_type, s_mask, r_mask)
    if len(ia_ot) == 0:
        return None

    # Supply: sender ligand expression (restricted to sender cells, normalized)
    a = np.zeros(n, dtype=np.float32)
    a[s_mask] = np.maximum(L[s_mask], 0.0)
    a_sum = a.sum()
    if a_sum < 1e-10:
        return None
    a /= a_sum

    # Demand: receiver receptor expression (restricted to receiver cells, normalized)
    b = np.zeros(n, dtype=np.float32)
    b[r_mask] = np.maximum(R[r_mask], 0.0)
    b_sum = b.sum()
    if b_sum < 1e-10:
        return None
    b /= b_sum

    T_vals, sender_ot, receiver_ot = _sinkhorn(a, b, ia_ot, ib_ot, cost, n, reg=ot_reg, n_iter=ot_iter)

    # ── Message Passing ───────────────────────────────────────────────────────
    ia_mp, ib_mp, w_mp = _geo_weight(geo, sig_type, s_mask, r_mask)
    m_vals, signal_mp, outgoing_mp = _message_passing(L, R, ia_mp, ib_mp, w_mp, n)

    # ── Combined score: geometric mean of normalized OT and MP ───────────────
    # Normalize each to [0,1] over all cells
    def _norm01(v: np.ndarray) -> np.ndarray:
        vmax = v.max()
        return v / vmax if vmax > 1e-10 else v

    sender_combined = np.sqrt(_norm01(sender_ot) * _norm01(outgoing_mp))
    receiver_combined = np.sqrt(_norm01(receiver_ot) * _norm01(signal_mp))

    # ── Hub detection ─────────────────────────────────────────────────────────
    hub_senders = _detect_hubs(sender_combined, cell_ids, s_mask, hub_pct)
    hub_receivers = _detect_hubs(receiver_combined, cell_ids, r_mask, hub_pct)

    # ── Total communication strength scalars ─────────────────────────────────
    ot_strength = float(T_vals.sum())
    mp_strength = float(m_vals.sum())
    combined_strength = float(
        np.sqrt(
            (sender_combined[s_mask].mean() if s_mask.sum() > 0 else 0.0)
            * (receiver_combined[r_mask].mean() if r_mask.sum() > 0 else 0.0)
        )
    )

    return {
        "lr_name": pair.lr_name,
        "ligand": pair.ligand,
        "receptor": pair.receptor,
        "lr_type": pair.lr_type,
        "sig_type": sig_type,
        "pathway": pair.pathway,
        "sender_type": sender_type,
        "receiver_type": recv_type,
        # Scalar summaries
        "ot_strength": ot_strength,
        "mp_strength": mp_strength,
        "combined_strength": combined_strength,
        # Per-cell arrays (stored separately in layer2_cells dict)
        "_sender_ot": sender_ot,
        "_receiver_ot": receiver_ot,
        "_outgoing_mp": outgoing_mp,
        "_signal_mp": signal_mp,
        "_sender_combined": sender_combined,
        "_receiver_combined": receiver_combined,
        # Hubs
        "hub_senders": hub_senders,
        "hub_receivers": hub_receivers,
        "n_hub_senders": len(hub_senders),
        "n_hub_receivers": len(hub_receivers),
        # Edge-level arrays (for Layer 3 tensor)
        "_T_vals": T_vals,
        "_m_vals": m_vals,
        "_ia_ot": ia_ot,
        "_ib_ot": ib_ot,
        "_ia_mp": ia_mp,
        "_ib_mp": ib_mp,
    }


# ── Public entry point ────────────────────────────────────────────────────────


def run_layer2(
    sp: spatioloji,
    layer1_results: dict,
    contact_frac_df: pd.DataFrame,
    sigma: float | None = None,
    coord_type: str = "global",
    ot_reg: float = 0.1,
    ot_iter: int = 50,
    hub_percentile: float = 90.0,
    top_k: int | None = None,
) -> dict:
    """
    Run Layer 2: cell-pair level scoring for all significant LR pairs.

    For each (lr_name, sender_type, receiver_type) from Layer 1:
      1. Polygon OT  — transport plan T[i,j], sender/receiver scores
      2. Message Passing — m[i,j], Signal[j], Outgoing[i]
      3. Combined score  — geometric mean of normalized OT and MP
      4. Hub detection   — top hub_percentile% by combined score

    Parameters
    ----------
    sp : spatioloji
        spatioloji object (same FOV as Layer 1).
    layer1_results : dict
        Output of run_layer1().
    contact_frac_df : pd.DataFrame
        Output of boundaries.contact_fraction().
    sigma : float | None
        Distance decay parameter for secreted/ecm weights.
        None → estimated from median contact distance (recommended).
    coord_type : str
        'global' or 'local' — must match Layer 1.
    ot_reg : float
        Sinkhorn entropy regularization. Larger = smoother. Default 0.1.
    ot_iter : int
        Max Sinkhorn iterations. Default 50.
    hub_percentile : float
        Percentile threshold for hub cell detection. Default 90.
    top_k : int | None
        Limit to top-K pairs from Layer 1 by layer1_score.
        None = all significant pairs.

    Returns
    -------
    dict with keys:
        'scores'       : pd.DataFrame — one row per (lr, sender, receiver)
        'cell_scores'  : dict[str, pd.DataFrame]
                         key = lr_name, value = per-cell sender/receiver scores
        'edge_data'    : list[dict]  — raw OT/MP edge arrays (for Layer 3)
        'geo'          : dict        — geometry arrays (reused by Layer 3)
        'cell_ids'     : np.ndarray

    Examples
    --------
    >>> layer2_results = run_layer2(sp_fov, layer1_results, cf_df)
    >>> layer2_results['scores'].head()
    """
    print("\n" + "=" * 60)
    print("LAYER 2: Cell-Pair Scoring (OT + Message Passing)")
    print("=" * 60)

    sig = layer1_results["significant_pairs"]
    type_masks = layer1_results["type_masks"]
    expr = layer1_results["expr"]
    gene2idx = layer1_results["gene2idx"]
    cell_ids = np.array(sp.cell_index)

    if sig.empty:
        print("  ⚠ No significant pairs from Layer 1.")
        return {"scores": pd.DataFrame(), "cell_scores": {}, "edge_data": [], "geo": {}, "cell_ids": cell_ids}

    if top_k is not None:
        sig = sig.head(top_k)
        print(f"  Using top {top_k} pairs from Layer 1.")

    # ── Estimate sigma from median contact distance ───────────────────────────
    if sigma is None:
        if coord_type == "global":
            xs, ys = sp.spatial.x_global, sp.spatial.y_global
        else:
            xs, ys = sp.spatial.x_local, sp.spatial.y_local

        coords = pd.DataFrame({"x": xs, "y": ys}, index=cell_ids)
        dists = []
        for _, row in contact_frac_df.head(500).iterrows():
            ca, cb = row["cell_a"], row["cell_b"]
            if ca in coords.index and cb in coords.index:
                dx = coords.loc[ca, "x"] - coords.loc[cb, "x"]
                dy = coords.loc[ca, "y"] - coords.loc[cb, "y"]
                dists.append(np.sqrt(dx**2 + dy**2))
        sigma = float(np.median(dists)) if dists else 100.0
        print(f"  σ (distance decay) = {sigma:.2f} (auto from data)")

    # ── Pre-compute geometry arrays (shared across all LR pairs) ─────────────
    print("  Building geometry arrays...")
    geo = _build_geometry_arrays(sp, contact_frac_df, sigma, coord_type)
    print(f"  {len(geo['ia'])} valid contact edges")

    # ── Score each (lr_name, sender_type, receiver_type) row ─────────────────
    score_records: list[dict] = []
    cell_scores: dict = {}  # lr_name → pd.DataFrame
    edge_data: list[dict] = []  # for Layer 3

    # Build LR pair lookup
    # sig may have multiple rows per lr_name (different type pairs)
    # We extract expression once per lr_name
    pair_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    n_rows = len(sig)
    print(f"\n  Scoring {n_rows} (lr_pair × type_pair) combinations...")

    for row_idx, row in sig.iterrows():
        lr_name = row["lr_name"]
        sender_type = row["sender_type"]
        recv_type = row["receiver_type"]
        sig_type = row.get("sig_type", row["lr_type"])

        # Get expression (cached per lr_name)
        if lr_name not in pair_cache:
            # Reconstruct LRPair minimally for _get_expr_vector
            ligand_genes = str(row["ligand"]).split("|")
            receptor_genes = str(row["receptor"]).split("|")
            L = _get_expr_vector(expr, gene2idx, ligand_genes)
            R = _get_expr_vector(expr, gene2idx, receptor_genes)
            pair_cache[lr_name] = (L, R)
        L, R = pair_cache[lr_name]

        # Minimal LRPair for _score_one_pair
        pair = LRPair(
            lr_name=lr_name,
            ligand=row["ligand"],
            receptor=row["receptor"],
            pathway=row["pathway"],
            lr_type=row["lr_type"],
        )

        result = _score_one_pair(
            pair,
            sig_type,
            sender_type,
            recv_type,
            L,
            R,
            geo,
            type_masks,
            cell_ids,
            ot_reg,
            ot_iter,
            hub_percentile,
        )

        if result is None:
            continue

        # Separate scalar row from per-cell arrays
        scalar_keys = [k for k in result if not k.startswith("_")]
        scalar_row = {k: result[k] for k in scalar_keys}
        score_records.append(scalar_row)

        # Store per-cell scores (indexed by cell_id)
        if lr_name not in cell_scores:
            cell_scores[lr_name] = {}

        pair_key = f"{sender_type}_to_{recv_type}"
        cell_scores[lr_name][pair_key] = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "sender_ot": result["_sender_ot"],
                "receiver_ot": result["_receiver_ot"],
                "outgoing_mp": result["_outgoing_mp"],
                "signal_mp": result["_signal_mp"],
                "sender_combined": result["_sender_combined"],
                "receiver_combined": result["_receiver_combined"],
            }
        )

        # Store edge data for Layer 3 tensor construction
        edge_data.append(
            {
                "lr_name": lr_name,
                "sender_type": sender_type,
                "receiver_type": recv_type,
                "sig_type": sig_type,
                "combined_strength": result["combined_strength"],
                "ia": result["_ia_mp"],
                "ib": result["_ib_mp"],
                "m_vals": result["_m_vals"],
                "T_vals": result["_T_vals"],
                "ia_ot": result["_ia_ot"],
                "ib_ot": result["_ib_ot"],
            }
        )

        if (row_idx + 1) % 20 == 0:
            print(f"  [{row_idx + 1}/{n_rows}] {lr_name} {sender_type}→{recv_type} done")

    scores_df = pd.DataFrame(score_records).sort_values("combined_strength", ascending=False).reset_index(drop=True)

    print(f"\n[Layer 2] Done. {len(scores_df)} combinations scored.")
    if len(scores_df):
        print(
            f"  combined_strength range: "
            f"{scores_df['combined_strength'].min():.4f} – "
            f"{scores_df['combined_strength'].max():.4f}"
        )

    return {
        "scores": scores_df,
        "cell_scores": cell_scores,
        "edge_data": edge_data,
        "geo": geo,
        "cell_ids": cell_ids,
    }
