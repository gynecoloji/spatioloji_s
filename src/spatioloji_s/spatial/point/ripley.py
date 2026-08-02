"""
ripley.py - Ripley's spatial statistics (K, L, cross-K, cross-L)

Classic point process statistics for analyzing spatial clustering
and dispersion patterns. Works directly from cell coordinates
without requiring a pre-built graph.

Univariate K/L and simulation envelopes are delegated to ``pointpats``
(PySAL) for optimised, parallel computation.  Cross-K/L are computed
internally with vectorised edge correction (pointpats does not support
bivariate functions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


@dataclass
class RipleyResult:
    """
    Container for Ripley's statistic results.

    Attributes
    ----------
    r : np.ndarray
        Distance values where statistic was evaluated.
    statistic : np.ndarray
        K(r) or L(r) values.
    csr_expected : np.ndarray
        Expected values under complete spatial randomness.
    function_type : str
        'K' or 'L'.
    correction : str
        Edge correction method used.
    envelope_lo : np.ndarray or None
        Lower simulation envelope (if computed).
    envelope_hi : np.ndarray or None
        Upper simulation envelope (if computed).
    label : str
        Description (e.g., gene name or cell type).
    """

    r: np.ndarray
    statistic: np.ndarray
    csr_expected: np.ndarray
    function_type: str
    correction: str
    envelope_lo: np.ndarray | None = None
    envelope_hi: np.ndarray | None = None
    label: str = ""

    @property
    def deviation(self) -> np.ndarray:
        """Difference from CSR expectation."""
        return self.statistic - self.csr_expected

    def summary(self) -> dict:
        dev = self.deviation
        return {
            "function": self.function_type,
            "label": self.label,
            "correction": self.correction,
            "max_r": self.r.max(),
            "n_distances": len(self.r),
            "max_deviation": dev.max(),
            "min_deviation": dev.min(),
            "has_envelope": self.envelope_lo is not None,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"RipleyResult({s['function']}, label={s['label']}, "
            f"max_r={s['max_r']:.1f}, "
            f"max_dev={s['max_deviation']:.2f})"
        )


# ---------------------------------------------------------------------------
# Helper: import pointpats lazily
# ---------------------------------------------------------------------------


def _import_pointpats():
    """Lazy import of pointpats with actionable error message."""
    try:
        import pointpats
    except ImportError as err:
        raise ImportError(
            "pointpats is required for Ripley's statistics. Install with: pip install spatioloji_s[ripley]"
        ) from err
    return pointpats


# ---------------------------------------------------------------------------
# Coordinate / bounding-box helpers
# ---------------------------------------------------------------------------


def _prepare_coords(
    sj: spatioloji,
    coord_type: str,
    cell_ids: list[str] | None,
    max_cells: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, float, tuple[float, float, float, float]]:
    """
    Extract coordinates, compute area and bounding box.

    Parameters
    ----------
    sj : spatioloji
    coord_type : str
    cell_ids : list or None
    max_cells : int or None
        If set and n_cells exceeds this, randomly subsample to ``max_cells``
        points.  The bounding box is computed from the *full* point set before
        subsampling so that the study-area estimate remains unbiased.
    seed : int or None
        Random seed for reproducible subsampling.

    Returns
    -------
    coords : np.ndarray (n, 2)
    area : float
    bbox : tuple (xmin, ymin, xmax, ymax)
    """
    coords = sj.get_spatial_coords(cell_ids=cell_ids, coord_type=coord_type)

    # Bounding box from full point set (before any subsampling)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    bbox = (xmin, ymin, xmax, ymax)
    area = (xmax - xmin) * (ymax - ymin)

    if area == 0:
        raise ValueError("Bounding box has zero area — cells are collinear")

    # Subsample if needed
    if max_cells is not None and len(coords) > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_cells, replace=False)
        print(f"    Subsampled {len(coords)} → {max_cells} cells for Ripley's statistic")
        coords = coords[idx]

    return coords, area, bbox


def _make_support(max_r: float | None, n_steps: int, bbox: tuple[float, float, float, float]) -> tuple:
    """
    Build a pointpats-compatible ``support`` tuple.

    Parameters
    ----------
    max_r : float or None
        Maximum radius.  If None, 25 % of shortest bbox side.
    n_steps : int
        Number of distance bins.
    bbox : (xmin, ymin, xmax, ymax)

    Returns
    -------
    support : tuple
        ``(0, max_r, n_steps)`` for pointpats.
    max_r : float
        Resolved max_r value.
    """
    if max_r is None:
        shortest_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        max_r = shortest_side * 0.25
    return (0, max_r, n_steps), max_r


# ---------------------------------------------------------------------------
# Univariate K / L  —  delegated to pointpats
# ---------------------------------------------------------------------------


def ripleys_k(
    sj: spatioloji,
    coord_type: str = "global",
    cell_ids: list[str] | None = None,
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
    max_cells: int | None = 10000,
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute Ripley's K function.

    K(r) counts the average number of neighbors within distance r,
    normalized by overall density. Compares to CSR expectation (pi*r^2).

    Uses ``pointpats.k_function`` for fast, vectorised computation.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    coord_type : str
        'global' or 'local'.
    cell_ids : list of str, optional
        Subset of cells. If None, uses all cells.
    max_r : float, optional
        Maximum radius. If None, uses 25% of shortest bounding box side
        (standard rule of thumb to avoid severe edge effects).
    n_steps : int
        Number of distance values to evaluate.
    correction : str
        Edge correction: 'ripley' (isotropic) or 'none'.
        Note: pointpats does not currently implement edge correction;
        the parameter is accepted for API compatibility but only
        uncorrected K is computed.
    max_cells : int or None
        Maximum number of cells. If the point set exceeds this, a random
        subsample is used.  Set to None to disable.  Default 10 000.
    seed : int or None
        Random seed for subsampling reproducibility.

    Returns
    -------
    RipleyResult
        K function values and CSR expectation.
    """
    pp = _import_pointpats()
    coords, area, bbox = _prepare_coords(sj, coord_type, cell_ids, max_cells=max_cells, seed=seed)
    n = len(coords)
    support, max_r = _make_support(max_r, n_steps, bbox)

    r_values, K_values = pp.k_function(coords, support=support)

    csr_expected = np.pi * r_values**2

    print(f"  ✓ Ripley's K: n={n}, max_r={max_r:.1f}, correction={correction}")

    return RipleyResult(
        r=r_values,
        statistic=K_values,
        csr_expected=csr_expected,
        function_type="K",
        correction=correction,
        label="all_cells",
    )


def ripleys_l(
    sj: spatioloji,
    coord_type: str = "global",
    cell_ids: list[str] | None = None,
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
    max_cells: int | None = 10000,
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute Ripley's L function (variance-stabilized K).

    L(r) = sqrt(K(r)/pi) - r.
    Under CSR, L(r) = 0 for all r. Positive = clustering,
    negative = dispersion. Much easier to interpret than raw K.

    Uses ``pointpats.l_function`` (linearized) for fast computation.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    coord_type : str
        'global' or 'local'.
    cell_ids : list of str, optional
        Subset of cells.
    max_r : float, optional
        Maximum radius.
    n_steps : int
        Number of distance values.
    correction : str
        Edge correction method.
    max_cells : int or None
        Maximum number of cells. If the point set exceeds this, a random
        subsample is used.  Set to None to disable.  Default 10 000.
    seed : int or None
        Random seed for subsampling reproducibility.

    Returns
    -------
    RipleyResult
        L function values (CSR expectation = 0).
    """
    pp = _import_pointpats()
    coords, area, bbox = _prepare_coords(sj, coord_type, cell_ids, max_cells=max_cells, seed=seed)
    n = len(coords)
    support, max_r = _make_support(max_r, n_steps, bbox)

    r_values, L_values = pp.l_function(coords, support=support, linearized=True)

    csr_expected = np.zeros_like(r_values)

    print(f"  ✓ Ripley's L: n={n}, max_r={max_r:.1f}, max deviation = {np.max(np.abs(L_values)):.4f}")

    return RipleyResult(
        r=r_values,
        statistic=L_values,
        csr_expected=csr_expected,
        function_type="L",
        correction=correction,
        label="all_cells",
    )


# ---------------------------------------------------------------------------
# Simulation envelope  —  K/L via pointpats, cross via internal
# ---------------------------------------------------------------------------


def simulation_envelope(
    sj: spatioloji,
    function: str = "L",
    coord_type: str = "global",
    cell_ids: list[str] | None = None,
    cell_type_col: str | None = None,
    type_a: str | None = None,
    type_b: str | None = None,
    n_simulations: int = 99,
    confidence: float = 0.95,
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
    max_cells: int | None = 10000,
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute simulation envelopes for significance testing.

    For K/L: uses ``pointpats.k_test`` / ``pointpats.l_test`` with
    parallel Monte-Carlo simulations (n_jobs=-1).
    For cross-K/L: permutes cell type labels while keeping positions
    (internal vectorised implementation).

    The observed curve is significant at distance r if it falls
    outside the envelope at that r.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    function : str
        'K', 'L', 'cross-K', or 'cross-L'.
    coord_type : str
        'global' or 'local'.
    cell_ids : list, optional
        Subset of cells (for K/L).
    cell_type_col : str, optional
        Cell type column (required for cross functions).
    type_a, type_b : str, optional
        Cell types (required for cross functions).
    n_simulations : int
        Number of simulations for envelope.
    confidence : float
        Confidence level for envelope (e.g., 0.95 = 95%).
    max_r : float, optional
        Maximum radius.
    n_steps : int
        Number of distance values.
    correction : str
        Edge correction method.
    max_cells : int or None
        Maximum number of cells before subsampling. Ripley's statistics
        are O(n²), so large point sets are subsampled for tractability.
        The bounding box is computed from the full set before subsampling.
        Set to None to disable. Default 10 000.
    seed : int, optional
        Random seed.

    Returns
    -------
    RipleyResult
        Observed statistic with envelope_lo and envelope_hi filled.
    """
    is_cross = function.startswith("cross")

    if is_cross:
        if cell_type_col is None or type_a is None or type_b is None:
            raise ValueError("cross functions require cell_type_col, type_a, and type_b")
        return _cross_envelope(
            sj,
            function,
            coord_type,
            cell_type_col,
            type_a,
            type_b,
            n_simulations,
            confidence,
            max_r,
            n_steps,
            correction,
            max_cells,
            seed,
        )

    return _univariate_envelope(
        sj,
        function,
        coord_type,
        cell_ids,
        n_simulations,
        confidence,
        max_r,
        n_steps,
        correction,
        max_cells,
        seed,
    )


def _univariate_envelope(
    sj: spatioloji,
    function: str,
    coord_type: str,
    cell_ids: list[str] | None,
    n_simulations: int,
    confidence: float,
    max_r: float | None,
    n_steps: int,
    correction: str,
    max_cells: int | None,
    seed: int | None,
) -> RipleyResult:
    """Envelope for K/L via pointpats (parallel Monte-Carlo)."""
    pp = _import_pointpats()
    coords, _area, bbox = _prepare_coords(sj, coord_type, cell_ids, max_cells=max_cells, seed=seed)
    support, max_r = _make_support(max_r, n_steps, bbox)

    print(f"  Running {n_simulations} simulations for {function} envelope (pointpats, parallel)...")

    if function == "K":
        result = pp.k_test(
            coords,
            support=support,
            keep_simulations=True,
            n_simulations=n_simulations,
            n_jobs=-1,
        )
        statistic = result.statistic
        csr_expected = np.pi * result.support**2
        func_type = "K"
    elif function == "L":
        result = pp.l_test(
            coords,
            support=support,
            linearized=True,
            keep_simulations=True,
            n_simulations=n_simulations,
            n_jobs=-1,
        )
        statistic = result.statistic
        csr_expected = np.zeros_like(result.support)
        func_type = "L"
    else:
        raise ValueError(f"Unknown univariate function: {function}")

    r_values = result.support

    # Build envelope from simulation matrix
    alpha = 1 - confidence
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100
    envelope_lo = np.percentile(result.simulations, lo_pct, axis=0)
    envelope_hi = np.percentile(result.simulations, hi_pct, axis=0)

    above = np.sum(statistic > envelope_hi)
    below = np.sum(statistic < envelope_lo)
    n_r = len(r_values)
    print(f"  ✓ Envelope ({confidence:.0%}): {above} distances above, {below} below, {n_r - above - below} inside")

    return RipleyResult(
        r=r_values,
        statistic=statistic,
        csr_expected=csr_expected,
        function_type=func_type,
        correction=correction,
        label="all_cells",
        envelope_lo=envelope_lo,
        envelope_hi=envelope_hi,
    )


def _cross_envelope(
    sj: spatioloji,
    function: str,
    coord_type: str,
    cell_type_col: str,
    type_a: str,
    type_b: str,
    n_simulations: int,
    confidence: float,
    max_r: float | None,
    n_steps: int,
    correction: str,
    max_cells: int | None,
    seed: int | None,
) -> RipleyResult:
    """Envelope for cross-K/L via label permutation (internal)."""
    rng = np.random.default_rng(seed)

    # Observed
    if function == "cross-K":
        observed = cross_k(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction, max_cells, seed)
    else:
        observed = cross_l(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction, max_cells, seed)

    r_values = observed.r
    n_r = len(r_values)

    print(f"  Running {n_simulations} simulations for {function} envelope (label permutation)...")
    sim_stats = np.zeros((n_simulations, n_r))

    labels = sj.cell_meta[cell_type_col].values.copy()

    for s in range(n_simulations):
        shuffled = rng.permutation(labels)
        sj.cell_meta[cell_type_col] = shuffled

        if function == "cross-K":
            sim = cross_k(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction, max_cells, seed)
        else:
            sim = cross_l(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction, max_cells, seed)
        sim_stats[s] = sim.statistic

    # Restore original labels
    sj.cell_meta[cell_type_col] = labels

    alpha = 1 - confidence
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100
    envelope_lo = np.percentile(sim_stats, lo_pct, axis=0)
    envelope_hi = np.percentile(sim_stats, hi_pct, axis=0)

    above = np.sum(observed.statistic > envelope_hi)
    below = np.sum(observed.statistic < envelope_lo)
    print(f"  ✓ Envelope ({confidence:.0%}): {above} distances above, {below} below, {n_r - above - below} inside")

    observed.envelope_lo = envelope_lo
    observed.envelope_hi = envelope_hi
    return observed


# ---------------------------------------------------------------------------
# Cross-K / Cross-L  —  internal vectorised implementation
# ---------------------------------------------------------------------------


def _edge_correction_weights_vectorized(
    points: np.ndarray,
    distances: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Vectorised Ripley isotropic edge correction weights.

    Estimates the fraction of each circle (centred at each point with
    the corresponding radius) that falls inside the bounding box.

    Parameters
    ----------
    points : np.ndarray (m, 2)
        Point coordinates.
    distances : np.ndarray (m,)
        Radius for each point.
    bbox : (xmin, ymin, xmax, ymax)

    Returns
    -------
    np.ndarray (m,)
        Weights in (0, 1]. 1.0 = entire circle inside bbox.
    """
    xmin, ymin, xmax, ymax = bbox

    # Distance from each point to each edge: shape (m, 4)
    edge_dists = np.column_stack(
        [
            points[:, 0] - xmin,
            xmax - points[:, 0],
            points[:, 1] - ymin,
            ymax - points[:, 1],
        ]
    )

    d = distances[:, np.newaxis]  # (m, 1)

    # Where circle extends beyond an edge
    outside = (d > edge_dists) & (edge_dists >= 0)
    ratios = np.clip(edge_dists / np.maximum(d, 1e-12), -1, 1)

    # Lost fraction per edge = arccos(dist/d) / (2*pi)
    lost = np.where(outside, np.arccos(ratios) / (2 * np.pi), 0.0)
    fraction_inside = 1.0 - lost.sum(axis=1)

    return np.maximum(fraction_inside, 0.01)


def cross_k(
    sj: spatioloji,
    cell_type_col: str,
    type_a: str,
    type_b: str,
    coord_type: str = "global",
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
    max_cells: int | None = 10000,
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute cross-K function between two cell types.

    Measures whether cells of type_a are clustered around cells
    of type_b at each distance r. Asymmetric: K_AB != K_BA.

    Uses a vectorised internal implementation (pointpats does not
    support bivariate / cross-type Ripley functions).

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column in sj.cell_meta with cell type labels.
    type_a : str
        Source cell type ("are these clustered...").
    type_b : str
        Target cell type ("...around these?").
    coord_type : str
        'global' or 'local'.
    max_r : float, optional
        Maximum radius.
    n_steps : int
        Number of distance values.
    correction : str
        Edge correction method.
    max_cells : int or None
        Maximum number of cells per type before subsampling.
        Set to None to disable. Default 10 000.
    seed : int or None
        Random seed for subsampling reproducibility.

    Returns
    -------
    RipleyResult
        Cross-K values. CSR expectation = pi*r^2.
    """
    if cell_type_col not in sj.cell_meta.columns:
        raise ValueError(f"'{cell_type_col}' not found in cell_meta")

    labels = sj.cell_meta[cell_type_col].values
    all_coords = sj.get_spatial_coords(coord_type=coord_type)

    mask_a = labels == type_a
    mask_b = labels == type_b
    coords_a = all_coords[mask_a]
    coords_b = all_coords[mask_b]

    # Subsample each type if needed
    if max_cells is not None:
        rng = np.random.default_rng(seed)
        if len(coords_a) > max_cells:
            idx = rng.choice(len(coords_a), size=max_cells, replace=False)
            coords_a = coords_a[idx]
            print(f"    Subsampled type_a '{type_a}': {mask_a.sum()} → {max_cells}")
        if len(coords_b) > max_cells:
            idx = rng.choice(len(coords_b), size=max_cells, replace=False)
            coords_b = coords_b[idx]
            print(f"    Subsampled type_b '{type_b}': {mask_b.sum()} → {max_cells}")

    n_a = len(coords_a)
    n_b = len(coords_b)

    if n_a == 0 or n_b == 0:
        raise ValueError(f"No cells found for type_a='{type_a}' ({n_a}) or type_b='{type_b}' ({n_b})")

    # Bounding box from all cells (before subsampling)
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    bbox = (xmin, ymin, xmax, ymax)
    area = (xmax - xmin) * (ymax - ymin)

    if max_r is None:
        shortest_side = min(xmax - xmin, ymax - ymin)
        max_r = shortest_side * 0.25

    r_values = np.linspace(0, max_r, n_steps + 1)[1:]

    # Build KDTree on type_b (targets) and query all at max_r once
    tree_b = KDTree(coords_b)

    # Compute all cross-distances at once up to max_r
    # sparse_dist is a csr_matrix (n_a, n_b) with distances <= max_r
    sparse_dist = tree_b.sparse_distance_matrix(KDTree(coords_a), max_r, output_type="coo_matrix")

    # sparse_dist rows = tree_b indices, cols = tree_a indices (coords_a)
    # We want: for each type_a cell (col), its distances to type_b cells (row)
    b_indices = sparse_dist.row
    a_indices = sparse_dist.col
    pair_dists = sparse_dist.data

    # Filter out zero distances (same point shouldn't happen cross-type, but be safe)
    nonzero = pair_dists > 0
    b_indices = b_indices[nonzero]
    a_indices = a_indices[nonzero]
    pair_dists = pair_dists[nonzero]

    lambda_b = n_b / area

    if correction == "ripley" and len(pair_dists) > 0:
        # Vectorised edge correction weights for type_a source points
        source_points = coords_a[a_indices]
        weights = _edge_correction_weights_vectorized(source_points, pair_dists, bbox)

        # For each r, sum 1/weight for all pairs with distance <= r
        K_values = np.zeros(len(r_values))
        for ri, r in enumerate(r_values):
            mask = pair_dists <= r
            K_values[ri] = np.sum(1.0 / weights[mask]) / (n_a * lambda_b)
    else:
        # No correction: count pairs per r bin
        K_values = np.zeros(len(r_values))
        sorted_dists = np.sort(pair_dists)
        counts = np.searchsorted(sorted_dists, r_values, side="right")
        K_values = counts.astype(float) / (n_a * lambda_b)

    csr_expected = np.pi * r_values**2

    print(f"  ✓ Cross-K ({type_a}→{type_b}): n_a={n_a}, n_b={n_b}, max_r={max_r:.1f}")

    return RipleyResult(
        r=r_values,
        statistic=K_values,
        csr_expected=csr_expected,
        function_type="cross-K",
        correction=correction,
        label=f"{type_a}→{type_b}",
    )


def cross_l(
    sj: spatioloji,
    cell_type_col: str,
    type_a: str,
    type_b: str,
    coord_type: str = "global",
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
    max_cells: int | None = 10000,
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute cross-L function (variance-stabilized cross-K).

    L_AB(r) = sqrt(K_AB(r)/pi) - r. Under CSR, L = 0.
    Positive = type_a clusters around type_b.

    Parameters
    ----------
    sj : spatioloji
        spatioloji object.
    cell_type_col : str
        Column with cell type labels.
    type_a, type_b : str
        Source and target cell types.
    coord_type : str
        'global' or 'local'.
    max_r : float, optional
        Maximum radius.
    n_steps : int
        Number of distance values.
    correction : str
        Edge correction method.
    max_cells : int or None
        Maximum number of cells per type before subsampling.
        Set to None to disable. Default 10 000.
    seed : int or None
        Random seed for subsampling reproducibility.

    Returns
    -------
    RipleyResult
        Cross-L values.
    """
    k_result = cross_k(
        sj,
        cell_type_col,
        type_a,
        type_b,
        coord_type=coord_type,
        max_r=max_r,
        n_steps=n_steps,
        correction=correction,
        max_cells=max_cells,
        seed=seed,
    )

    L_values = np.sqrt(k_result.statistic / np.pi) - k_result.r
    csr_expected = np.zeros_like(k_result.r)

    print(f"  ✓ Cross-L ({type_a}→{type_b}): max deviation = {np.max(np.abs(L_values)):.4f}")

    return RipleyResult(
        r=k_result.r,
        statistic=L_values,
        csr_expected=csr_expected,
        function_type="cross-L",
        correction=k_result.correction,
        label=f"{type_a}→{type_b}",
    )
