"""
ripley.py - Ripley's spatial statistics (K, L, cross-K, cross-L)

Classic point process statistics for analyzing spatial clustering
and dispersion patterns. Works directly from cell coordinates
without requiring a pre-built graph.
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


def ripleys_k(
    sj: spatioloji,
    coord_type: str = "global",
    cell_ids: list[str] | None = None,
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
) -> RipleyResult:
    """
    Compute Ripley's K function.

    K(r) counts the average number of neighbors within distance r,
    normalized by overall density. Compares to CSR expectation (pi*r^2).

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

    Returns
    -------
    RipleyResult
        K function values and CSR expectation.
    """
    coords, area, bbox = _prepare_coords(sj, coord_type, cell_ids)
    n = len(coords)

    if max_r is None:
        shortest_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        max_r = shortest_side * 0.25

    r_values = np.linspace(0, max_r, n_steps + 1)[1:]  # skip r=0
    # lambda_density = n / area

    # Build KDTree for efficient distance queries
    tree = KDTree(coords)

    # For each r, count pairs within that distance
    K_values = np.zeros(len(r_values))

    for ri, r in enumerate(r_values):
        # Count all pairs within distance r
        count = tree.query_pairs(r, output_type="ndarray")
        n_pairs = len(count)

        if correction == "ripley" and n_pairs > 0:
            # Ripley isotropic edge correction
            # Weight each pair by 1 / (fraction of circle inside bbox)
            weighted_count = 0.0
            for idx_i, idx_j in count:
                d = np.sqrt(np.sum((coords[idx_i] - coords[idx_j]) ** 2))
                w_i = _edge_correction_weight(coords[idx_i], d, bbox)
                w_j = _edge_correction_weight(coords[idx_j], d, bbox)
                weighted_count += 1.0 / w_i + 1.0 / w_j
            K_values[ri] = area * weighted_count / (n * (n - 1)) / 2
        else:
            # No correction: simple count
            K_values[ri] = area * 2 * n_pairs / (n * (n - 1))

    csr_expected = np.pi * r_values**2

    print(f"  ✓ Ripley's K: n={n}, max_r={max_r:.1f}, " f"correction={correction}")

    return RipleyResult(
        r=r_values,
        statistic=K_values,
        csr_expected=csr_expected,
        function_type="K",
        correction=correction,
        label="all_cells",
    )


def _prepare_coords(
    sj: spatioloji,
    coord_type: str,
    cell_ids: list[str] | None,
) -> tuple[np.ndarray, float, tuple[float, float, float, float]]:
    """
    Extract coordinates, compute area and bounding box.

    Returns
    -------
    coords : np.ndarray (n, 2)
    area : float
    bbox : tuple (xmin, ymin, xmax, ymax)
    """
    coords = sj.get_spatial_coords(cell_ids=cell_ids, coord_type=coord_type)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    bbox = (xmin, ymin, xmax, ymax)
    area = (xmax - xmin) * (ymax - ymin)

    if area == 0:
        raise ValueError("Bounding box has zero area — cells are collinear")

    return coords, area, bbox


def _edge_correction_weight(
    point: np.ndarray,
    d: float,
    bbox: tuple[float, float, float, float],
) -> float:
    """
    Ripley isotropic edge correction weight.

    Estimates the fraction of the circle centered at `point` with
    radius `d` that falls inside the bounding box. Uses the
    proportion-of-circumference approximation.

    Parameters
    ----------
    point : np.ndarray (2,)
    d : float, distance (radius)
    bbox : (xmin, ymin, xmax, ymax)

    Returns
    -------
    float
        Weight in (0, 1]. 1.0 = entire circle inside bbox.
    """
    if d == 0:
        return 1.0

    xmin, ymin, xmax, ymax = bbox
    px, py = point

    # Distance from point to each edge
    dx_min = px - xmin
    dx_max = xmax - px
    dy_min = py - ymin
    dy_max = ymax - py

    # Fraction of circle outside each edge
    # A circle extends beyond an edge if d > distance_to_edge
    # The "lost" fraction per edge ≈ arccos(dist/d) / pi
    fraction_inside = 1.0

    for dist_to_edge in [dx_min, dx_max, dy_min, dy_max]:
        if d > dist_to_edge and dist_to_edge >= 0:
            fraction_inside -= np.arccos(np.clip(dist_to_edge / d, -1, 1)) / (2 * np.pi)

    return max(fraction_inside, 0.01)  # floor to avoid division by ~0


def ripleys_l(
    sj: spatioloji,
    coord_type: str = "global",
    cell_ids: list[str] | None = None,
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
) -> RipleyResult:
    """
    Compute Ripley's L function (variance-stabilized K).

    L(r) = sqrt(K(r)/pi) - r.
    Under CSR, L(r) = 0 for all r. Positive = clustering,
    negative = dispersion. Much easier to interpret than raw K.

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

    Returns
    -------
    RipleyResult
        L function values (CSR expectation = 0).
    """
    # Compute K first
    k_result = ripleys_k(
        sj,
        coord_type=coord_type,
        cell_ids=cell_ids,
        max_r=max_r,
        n_steps=n_steps,
        correction=correction,
    )

    # Transform: L(r) = sqrt(K(r)/pi) - r
    L_values = np.sqrt(k_result.statistic / np.pi) - k_result.r
    csr_expected = np.zeros_like(k_result.r)  # L = 0 under CSR

    print(f"  ✓ Ripley's L: max deviation = {np.max(np.abs(L_values)):.4f}")

    return RipleyResult(
        r=k_result.r,
        statistic=L_values,
        csr_expected=csr_expected,
        function_type="L",
        correction=correction,
        label="all_cells",
    )


def cross_k(
    sj: spatioloji,
    cell_type_col: str,
    type_a: str,
    type_b: str,
    coord_type: str = "global",
    max_r: float | None = None,
    n_steps: int = 50,
    correction: str = "ripley",
) -> RipleyResult:
    """
    Compute cross-K function between two cell types.

    Measures whether cells of type_a are clustered around cells
    of type_b at each distance r. Asymmetric: K_AB != K_BA.

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
    n_a = len(coords_a)
    n_b = len(coords_b)

    if n_a == 0 or n_b == 0:
        raise ValueError(f"No cells found for type_a='{type_a}' " f"({n_a}) or type_b='{type_b}' ({n_b})")

    # Bounding box from all cells
    xmin, ymin = all_coords.min(axis=0)
    xmax, ymax = all_coords.max(axis=0)
    bbox = (xmin, ymin, xmax, ymax)
    area = (xmax - xmin) * (ymax - ymin)

    if max_r is None:
        shortest_side = min(xmax - xmin, ymax - ymin)
        max_r = shortest_side * 0.25

    r_values = np.linspace(0, max_r, n_steps + 1)[1:]

    # Build KDTree on type_b (targets)
    tree_b = KDTree(coords_b)

    # For each cell in type_a, count type_b neighbors at each r
    K_values = np.zeros(len(r_values))

    for ri, r in enumerate(r_values):
        total_weighted = 0.0

        for i in range(n_a):
            # Find type_b neighbors within r of this type_a cell
            neighbors = tree_b.query_ball_point(coords_a[i], r)

            if correction == "ripley":
                for j_idx in neighbors:
                    d = np.sqrt(np.sum((coords_a[i] - coords_b[j_idx]) ** 2))
                    if d > 0:
                        w = _edge_correction_weight(coords_a[i], d, bbox)
                        total_weighted += 1.0 / w
            else:
                total_weighted += len(neighbors)

        lambda_b = n_b / area
        K_values[ri] = total_weighted / (n_a * lambda_b)

    csr_expected = np.pi * r_values**2

    print(f"  ✓ Cross-K ({type_a}→{type_b}): n_a={n_a}, n_b={n_b}, " f"max_r={max_r:.1f}")

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
    )

    L_values = np.sqrt(k_result.statistic / np.pi) - k_result.r
    csr_expected = np.zeros_like(k_result.r)

    print(f"  ✓ Cross-L ({type_a}→{type_b}): " f"max deviation = {np.max(np.abs(L_values)):.4f}")

    return RipleyResult(
        r=k_result.r,
        statistic=L_values,
        csr_expected=csr_expected,
        function_type="cross-L",
        correction=k_result.correction,
        label=f"{type_a}→{type_b}",
    )


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
    seed: int | None = None,
) -> RipleyResult:
    """
    Compute simulation envelopes for significance testing.

    For K/L: simulates complete spatial randomness (uniform points).
    For cross-K/L: permutes cell type labels while keeping positions.

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
    seed : int, optional
        Random seed.

    Returns
    -------
    RipleyResult
        Observed statistic with envelope_lo and envelope_hi filled.
    """
    rng = np.random.default_rng(seed)
    is_cross = function.startswith("cross")

    # Validate inputs for cross functions
    if is_cross:
        if cell_type_col is None or type_a is None or type_b is None:
            raise ValueError("cross functions require cell_type_col, type_a, and type_b")

    # Step 1: Compute observed
    if function == "K":
        observed = ripleys_k(sj, coord_type, cell_ids, max_r, n_steps, correction)
    elif function == "L":
        observed = ripleys_l(sj, coord_type, cell_ids, max_r, n_steps, correction)
    elif function == "cross-K":
        observed = cross_k(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction)
    elif function == "cross-L":
        observed = cross_l(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction)
    else:
        raise ValueError(f"Unknown function: {function}")

    r_values = observed.r
    n_r = len(r_values)

    # Step 2: Simulate
    print(f"  Running {n_simulations} simulations for envelope...")
    sim_stats = np.zeros((n_simulations, n_r))

    if not is_cross:
        # CSR simulation: scatter random points in bounding box
        coords, area, bbox = _prepare_coords(sj, coord_type, cell_ids)
        n_points = len(coords)
        xmin, ymin, xmax, ymax = bbox

        for s in range(n_simulations):
            # Generate random points
            rand_x = rng.uniform(xmin, xmax, n_points)
            rand_y = rng.uniform(ymin, ymax, n_points)
            rand_coords = np.column_stack([rand_x, rand_y])

            # Compute statistic for random points
            sim_result = _compute_k_from_coords(rand_coords, bbox, area, r_values, correction)

            if function == "L":
                sim_stats[s] = np.sqrt(sim_result / np.pi) - r_values
            else:
                sim_stats[s] = sim_result
    else:
        # Label permutation: keep positions, shuffle types
        labels = sj.cell_meta[cell_type_col].values.copy()

        for s in range(n_simulations):
            # Shuffle labels
            shuffled = rng.permutation(labels)
            sj.cell_meta[cell_type_col] = shuffled

            if function == "cross-K":
                sim = cross_k(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction)
                sim_stats[s] = sim.statistic
            else:
                sim = cross_l(sj, cell_type_col, type_a, type_b, coord_type, max_r, n_steps, correction)
                sim_stats[s] = sim.statistic

        # Restore original labels
        sj.cell_meta[cell_type_col] = labels

    # Step 3: Build envelope
    alpha = 1 - confidence
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    envelope_lo = np.percentile(sim_stats, lo_pct, axis=0)
    envelope_hi = np.percentile(sim_stats, hi_pct, axis=0)

    # Report
    above = np.sum(observed.statistic > envelope_hi)
    below = np.sum(observed.statistic < envelope_lo)
    print(
        f"  ✓ Envelope ({confidence:.0%}): " f"{above} distances above, {below} below, " f"{n_r - above - below} inside"
    )

    # Return observed with envelope attached
    observed.envelope_lo = envelope_lo
    observed.envelope_hi = envelope_hi

    return observed


def _compute_k_from_coords(
    coords: np.ndarray,
    bbox: tuple[float, float, float, float],
    area: float,
    r_values: np.ndarray,
    correction: str,
) -> np.ndarray:
    """
    Compute K(r) directly from coordinate array.
    Used internally for CSR simulations.
    """
    n = len(coords)
    tree = KDTree(coords)
    K_values = np.zeros(len(r_values))

    for ri, r in enumerate(r_values):
        pairs = tree.query_pairs(r, output_type="ndarray")
        n_pairs = len(pairs)

        if correction == "ripley" and n_pairs > 0:
            weighted = 0.0
            for idx_i, idx_j in pairs:
                d = np.sqrt(np.sum((coords[idx_i] - coords[idx_j]) ** 2))
                w_i = _edge_correction_weight(coords[idx_i], d, bbox)
                w_j = _edge_correction_weight(coords[idx_j], d, bbox)
                weighted += 1.0 / w_i + 1.0 / w_j
            K_values[ri] = area * weighted / (n * (n - 1)) / 2
        else:
            K_values[ri] = area * 2 * n_pairs / (n * (n - 1))

    return K_values
