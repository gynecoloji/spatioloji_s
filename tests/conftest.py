"""
conftest.py - Shared test fixtures for spatioloji

What is this file?
------------------
pytest automatically reads this file before running any test.
Anything defined here (called a "fixture") is available to ALL test files
without needing to import it — pytest injects it automatically.

Think of fixtures as reusable Lego blocks:
  - You build the block once here
  - Every test that needs it just asks for it by name
  - pytest hands it over automatically

How fixtures work:
------------------
    @pytest.fixture          ← decorator that marks a reusable setup block
    def my_fixture():
        return something_useful

    def test_something(my_fixture):   ← pytest sees the name and injects it
        assert my_fixture == expected
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.data.core import spatioloji

# ===========================================================================
# Constants — the size of our fake dataset
# ===========================================================================

N_CELLS = 60  # total fake cells (30 per FOV)
N_GENES = 25  # total fake genes
N_FOVS = 2  # two fake FOVs (field of views)


# ===========================================================================
# Fixture 1: minimal spatioloji object (no polygons, no images)
# ===========================================================================


@pytest.fixture
def sp_basic():
    """
    The simplest possible spatioloji object.

    Use this when a test only needs:
      - expression matrix
      - cell/gene metadata
      - spatial coordinates
      - FOV labels

    60 cells × 25 genes, split across 2 FOVs.
    """

    np.random.seed(42)  # fix random seed → reproducible

    # --- Expression matrix (cells × genes) ---
    # We use integers 0-9 so the matrix is sparse-ish (many zeros)
    expression = np.random.randint(0, 5, (N_CELLS, N_GENES)).astype(float)

    # --- Cell and gene identifiers ---
    cell_ids = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    # --- Cell metadata ---
    # fov: first 30 cells → FOV "1", next 30 → FOV "2"
    # cell_type: randomly assigned A or B
    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
            "cell_type": np.random.choice(["TypeA", "TypeB"], N_CELLS),
            "n_counts": expression.sum(axis=1),  # total counts per cell
        },
        index=cell_ids,
    )

    # --- Gene metadata ---
    # NegProbe: last 2 genes are negative control probes
    gene_meta = pd.DataFrame(
        {
            "NegProbe": [False] * (N_GENES - 2) + [True, True],
        },
        index=gene_names,
    )

    # --- Spatial coordinates ---
    # global x/y: scattered in a 1000×1000 space
    # local x/y:  within each FOV (0–500)
    spatial = {
        "x_global": np.random.uniform(0, 1000, N_CELLS),
        "y_global": np.random.uniform(0, 1000, N_CELLS),
        "x_local": np.random.uniform(0, 500, N_CELLS),
        "y_local": np.random.uniform(0, 500, N_CELLS),
    }

    # --- FOV positions ---
    fov_positions = pd.DataFrame(
        {
            "x_offset": [0, 500],
            "y_offset": [0, 500],
        },
        index=pd.Index(["1", "2"], name="fov"),
    )

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        gene_metadata=gene_meta,
        spatial_coords=spatial,
        fov_positions=fov_positions,
    )


# ===========================================================================
# Fixture 2: spatioloji object WITH polygon data
# ===========================================================================


@pytest.fixture
def sp_with_polygons(sp_basic):
    """
    Same as sp_basic but with fake cell boundary polygons attached.

    Each cell gets a tiny square polygon (4 vertices) around its centroid.
    Use this for any test that touches sp.polygons.

    Builds on sp_basic so we don't duplicate setup code — this is called
    a 'fixture that depends on another fixture'.
    """

    x = sp_basic.spatial.x_global
    y = sp_basic.spatial.y_global
    cell_ids = list(sp_basic.cell_index)

    rows = []
    for cid, cx, cy in zip(cell_ids, x, y, strict=False):
        # A tiny 5×5 square around the centroid — 4 corners + closing vertex
        for vx, vy in [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx + 2, cy + 2),
            (cx - 2, cy + 2),
            (cx - 2, cy - 2),
        ]:  # close the polygon
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})

    polygons = pd.DataFrame(rows)

    # Rebuild spatioloji with polygons attached
    np.random.seed(42)
    expression = np.random.randint(0, 5, (N_CELLS, N_GENES)).astype(float)
    cell_ids_list = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
            "cell_type": np.random.choice(["TypeA", "TypeB"], N_CELLS),
            "n_counts": expression.sum(axis=1),
        },
        index=cell_ids_list,
    )

    gene_meta = pd.DataFrame(
        {
            "NegProbe": [False] * (N_GENES - 2) + [True, True],
        },
        index=gene_names,
    )

    spatial = {
        "x_global": x,
        "y_global": y,
        "x_local": sp_basic.spatial.x_local,
        "y_local": sp_basic.spatial.y_local,
    }

    fov_positions = pd.DataFrame(
        {
            "x_offset": [0, 500],
            "y_offset": [0, 500],
        },
        index=pd.Index(["1", "2"], name="fov"),
    )

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids_list,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        gene_metadata=gene_meta,
        spatial_coords=spatial,
        polygons=polygons,
        fov_positions=fov_positions,
    )


# ===========================================================================
# Fixture 3: sparse-specific object (highly sparse expression matrix)
# ===========================================================================


@pytest.fixture
def sp_sparse():
    """
    spatioloji object where the expression matrix is highly sparse (>80% zeros).

    spatioloji auto-converts dense → sparse when sparsity > 50%.
    Use this to test that sparse storage works correctly.
    """

    np.random.seed(0)

    # Only ~15% of entries are non-zero → triggers auto-sparse conversion
    expression = np.random.choice(
        [0, 0, 0, 0, 0, 1, 2, 3],  # mostly zeros
        size=(N_CELLS, N_GENES),
    ).astype(float)

    cell_ids = [f"cell_{i}" for i in range(N_CELLS)]
    gene_names = [f"gene_{i}" for i in range(N_GENES)]

    cell_meta = pd.DataFrame(
        {
            "fov": ["1"] * 30 + ["2"] * 30,
        },
        index=cell_ids,
    )

    spatial = {
        "x_global": np.random.uniform(0, 1000, N_CELLS),
        "y_global": np.random.uniform(0, 1000, N_CELLS),
        "x_local": np.random.uniform(0, 500, N_CELLS),
        "y_local": np.random.uniform(0, 500, N_CELLS),
    }

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
    )


# ===========================================================================
# Fixture 4: DEG-specific object (two cell types with known signal)
# ===========================================================================


@pytest.fixture
def sp_deg():
    """500-cell × 50-gene spatioloji with two clearly separated cell types.

    TypeA (cells 0–249): high expression in genes 0–24 (mean += 8).
    TypeB (cells 250–499): high expression in genes 25–49 (mean += 8).
    TypeA cells are in x_global [0, 500]; TypeB in [500, 1000] (for spatial tests).
    Replicate labels: rep1 (first 125 of each type), rep2 (last 125 of each type).
    """
    np.random.seed(42)
    n_cells, n_genes, n_fg = 500, 50, 250

    expr = np.random.poisson(1.0, (n_cells, n_genes)).astype(float)
    expr[:n_fg, :25] += np.random.poisson(8.0, (n_fg, 25))
    expr[n_fg:, 25:] += np.random.poisson(8.0, (n_fg, 25))

    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    cell_meta = pd.DataFrame(
        {
            "cell_type": ["TypeA"] * n_fg + ["TypeB"] * n_fg,
            "replicate": ["rep1"] * 125 + ["rep2"] * 125 + ["rep1"] * 125 + ["rep2"] * 125,
        },
        index=cell_ids,
    )

    spatial = {
        "x_global": np.concatenate([np.random.uniform(0, 500, n_fg), np.random.uniform(500, 1000, n_fg)]),
        "y_global": np.random.uniform(0, 1000, n_cells),
        "x_local": np.random.uniform(0, 500, n_cells),
        "y_local": np.random.uniform(0, 500, n_cells),
    }

    return spatioloji(
        expression=expr,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
    )


# ===========================================================================
# Fixture 5: interface-specific object (two spatially separated regions)
# ===========================================================================


@pytest.fixture
def sp_interface():
    """100-cell spatioloji with two spatially separated regions and polygons.

    TypeA (cells 0-49):  x_global in [50, 450], scattered in y [0, 1000].
    TypeB (cells 50-99): x_global in [550, 950], scattered in y [0, 1000].
    Interface cells: ~10 cells near x=500 on each side (x in [460, 540]).
    Each cell gets a 4x4 square polygon around its centroid.
    """
    np.random.seed(99)
    n_cells = 100
    n_genes = 10
    n_per_region = 50

    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # TypeA: x in [50, 450] with a few cells near 490-500
    # TypeB: x in [550, 950] with a few cells near 500-510
    x_a = np.concatenate(
        [
            np.random.uniform(50, 450, n_per_region - 5),
            np.random.uniform(460, 500, 5),  # interface cells
        ]
    )
    x_b = np.concatenate(
        [
            np.random.uniform(550, 950, n_per_region - 5),
            np.random.uniform(500, 540, 5),  # interface cells
        ]
    )
    x_global = np.concatenate([x_a, x_b])
    y_global = np.random.uniform(0, 1000, n_cells)

    cell_meta = pd.DataFrame(
        {"cell_type": ["TypeA"] * n_per_region + ["TypeB"] * n_per_region},
        index=cell_ids,
    )

    spatial = {
        "x_global": x_global,
        "y_global": y_global,
        "x_local": x_global,
        "y_local": y_global,
    }

    # Build 4x4 square polygons
    rows = []
    for cid, cx, cy in zip(cell_ids, x_global, y_global, strict=True):
        for vx, vy in [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx + 2, cy + 2),
            (cx - 2, cy + 2),
            (cx - 2, cy - 2),
        ]:
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})
    polygons = pd.DataFrame(rows)

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
        polygons=polygons,
    )


# ===========================================================================
# Fixture 6: gradient/infiltration-specific object
# ===========================================================================


@pytest.fixture
def sp_gradient():
    """200-cell spatioloji with spatial expression gradient + immune cell types.

    Layout:
    - TypeA (cells 0-99): x_global in [50, 490] (tumor-like region)
    - TypeB (cells 100-199): x_global in [510, 950] (stroma-like region)
    - Interface around x=500.
    - 10 cells per side near boundary (x in [460, 540]).

    Expression:
    - gene_0: positively correlated with distance from interface (gradient gene)
    - gene_1: negatively correlated with distance (inverse gradient)
    - gene_2 to gene_9: random noise (no gradient)

    Cell types (in cell_meta 'immune_type' column):
    - TypeA cells: 80 'Tumor', 20 'CD8_T' (infiltrating immune cells)
    - TypeB cells: 60 'Stroma', 30 'CD8_T' (resident), 10 'Macrophage'

    Each cell has a 4x4 square polygon.
    """
    np.random.seed(42)
    n_cells = 200
    n_genes = 10
    n_per_region = 100

    # Spatial layout — two regions with interface at x=500
    x_a = np.concatenate(
        [
            np.random.uniform(50, 460, n_per_region - 10),
            np.random.uniform(460, 490, 10),
        ]
    )
    x_b = np.concatenate(
        [
            np.random.uniform(510, 540, 10),
            np.random.uniform(540, 950, n_per_region - 10),
        ]
    )
    x_global = np.concatenate([x_a, x_b])
    y_global = np.random.uniform(0, 1000, n_cells)

    # Expression with gradient signal
    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)
    # gene_0: expression increases with distance from x=500
    dist_from_interface = np.abs(x_global - 500)
    expression[:, 0] += (dist_from_interface / 100).astype(float)
    # gene_1: expression decreases with distance from x=500
    expression[:, 1] += np.maximum(0, 5 - dist_from_interface / 100).astype(float)

    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Cell metadata with region and immune type
    region_labels = ["TypeA"] * n_per_region + ["TypeB"] * n_per_region
    # Immune types: CD8_T cells in both regions (20 in A = infiltrating, 30 in B = resident)
    immune_a = ["Tumor"] * 80 + ["CD8_T"] * 20
    immune_b = ["Stroma"] * 60 + ["CD8_T"] * 30 + ["Macrophage"] * 10
    immune_labels = immune_a + immune_b

    cell_meta = pd.DataFrame(
        {
            "cell_type": region_labels,
            "immune_type": immune_labels,
        },
        index=cell_ids,
    )

    spatial = {
        "x_global": x_global,
        "y_global": y_global,
        "x_local": x_global,
        "y_local": y_global,
    }

    # Build 4x4 square polygons
    rows = []
    for cid, cx, cy in zip(cell_ids, x_global, y_global, strict=True):
        for vx, vy in [
            (cx - 2, cy - 2),
            (cx + 2, cy - 2),
            (cx + 2, cy + 2),
            (cx - 2, cy + 2),
            (cx - 2, cy - 2),
        ]:
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})
    polygons = pd.DataFrame(rows)

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
        polygons=polygons,
    )


# ===========================================================================
# Fixture 7: motif-specific object (structured tissue layout)
# ===========================================================================


@pytest.fixture
def sp_motif():
    """500-cell spatioloji with structured tissue layout for motif tests.

    Layout:
    - Center (100 cells, x=400-600, y=400-600): Dense Tumor core (80% Tumor, 20% Fibroblast)
    - Inner ring (100 cells, x=250-750, y=250-750 minus center): Macrophage + Fibroblast stroma
    - Left lobe (100 cells, x=50-250, y=350-650): T_cell + B_cell aggregate (TLS-like)
    - Right lobe (100 cells, x=750-950, y=350-650): Scattered T_cell + Macrophage
    - Periphery (100 cells, x=0-1000, y=0-1000 outer): Sparse Fibroblast + Tumor
    """
    np.random.seed(42)
    n_cells = 500
    n_genes = 10
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # -- Coordinates --
    cx = np.random.uniform(400, 600, 100)
    cy = np.random.uniform(400, 600, 100)

    rx = np.random.uniform(250, 750, 100)
    ry = np.random.uniform(250, 750, 100)
    for i in range(100):
        while 400 <= rx[i] <= 600 and 400 <= ry[i] <= 600:
            rx[i] = np.random.uniform(250, 750)
            ry[i] = np.random.uniform(250, 750)

    lx = np.random.uniform(50, 250, 100)
    ly = np.random.uniform(350, 650, 100)

    rlx = np.random.uniform(750, 950, 100)
    rly = np.random.uniform(350, 650, 100)

    px = np.random.uniform(0, 1000, 100)
    py = np.random.uniform(0, 1000, 100)

    x_global = np.concatenate([cx, rx, lx, rlx, px])
    y_global = np.concatenate([cy, ry, ly, rly, py])

    cell_types = []
    cell_types += ["Tumor"] * 80 + ["Fibroblast"] * 20
    cell_types += ["Macrophage"] * 60 + ["Fibroblast"] * 40
    cell_types += ["T_cell"] * 50 + ["B_cell"] * 40 + ["Macrophage"] * 10
    cell_types += ["T_cell"] * 60 + ["Macrophage"] * 30 + ["Fibroblast"] * 10
    cell_types += ["Fibroblast"] * 50 + ["Tumor"] * 30 + ["T_cell"] * 20

    cell_meta = pd.DataFrame({"cell_type": cell_types}, index=cell_ids)

    expression = np.random.poisson(2.0, (n_cells, n_genes)).astype(float)

    spatial = {
        "x_global": x_global,
        "y_global": y_global,
        "x_local": x_global,
        "y_local": y_global,
    }

    rows = []
    for i, cid in enumerate(cell_ids):
        bx, by = x_global[i], y_global[i]
        for vx, vy in [
            (bx - 2, by - 2),
            (bx + 2, by - 2),
            (bx + 2, by + 2),
            (bx - 2, by + 2),
            (bx - 2, by - 2),
        ]:
            rows.append({"cell": cid, "x_global_px": vx, "y_global_px": vy})
    polygons = pd.DataFrame(rows)

    return spatioloji(
        expression=expression,
        cell_ids=cell_ids,
        gene_names=gene_names,
        cell_metadata=cell_meta,
        spatial_coords=spatial,
        polygons=polygons,
    )
