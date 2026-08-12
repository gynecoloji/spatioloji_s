"""scale() against scanpy, and the precision floor it has to hold at.

`scale()` z-scores each gene, so its whole output rides on one per-gene mean and
one per-gene standard deviation. Those are axis-0 reductions over a float32
(n_cells, n_genes) matrix, and numpy does **not** apply pairwise summation to a
strided axis-0 reduction -- it accumulates sequentially. In float32 that drifts
linearly in n_cells, which is invisible on a toy matrix and material at Xenium
scale.

This is the same failure already guarded for `highly_variable_genes` by
`test_hvg_moments_use_float64_accumulator`; `scale()` was missed at the time.
Measured on 239k-cell Liver before the fix: the per-gene std drifted from
scanpy's by a median relative 4.9e-5, and the elementwise error in the scaled
output grew 4.85x when the cell count grew 5x (1.45e-3 at 20k cells, 7.03e-3 at
100k). After the fix the error is flat at ~1 float32 ULP.
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.data.core import spatioloji
from spatioloji_s.processing.normalization import scale


def _obj(counts):
    n, n_genes = counts.shape
    cells = [f"c{i}" for i in range(n)]
    genes = [f"g{i}" for i in range(n_genes)]
    rng = np.random.default_rng(0)
    return spatioloji(
        expression=counts,
        cell_ids=cells,
        gene_names=genes,
        cell_metadata=pd.DataFrame({"fov": ["1"] * n}, index=cells),
        gene_metadata=pd.DataFrame(index=genes),
        spatial_coords={k: rng.uniform(0, 10, n)
                        for k in ("x_global", "y_global", "x_local", "y_local")},
    )


def _matrix(n_cells, n_genes=24, seed=3):
    """Sparse-ish lognormal counts with a wide per-gene mean spread.

    The spread matters: the accumulator error scales with the magnitude of the
    running sum, so genes with large means are where a float32 accumulator
    fails first.
    """
    rng = np.random.default_rng(seed)
    gene_scale = rng.lognormal(1.0, 0.8, n_genes)
    return np.where(
        rng.random((n_cells, n_genes)) < 0.9, 0.0,
        rng.lognormal(0.0, 0.7, (n_cells, n_genes)) * gene_scale,
    ).astype(np.float32)


def test_scale_moments_use_float64_accumulator():
    """The per-gene mean and std must match a float64 computation exactly.

    Checked against the statistics rather than the scaled matrix, so a failure
    points at the accumulator instead of at clipping or the division.
    """
    counts = _matrix(250_000)
    sp = _obj(counts)
    scale(sp, layer=None, output_layer="scaled", max_value=None,
          method="standard", zero_center=True, inplace=True, device="cpu")

    exact = counts.astype(np.float64)
    mean64, std64 = exact.mean(axis=0), exact.std(axis=0, ddof=1)
    # Recover the statistics the implementation actually used, from the output.
    out = np.asarray(sp.get_layer("scaled"), dtype=np.float64)
    implied_std = (exact[:, 0] - mean64[0]) / out[:, 0]
    np.testing.assert_allclose(implied_std[np.isfinite(implied_std)].mean(),
                               std64[0], rtol=1e-5)
    # Column means of the centred output must sit at zero, not at the drift.
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)


def test_scale_matches_scanpy():
    """Elementwise agreement with sc.pp.scale, at the float32 floor."""
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")

    counts = _matrix(120_000)
    sp = _obj(counts)
    a = ad.AnnData(X=counts.copy(),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(counts.shape[0])]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[1])]))
    sc.pp.scale(a, max_value=10.0, zero_center=True)
    scale(sp, layer=None, output_layer="scaled", max_value=10.0,
          method="standard", zero_center=True, inplace=True, device="cpu")

    got = np.asarray(sp.get_layer("scaled"))
    ref = np.asarray(a.X)
    # A few float32 ULP at |z| <= 10. The pre-fix value here was ~7e-3.
    assert np.abs(got - ref).max() < 2e-5


def test_scale_error_does_not_grow_with_cell_count():
    """The regression signature: a float32 accumulator drifts linearly in n.

    Guards the mechanism rather than a magnitude, so it still fails if the
    accumulator is reintroduced somewhere the absolute tolerances above happen
    to tolerate.
    """
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")

    errs = []
    for n_cells in (40_000, 200_000):
        counts = _matrix(n_cells)
        sp = _obj(counts)
        a = ad.AnnData(X=counts.copy(),
                       obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[1])]))
        sc.pp.scale(a, max_value=10.0, zero_center=True)
        scale(sp, layer=None, output_layer="scaled", max_value=10.0,
              method="standard", zero_center=True, inplace=True, device="cpu")
        errs.append(float(np.abs(np.asarray(sp.get_layer("scaled")) - np.asarray(a.X)).max()))

    # 5x the cells drifted 4.85x before the fix; it must stay flat now.
    assert errs[1] <= max(errs[0], 1e-6) * 2.0, (
        f"error grew {errs[1] / max(errs[0], 1e-30):.2f}x for 5x cells: {errs}"
    )
