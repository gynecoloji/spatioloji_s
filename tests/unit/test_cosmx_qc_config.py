"""Tests for CosmxQCConfig and the percent-unit contract in CosMx QC.

`spatioloji_qc` computes pct_counts_NegProbe as `neg_counts * 100 / total_counts` -- a
PERCENTAGE -- but `QCConfig.pct_counts_neg_max` defaulted to 0.1, so the effective cut was
0.1%, not the 10% the protocol documents. On a 39,939-cell CosMx section that is the
difference between keeping 25,368 cells and 38,045. These tests pin the units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioloji_s import CosmxQCConfig, QCConfig, spatioloji, spatioloji_qc


# --------------------------------------------------------------------------- config
def test_defaults_are_percentages_not_fractions():
    c = CosmxQCConfig()
    assert c.pct_counts_neg_max == 10.0, "10% of counts from negative probes"
    assert c.pct_counts_mt_max == 25.0, "25% mitochondrial"


def test_docstring_states_the_unit():
    assert "0-100" in CosmxQCConfig.__doc__ or "percent" in CosmxQCConfig.__doc__.lower()


def test_fraction_shaped_value_warns(recwarn):
    """0.1 is legal (0.1%) but is almost always someone meaning 10%."""
    CosmxQCConfig(pct_counts_neg_max=0.1)
    msgs = " ".join(str(w.message) for w in recwarn)
    assert "pct_counts_neg_max" in msgs and "10.0" in msgs


def test_unambiguous_values_do_not_warn(recwarn):
    CosmxQCConfig(pct_counts_neg_max=10.0, pct_counts_mt_max=25.0)
    assert not [w for w in recwarn if "pct_counts" in str(w.message)]


def test_zero_does_not_warn(recwarn):
    """0 means 'allow none' and is not a unit confusion."""
    CosmxQCConfig(pct_counts_neg_max=0.0)
    assert not [w for w in recwarn if "pct_counts_neg_max" in str(w.message)]


def test_qcconfig_is_a_backwards_compatible_alias():
    assert issubclass(QCConfig, CosmxQCConfig) or QCConfig is CosmxQCConfig


def test_area_outlier_method_is_validated():
    CosmxQCConfig(area_outlier_method="esd")
    CosmxQCConfig(area_outlier_method="percentile")
    CosmxQCConfig(area_outlier_method="none")
    with pytest.raises(ValueError, match="area_outlier_method"):
        CosmxQCConfig(area_outlier_method="grubs")      # typo


# --------------------------------------------------------------------------- fixture
def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(8)] + ["NegPrb1", "NegPrb2"]
    X = rng.integers(1, 40, size=(n, len(genes))).astype(float)
    X[:, -2:] = 0                                        # clean cells: no neg counts
    # A graded band of negative-probe contamination. The 1-5% cells are the ones that
    # discriminate 0.1% from 10%; without them both thresholds keep the same cells and
    # the regression test passes vacuously.
    gene_tot = X[:, :-2].sum(1)
    for i, frac in enumerate((0.50, 0.30, 0.012, 0.02, 0.03, 0.04, 0.05)):
        X[i, -2] = max(round(gene_tot[i] * frac / (1 - frac)), 1)
    ids = [f"1_{i}" for i in range(n)]
    meta = pd.DataFrame({"cell": ids, "fov": "1",
                         "Area": rng.normal(5000, 500, n).clip(100)}, index=ids)
    meta.loc[ids[0], "Area"] = 1e6                       # one extreme area
    return spatioloji(
        expression=X, cell_ids=ids, gene_names=genes, cell_metadata=meta,
        spatial_coords={"x_local": rng.random(n), "y_local": rng.random(n),
                        "x_global": rng.random(n), "y_global": rng.random(n)},
    )


# --------------------------------------------------------------------------- behaviour
def test_ten_percent_keeps_more_cells_than_tenth_of_a_percent():
    """The regression this whole change exists for."""
    kept = {}
    for thr in (0.1, 10.0):
        q = spatioloji_qc(_toy(), CosmxQCConfig(pct_counts_neg_max=thr, save_plots=False))
        q.qc_cell_metrics(plot=False)
        q.qc_cell_area(plot=False)
        kept[thr] = int(q.filter_cells().sum())
    assert kept[10.0] > kept[0.1], f"10% must be more permissive than 0.1%: {kept}"


def test_generalized_esd_removes_more_than_one_outlier():
    """Classic Grubbs returns a single index; ESD iterates."""
    sp = _toy()
    ids = list(sp.cell_index)
    sp.cell_meta.loc[ids[:6], "Area"] = [1e6, 9e5, 8e5, 7e5, 6e5, 5e5]

    g = spatioloji_qc(sp, CosmxQCConfig(area_outlier_method="grubbs", save_plots=False))
    g.qc_cell_area(plot=False)
    n_grubbs = int(g.sp.cell_meta["QC_Area_outlier"].sum())

    e = spatioloji_qc(_toy(), CosmxQCConfig(area_outlier_method="esd", save_plots=False))
    ids2 = list(e.sp.cell_index)
    e.sp.cell_meta.loc[ids2[:6], "Area"] = [1e6, 9e5, 8e5, 7e5, 6e5, 5e5]
    e.qc_cell_area(plot=False)
    n_esd = int(e.sp.cell_meta["QC_Area_outlier"].sum())

    assert n_grubbs == 1, f"classic Grubbs flags exactly one, got {n_grubbs}"
    assert n_esd > n_grubbs, f"ESD should find several, got {n_esd}"


def test_percentile_area_method():
    q = spatioloji_qc(_toy(), CosmxQCConfig(area_outlier_method="percentile",
                                            area_percentile=(5.0, 95.0), save_plots=False))
    q.qc_cell_area(plot=False)
    n = int(q.sp.cell_meta["QC_Area_outlier"].sum())
    assert 20 <= n <= 40, f"~10% of 300 cells should fall outside 5-95, got {n}"


def test_filter_cells_warns_when_area_step_was_skipped():
    """The area term is silently dropped if qc_cell_area() was never called."""
    q = spatioloji_qc(_toy(), CosmxQCConfig(save_plots=False))
    q.qc_cell_metrics(plot=False)
    with pytest.warns(UserWarning, match="qc_cell_area"):
        q.filter_cells()


def test_filter_cells_does_not_warn_when_area_step_ran():
    q = spatioloji_qc(_toy(), CosmxQCConfig(save_plots=False))
    q.qc_cell_metrics(plot=False)
    q.qc_cell_area(plot=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        q.filter_cells()


def test_filter_cells_names_the_missing_step():
    """A bare KeyError on a column name does not tell you which step to run."""
    q = spatioloji_qc(_toy(), CosmxQCConfig(save_plots=False))
    with pytest.raises(RuntimeError, match="qc_cell_metrics"):
        q.filter_cells()


# ------------------------------------------------- gene percentile threshold (units bug)
def _panel_toy(gene_totals, neg_totals, n=100):
    """A spatioloji whose per-gene column sums are exactly `gene_totals` / `neg_totals`.

    filter_genes only reads column sums, so pinning them exactly makes the threshold
    arithmetic unambiguous.
    """
    genes = [f"G{i}" for i in range(len(gene_totals))] + \
            [f"NegPrb{i}" for i in range(len(neg_totals))]
    X = np.zeros((n, len(genes)))
    for j, tot in enumerate(list(gene_totals) + list(neg_totals)):
        tot = int(tot)
        X[:tot % n, j] = tot // n + 1
        X[tot % n:, j] = tot // n
        X[:, j] *= tot / X[:, j].sum() if X[:, j].sum() else 1
    ids = [f"1_{i}" for i in range(n)]
    meta = pd.DataFrame({"cell": ids, "fov": "1", "Area": np.full(n, 5000.0)}, index=ids)
    return spatioloji(
        expression=X, cell_ids=ids, gene_names=genes, cell_metadata=meta,
        spatial_coords={"x_local": np.arange(n, dtype=float), "y_local": np.zeros(n),
                        "x_global": np.arange(n, dtype=float), "y_global": np.zeros(n)},
    )


def test_gene_percentile_uses_the_percentile_not_the_total():
    """The threshold is a percentile ACROSS negative probes, not their sum.

    `np.percentile(neg_counts.sum(), 50)` collapses the per-probe totals to one number and
    returns it, so the cut scales with the NUMBER of control probes. With 19 NegPrb probes
    on a real CosMx panel that inflated the threshold ~19x and kept 72 of 960 genes, which
    is only visible later as cell typing that cannot find most lineages.
    """
    # per-probe neg totals 10/20/30 -> median 20 (intended) vs sum 60 (shipped bug)
    sp = _panel_toy(gene_totals=[15, 25, 50, 100], neg_totals=[10, 20, 30])
    q = spatioloji_qc(sp, CosmxQCConfig(gene_filter_method="percentile",
                                        gene_percentile_threshold=50, save_plots=False))
    mask = q.filter_genes(plot=False)
    kept = [g for g in sp.gene_index[np.asarray(mask)] if not g.startswith("NegPrb")]
    assert kept == ["G1", "G2", "G3"], (
        f"expected the genes above the 20-count median (25/50/100), got {kept}. "
        f"Only G3 surviving means the threshold was the 60-count sum."
    )


def test_gene_percentile_threshold_is_independent_of_probe_count():
    """Adding control probes must not make the gene filter stricter."""
    kept = {}
    for n_probes in (3, 12):
        sp = _panel_toy(gene_totals=[15, 25, 50, 100], neg_totals=[20] * n_probes)
        q = spatioloji_qc(sp, CosmxQCConfig(gene_filter_method="percentile",
                                            gene_percentile_threshold=50, save_plots=False))
        mask = q.filter_genes(plot=False)
        kept[n_probes] = sum(1 for g in sp.gene_index[np.asarray(mask)] if not g.startswith("NegPrb"))
    assert kept[3] == kept[12], (
        f"the gene cut moved when only the probe count changed: {kept}"
    )


def test_gene_percentile_survives_a_panel_with_no_control_probes():
    """An empty control set must mean "no baseline", not a crash.

    `from_cosmx(drop_negative_probes=True)` is the default and `gene_filter_method
    ="percentile"` is the default, so a panel whose controls were already dropped -- or any
    non-CosMx object -- reaches this path with `neg_counts` empty. `np.percentile` of an
    empty array raises IndexError. The Xenium twin guards exactly this (qc.py:2382-2384).
    """
    sp = _panel_toy(gene_totals=[15, 25, 50, 100], neg_totals=[])
    q = spatioloji_qc(sp, CosmxQCConfig(gene_filter_method="percentile",
                                        gene_percentile_threshold=50, save_plots=False))
    mask = q.filter_genes(plot=False)      # must not raise
    kept = [g for g in sp.gene_index[np.asarray(mask)]]
    assert kept == ["G0", "G1", "G2", "G3"], (
        f"with no controls the baseline is 0, so every expressed gene survives; got {kept}"
    )
