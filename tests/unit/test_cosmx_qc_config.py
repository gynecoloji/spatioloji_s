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
