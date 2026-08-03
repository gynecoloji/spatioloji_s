"""Concordance of deg_wilcoxon with scanpy's rank_genes_groups(method='wilcoxon').

Markers are *up*-regulated genes, so the ranking statistic has to be directional.
Ranking by a two-sided p-value gets this wrong twice over: it mixes in genes that
are strongly DOWN in the group, and at spatial-transcriptomics scale the p-values
underflow to exactly 0.0 for hundreds of genes, leaving the top-N to be decided by
an arbitrary tie-break. Both are fixed by ranking on the signed z-score, which is
what scanpy reports as `scores`.
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.data.core import spatioloji
from spatioloji_s.processing import deg_ttest, deg_wilcoxon


def _make_sp(counts: np.ndarray, labels: np.ndarray) -> spatioloji:
    n_cells, n_genes = counts.shape
    cells = [f"c{i}" for i in range(n_cells)]
    genes = [f"g{i}" for i in range(n_genes)]
    rng = np.random.default_rng(0)
    sp = spatioloji(
        expression=counts,
        cell_ids=cells,
        gene_names=genes,
        cell_metadata=pd.DataFrame(
            {"fov": ["1"] * n_cells, "grp": [str(x) for x in labels]}, index=cells
        ),
        gene_metadata=pd.DataFrame(index=genes),
        spatial_coords={
            k: rng.uniform(0, 100, n_cells)
            for k in ("x_global", "y_global", "x_local", "y_local")
        },
    )
    from spatioloji_s.processing import log_transform, normalize_total

    normalize_total(sp, target_sum=1e4, inplace=True, device="cpu")
    log_transform(sp, layer="normalized_counts", inplace=True, device="cpu")
    return sp


def _counts(n_cells: int, n_genes: int, n_types: int, seed: int = 31):
    """Poisson counts where each type up-regulates its own block of genes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_types, n_cells)
    base = rng.lognormal(0.0, 1.2, n_genes)
    block = n_genes // n_types
    prog = np.ones((n_types, n_genes))
    for t in range(n_types):
        prog[t, t * block:(t + 1) * block] = 3.0
    depth = rng.lognormal(0.0, 0.3, n_cells)[:, None]
    return rng.poisson(base[None, :] * prog[labels, :] * depth).astype(np.float32), labels


@pytest.fixture(scope="module")
def small():
    counts, labels = _counts(1500, 200, 4)
    return _make_sp(counts, labels), counts, labels


@pytest.fixture(scope="module")
def saturating():
    """Big enough that two-sided p-values underflow to exactly 0 for many genes."""
    counts, labels = _counts(20000, 200, 4)
    return _make_sp(counts, labels), counts, labels


# ---------------------------------------------------------------------------
# statistic and ordering
# ---------------------------------------------------------------------------


def test_wilcoxon_reports_signed_score(small):
    """A directional score must be reported, signed toward the foreground group."""
    sp, _, _ = small
    df = deg_wilcoxon(sp, groupby="grp", group_fg=["0"], group_bg="rest",
                      layer="log_normalized")["wilcoxon"]

    assert "score" in df.columns
    up = df["mean_fg"] > df["mean_bg"]
    # every clearly-shifted gene must carry the matching sign
    strong = df["padj"] < 0.01
    assert (df.loc[strong & up, "score"] > 0).all()
    assert (df.loc[strong & ~up, "score"] < 0).all()


def test_wilcoxon_sorted_by_score_descending(small):
    """Top rows must be the most up-regulated genes, not the smallest p-values."""
    sp, _, _ = small
    df = deg_wilcoxon(sp, groupby="grp", group_fg=["0"], group_bg="rest",
                      layer="log_normalized")["wilcoxon"]

    scores = df["score"].to_numpy()
    assert np.all(np.diff(scores) <= 1e-12), "result is not sorted by score descending"
    assert df["mean_fg"].iloc[0] > df["mean_bg"].iloc[0]


def test_other_backends_still_sorted_by_padj(small):
    """Backends without a directional score keep their padj ordering."""
    sp, _, _ = small
    df = deg_ttest(sp, groupby="grp", group_fg=["0"], group_bg="rest",
                   layer="log_normalized")["ttest"]

    assert "score" not in df.columns
    padj = df["padj"].to_numpy()
    assert np.all(np.diff(padj[~np.isnan(padj)]) >= -1e-12)


# ---------------------------------------------------------------------------
# agreement with scanpy
# ---------------------------------------------------------------------------


def _scanpy_wilcoxon(counts, labels, tie_correct=False):
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    ad = pytest.importorskip("anndata", reason="anndata not installed")

    adata = ad.AnnData(counts.copy())
    adata.obs_names = [f"c{i}" for i in range(counts.shape[0])]
    adata.var_names = [f"g{i}" for i in range(counts.shape[1])]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs["grp"] = pd.Categorical([str(x) for x in labels])
    sc.tl.rank_genes_groups(adata, "grp", method="wilcoxon", tie_correct=tie_correct)
    return adata


def _scanpy_scores(adata, group):
    names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])
    return pd.Series(scores[group].to_numpy(), index=names[group].astype(str))


def test_wilcoxon_scores_match_scanpy(small):
    """The z-score formula must be scanpy's, gene for gene.

    Fed scanpy's own normalized matrix, so the only thing under test is the
    statistic. Comparing whole pipelines instead would fold in the ~1e-7 float32
    difference in normalize+log, which shifts near-tied ranks.
    """
    _, counts, labels = small
    adata = _scanpy_wilcoxon(counts, labels)
    ref = _scanpy_scores(adata, "0")

    from spatioloji_s.processing.DEG import _wilcoxon_backend

    X = np.asarray(adata.X)
    fg = labels == 0
    got = pd.Series(_wilcoxon_backend(X[fg], X[~fg])["score"], index=adata.var_names.astype(str))

    np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=1e-5, atol=1e-5)


def test_wilcoxon_scores_match_scanpy_end_to_end(small):
    """Through the whole pipeline the scores still agree, to float32 tolerance."""
    sp, counts, labels = small
    ref = _scanpy_scores(_scanpy_wilcoxon(counts, labels), "0")

    df = deg_wilcoxon(sp, groupby="grp", group_fg=["0"], group_bg="rest",
                      layer="log_normalized")["wilcoxon"]
    got = pd.Series(df["score"].to_numpy(), index=df["gene"].astype(str))

    np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=1e-3, atol=1e-3)


def test_wilcoxon_top_markers_match_scanpy(small):
    """Top-50 markers per group must match scanpy's."""
    sp, counts, labels = small
    adata = _scanpy_wilcoxon(counts, labels)
    names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])

    for group in names.columns:
        df = deg_wilcoxon(sp, groupby="grp", group_fg=[group], group_bg="rest",
                          layer="log_normalized")["wilcoxon"]
        got = set(df["gene"].astype(str).head(50))
        ref = set(names[group].astype(str).head(50))
        overlap = len(got & ref) / 50
        assert overlap > 0.98, f"group {group}: top-50 overlap {overlap:.3f}"


def test_wilcoxon_ranking_survives_pvalue_saturation(saturating):
    """With many p-values underflowing to 0, the ranking must still match scanpy."""
    sp, counts, labels = saturating
    adata = _scanpy_wilcoxon(counts, labels)
    names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])

    df = deg_wilcoxon(sp, groupby="grp", group_fg=["0"], group_bg="rest",
                      layer="log_normalized")["wilcoxon"]

    # the condition this test exists for
    assert int((df["padj"] <= 0).sum()) > 20, "fixture did not saturate p-values"

    got = set(df["gene"].astype(str).head(50))
    ref = set(names["0"].astype(str).head(50))
    overlap = len(got & ref) / 50
    assert overlap > 0.98, f"top-50 overlap under saturation = {overlap:.3f}"


def test_wilcoxon_tie_correction_matches_scanpy(small):
    """tie_correct=True must reproduce scanpy's tie-corrected scores."""
    _, counts, labels = small
    adata = _scanpy_wilcoxon(counts, labels, tie_correct=True)
    ref = _scanpy_scores(adata, "0")

    from spatioloji_s.processing.DEG import _wilcoxon_backend

    X = np.asarray(adata.X)
    fg = labels == 0
    got = pd.Series(
        _wilcoxon_backend(X[fg], X[~fg], tie_correct=True)["score"],
        index=adata.var_names.astype(str),
    )

    np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=1e-5, atol=1e-5)


def test_tie_correction_changes_scores_when_ties_exist(small):
    """Guard that the tie_correct flag is actually wired through."""
    sp, _, _ = small
    kw = dict(groupby="grp", group_fg=["0"], group_bg="rest", layer="log_normalized")
    plain = deg_wilcoxon(sp, **kw)["wilcoxon"].set_index("gene")["score"]
    corrected = deg_wilcoxon(sp, tie_correct=True, **kw)["wilcoxon"].set_index("gene")["score"]

    # sparse counts guarantee shared zeros, so the correction must bite
    assert not np.allclose(plain.to_numpy(), corrected.loc[plain.index].to_numpy())
    # and it only rescales, so the sign of every score is preserved
    assert np.array_equal(np.sign(plain.to_numpy()), np.sign(corrected.loc[plain.index].to_numpy()))
