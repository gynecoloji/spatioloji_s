"""Regression test pinning the vectorized ``_analytical_test`` to the naive form.

``_analytical_test`` was rewritten from an O(rows x edges) loop -- which
re-filtered the whole edge table once per summary row -- to an O(edges) +
O(rows) groupby form. The rewrite is only legitimate if it is numerically
identical, so the naive implementation is reproduced here verbatim and the two
are compared on fixtures that exercise every branch.

Context: the benchmark suite previously monkey-patched this function at runtime
to get acceptable performance, which meant published timings described code
that was never distributed. This test is what allows the patch to be deleted.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from spatioloji_s.ccc.scoring import _analytical_test


def _analytical_test_naive(result, edge_df, N, type_counts, norm):
    """The original O(rows x edges) implementation, kept as the oracle."""
    pvals = np.ones(len(result))
    zscores = np.zeros(len(result))
    modes = (
        result["interaction_mode"].values
        if "interaction_mode" in result.columns
        else np.array(["paracrine"] * len(result))
    )

    for i, row in result.iterrows():
        lr = row["lr_name"]
        st = row["sender_type"]
        rt = row["receiver_type"]
        obs_sum = row["sum_score"]
        mode = modes[i] if i < len(modes) else "paracrine"

        n_s = type_counts.get(st, 0)
        n_r = type_counts.get(rt, 0)

        if n_s == 0 or n_r == 0 or N < 2:
            pvals[i] = 1.0
            continue

        lr_edges = edge_df[edge_df["lr_name"] == lr]
        if lr_edges.empty:
            pvals[i] = 1.0
            continue

        all_scores = lr_edges["score"].values
        total_score = all_scores.sum()
        mean_sq = (all_scores**2).mean()
        n_total_edges = len(all_scores)

        if mode == "autocrine":
            p_pair = n_s / N if N > 0 else 0.0
        else:
            p_pair = (n_s * n_r) / (N * (N - 1)) if N > 1 else 0.0
        e_sum = p_pair * total_score
        var_sum = p_pair * (1.0 - p_pair) * mean_sq * n_total_edges

        if var_sum < 1e-30:
            pvals[i] = 1.0
            zscores[i] = 0.0
            continue

        z = (obs_sum - e_sum) / np.sqrt(var_sum)
        zscores[i] = z
        pvals[i] = float(norm.sf(z))

    result = result.copy()
    result["z_score"] = zscores
    result["pvalue"] = pvals
    return result


def _make_case(seed, n_lr=6, n_rows=40, n_edges=800, with_modes=True,
               unknown_types=False, missing_lr=False):
    rng = np.random.RandomState(seed)
    lrs = [f"LR{i}" for i in range(n_lr)]
    types = ["A", "B", "C", "D"]

    edge_df = pd.DataFrame({
        "lr_name": rng.choice(lrs, n_edges),
        "score": rng.gamma(2.0, 1.5, n_edges),
    })
    # An LR present in the summary but absent from the edge table exercises
    # the "no null available" branch.
    row_lrs = list(lrs) + (["LR_absent"] if missing_lr else [])

    result = pd.DataFrame({
        "lr_name": rng.choice(row_lrs, n_rows),
        "sender_type": rng.choice(types, n_rows),
        "receiver_type": rng.choice(types, n_rows),
        "sum_score": rng.gamma(3.0, 4.0, n_rows),
    })
    if with_modes:
        result["interaction_mode"] = rng.choice(
            ["paracrine", "autocrine"], n_rows
        )
        # autocrine rows have sender == receiver by construction
        auto = result["interaction_mode"] == "autocrine"
        result.loc[auto, "receiver_type"] = result.loc[auto, "sender_type"]

    counts = {t: int(rng.randint(5, 300)) for t in types}
    if unknown_types:
        # a summary row referencing a type with zero cells
        result.loc[0, "sender_type"] = "GHOST"
    N = sum(counts.values())
    return result, edge_df, N, counts


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_naive_general(seed):
    result, edge_df, N, counts = _make_case(seed)
    got = _analytical_test(result, edge_df, N, counts, norm)
    want = _analytical_test_naive(result, edge_df, N, counts, norm)

    np.testing.assert_allclose(
        got["z_score"].to_numpy(), want["z_score"].to_numpy(),
        rtol=1e-12, atol=1e-12, err_msg="z_score diverged from naive form",
    )
    np.testing.assert_allclose(
        got["pvalue"].to_numpy(), want["pvalue"].to_numpy(),
        rtol=1e-12, atol=1e-12, err_msg="pvalue diverged from naive form",
    )


def test_matches_naive_without_mode_column():
    """No interaction_mode column -> every row treated as paracrine."""
    result, edge_df, N, counts = _make_case(11, with_modes=False)
    got = _analytical_test(result, edge_df, N, counts, norm)
    want = _analytical_test_naive(result, edge_df, N, counts, norm)
    np.testing.assert_allclose(got["z_score"], want["z_score"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["pvalue"], want["pvalue"], rtol=1e-12, atol=1e-12)


def test_matches_naive_with_unknown_celltype():
    """A sender type with no cells must yield p=1, z=0."""
    result, edge_df, N, counts = _make_case(12, unknown_types=True)
    got = _analytical_test(result, edge_df, N, counts, norm)
    want = _analytical_test_naive(result, edge_df, N, counts, norm)
    np.testing.assert_allclose(got["z_score"], want["z_score"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["pvalue"], want["pvalue"], rtol=1e-12, atol=1e-12)
    assert got.loc[0, "pvalue"] == 1.0
    assert got.loc[0, "z_score"] == 0.0


def test_matches_naive_with_lr_absent_from_edges():
    """An LR pair with no edges must yield p=1, z=0."""
    result, edge_df, N, counts = _make_case(13, missing_lr=True)
    got = _analytical_test(result, edge_df, N, counts, norm)
    want = _analytical_test_naive(result, edge_df, N, counts, norm)
    np.testing.assert_allclose(got["z_score"], want["z_score"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["pvalue"], want["pvalue"], rtol=1e-12, atol=1e-12)
    absent = got["lr_name"] == "LR_absent"
    assert absent.any(), "fixture did not actually produce an absent LR"
    assert (got.loc[absent, "pvalue"] == 1.0).all()


def test_empty_result_frame():
    _, edge_df, N, counts = _make_case(14)
    empty = pd.DataFrame(
        columns=["lr_name", "sender_type", "receiver_type", "sum_score"]
    )
    got = _analytical_test(empty, edge_df, N, counts, norm)
    assert len(got) == 0
    assert "z_score" in got.columns and "pvalue" in got.columns


def test_empty_edge_frame():
    """No edges at all -> every row p=1, z=0 rather than a crash."""
    result, _, N, counts = _make_case(15)
    empty_edges = pd.DataFrame(columns=["lr_name", "score"])
    got = _analytical_test(result, empty_edges, N, counts, norm)
    assert (got["pvalue"] == 1.0).all()
    assert (got["z_score"] == 0.0).all()


def test_n_below_two():
    """N < 2 makes the paracrine probability undefined -> p=1."""
    result, edge_df, _, counts = _make_case(16)
    got = _analytical_test(result, edge_df, 1, counts, norm)
    assert (got["pvalue"] == 1.0).all()
    assert (got["z_score"] == 0.0).all()


def test_zero_variance_edges():
    """All-zero edge scores give var_sum == 0 -> p=1, not a divide-by-zero."""
    result = pd.DataFrame({
        "lr_name": ["LR0", "LR0"],
        "sender_type": ["A", "B"],
        "receiver_type": ["B", "A"],
        "sum_score": [0.0, 1.0],
    })
    edge_df = pd.DataFrame({"lr_name": ["LR0"] * 10, "score": [0.0] * 10})
    counts = {"A": 50, "B": 50}
    got = _analytical_test(result, edge_df, 100, counts, norm)
    assert (got["pvalue"] == 1.0).all()
    assert (got["z_score"] == 0.0).all()


def test_is_not_monkeypatched():
    """The benchmark suite must measure the shipped implementation.

    A runtime patch would make published timings describe code that is not
    distributed, which is exactly the situation this rewrite removes.
    """
    assert _analytical_test.__module__.startswith("spatioloji_s"), (
        f"_analytical_test has been replaced by {_analytical_test.__module__}; "
        "benchmarks must run against the shipped implementation"
    )
