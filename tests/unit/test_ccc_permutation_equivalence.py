"""Regression test pinning the vectorized ``_permutation_test`` to the naive form.

``_permutation_test`` was rewritten from an O(n_permutations x rows x edges) form
-- which re-scanned the whole edge table once per result row *inside* every
permutation -- to an O(n_permutations x edges) integer-keyed accumulation. On a
LymphNode ROI the naive form cost ~39,555 s at n_permutations=100, against 7.3 s
for squidpy performing the same 100 permutations; six ROIs would have needed
~660 h against a 36 h wall.

The rewrite is only legitimate if it is numerically identical, so the naive
implementation is reproduced here verbatim and the two are compared on fixtures
that exercise every branch: paracrine and autocrine modes, self-edges, a key
present in ``result`` but absent from the edge table, and the subsampling path.

Exact equality is achievable -- not merely statistical agreement -- because the
rewrite preserves the RNG call sequence. ``RandomState.shuffle`` is Fisher-Yates
and draws once per position regardless of dtype, so shuffling an integer-coded
label array consumes the same stream as shuffling the original object array.
That is asserted directly in ``test_int_codes_share_the_rng_stream`` below: if it
ever fails, the equivalence claim is void and the other assertions here are
meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from spatioloji_s.ccc.scoring import _permutation_test
from spatioloji_s.data.core import spatioloji


def _naive_permutation_test(result, edge_df, sp, group_col, n_subsample,
                            n_permutations, seed):
    """The pre-2026-08-22 implementation, verbatim. Do not optimise."""
    rng = np.random.RandomState(seed)
    cell_types = sp.cell_meta[group_col].copy()
    all_cells = np.asarray(sp.cell_index)
    N = len(all_cells)

    n_sub = min(n_subsample, N)
    if n_sub < N:
        type_fracs = cell_types.value_counts(normalize=True)
        sub_indices = []
        for ctype, frac in type_fracs.items():
            ct_idx = np.where(cell_types.values == ctype)[0]
            n_take = max(1, int(round(frac * n_sub)))
            chosen = rng.choice(ct_idx, size=min(n_take, len(ct_idx)), replace=False)
            sub_indices.append(chosen)
        sub_indices = np.concatenate(sub_indices)
        sub_cells = set(all_cells[sub_indices])
    else:
        sub_cells = set(all_cells)

    mask = edge_df["sender"].isin(sub_cells) & edge_df["receiver"].isin(sub_cells)
    sub_edge = edge_df[mask].copy()

    if sub_edge.empty:
        result = result.copy()
        result["pvalue"] = 1.0
        return result

    if "interaction_mode" not in sub_edge.columns:
        sub_edge["interaction_mode"] = "paracrine"

    cells_in_edges = np.unique(
        np.concatenate([sub_edge["sender"].values, sub_edge["receiver"].values]))
    sub_types = cell_types.reindex(cells_in_edges)
    has_mode = "interaction_mode" in result.columns

    obs_key_to_sum = {}
    for _, row in result.iterrows():
        mode = row["interaction_mode"] if has_mode else "paracrine"
        key = (row["lr_name"], row["sender_type"], row["receiver_type"], mode)
        sub_lr = sub_edge[(sub_edge["lr_name"] == row["lr_name"])
                          & (sub_edge["interaction_mode"] == mode)]
        mask_st = sub_lr["sender"].map(lambda c, st=row["sender_type"]: sub_types.get(c) == st)
        mask_rt = sub_lr["receiver"].map(lambda c, rt=row["receiver_type"]: sub_types.get(c) == rt)
        obs_key_to_sum[key] = sub_lr.loc[mask_st & mask_rt, "score"].sum()

    perm_counts = {k: 0 for k in obs_key_to_sum}
    for _ in range(n_permutations):
        shuffled = sub_types.values.copy()
        rng.shuffle(shuffled)
        shuffled_map = dict(zip(cells_in_edges, shuffled, strict=True))
        perm_sender_types = sub_edge["sender"].map(shuffled_map)
        perm_receiver_types = sub_edge["receiver"].map(shuffled_map)
        for key in obs_key_to_sum:
            lr, st, rt, mode = key
            lr_mask = (sub_edge["lr_name"] == lr) & (sub_edge["interaction_mode"] == mode)
            perm_sum = sub_edge.loc[lr_mask & (perm_sender_types == st)
                                    & (perm_receiver_types == rt), "score"].sum()
            if perm_sum >= obs_key_to_sum[key]:
                perm_counts[key] += 1

    result = result.copy()
    pvals = []
    for _, row in result.iterrows():
        mode = row["interaction_mode"] if has_mode else "paracrine"
        key = (row["lr_name"], row["sender_type"], row["receiver_type"], mode)
        pvals.append((perm_counts.get(key, n_permutations) + 1) / (n_permutations + 1))
    result["pvalue"] = pvals
    return result


def _fixture(n_cells=60, seed=0):
    rng = np.random.default_rng(seed)
    types = np.array(["B cell", "T cell", "Macrophage", "NK cell"])
    cells = [f"c{i}" for i in range(n_cells)]
    ct = rng.choice(types, n_cells)
    sp = spatioloji(
        expression=rng.random((n_cells, 4), dtype=np.float32),
        cell_ids=cells, gene_names=[f"g{i}" for i in range(4)],
        cell_metadata=pd.DataFrame({"fov": ["1"] * n_cells, "cell_type": ct}, index=cells),
        gene_metadata=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        spatial_coords={k: rng.uniform(0, 50, n_cells)
                        for k in ("x_global", "y_global", "x_local", "y_local")},
    )
    # Edges: paracrine pairs plus autocrine self-edges, two LR pairs.
    rows = []
    for _ in range(400):
        i, j = rng.integers(0, n_cells, 2)
        rows.append(dict(sender=cells[i], receiver=cells[j],
                         lr_name=rng.choice(["LR_A", "LR_B"]),
                         interaction_mode="paracrine", score=float(rng.random())))
    for i in range(0, n_cells, 3):          # self-edges must keep sender==receiver type
        rows.append(dict(sender=cells[i], receiver=cells[i], lr_name="LR_A",
                         interaction_mode="autocrine", score=float(rng.random())))
    edge = pd.DataFrame(rows)
    agg = (edge.groupby(["lr_name", "interaction_mode"], as_index=False).size()
             .drop(columns="size"))
    combos = []
    for _, a in agg.iterrows():
        for st in types:
            for rt in types:
                if a.interaction_mode == "autocrine" and st != rt:
                    continue
                combos.append(dict(lr_name=a.lr_name, sender_type=st, receiver_type=rt,
                                   interaction_mode=a.interaction_mode))
    # One key deliberately absent from the edge table, to exercise the fallback.
    combos.append(dict(lr_name="LR_ABSENT", sender_type="B cell",
                       receiver_type="T cell", interaction_mode="paracrine"))
    return sp, edge, pd.DataFrame(combos)


def test_int_codes_share_the_rng_stream():
    """The premise of exact equivalence: shuffle() draws depend on length, not dtype."""
    types = np.array(["B cell", "T cell", "Macrophage", "B cell"] * 30, dtype=object)
    uniq, codes = np.unique(types, return_inverse=True)
    a, b = types.copy(), codes.copy()
    r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
    for _ in range(25):
        r1.shuffle(a); r2.shuffle(b)
        np.testing.assert_array_equal(a, uniq[b])


@pytest.mark.parametrize("n_permutations", [20, 100])
@pytest.mark.parametrize("n_subsample", [10_000, 40])   # 40 forces the subsample path
def test_matches_naive_exactly(n_permutations, n_subsample):
    sp, edge, result = _fixture()
    got = _permutation_test(result, edge, sp, "cell_type", n_subsample,
                            n_permutations, seed=42)
    want = _naive_permutation_test(result, edge, sp, "cell_type", n_subsample,
                                   n_permutations, seed=42)
    np.testing.assert_array_equal(got["pvalue"].to_numpy(), want["pvalue"].to_numpy())


def test_empty_subsample_returns_p_one():
    sp, edge, result = _fixture()
    empty = edge.iloc[0:0]
    out = _permutation_test(result, empty, sp, "cell_type", 10_000, 20, seed=1)
    assert (out["pvalue"] == 1.0).all()


def test_pvalues_are_in_range_and_use_the_plus_one_estimator():
    sp, edge, result = _fixture()
    out = _permutation_test(result, edge, sp, "cell_type", 10_000, 50, seed=3)
    p = out["pvalue"].to_numpy()
    assert p.min() >= 1 / 51 - 1e-12, "floor must be (0+1)/(N+1), never 0"
    assert p.max() <= 1.0
