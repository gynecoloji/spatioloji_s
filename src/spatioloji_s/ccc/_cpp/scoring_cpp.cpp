// scoring_cpp.cpp — pybind11 C++ acceleration for spatioloji_s CCC scoring
//
// Two functions:
//   score_edges_batch  — vectorised edge scoring for all LR pairs
//   permutation_test   — label-permutation significance test
//
// Compile flag SPATIOLOJI_USE_OPENMP enables OpenMP parallelism.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#ifdef SPATIOLOJI_USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ── score_edges_batch ───────────────────────────────────────────────────────
//
// For each LR pair p and each edge e:
//   L_i = min(expr_mat[sender[e], ligand_cols[p]...])
//   R_j = min(expr_mat[receiver[e], receptor_cols[p]...])
//   out[p * n_edges + e] = sqrt(max(L_i, 0) * max(R_j, 0)) * weight[e]

py::array_t<double> score_edges_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> expr_mat,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> sender_idx,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> receiver_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    std::vector<std::vector<int>> ligand_cols,
    std::vector<std::vector<int>> receptor_cols)
{
    // ── Validate shapes ────────────────────────────────────────────────────
    auto expr = expr_mat.unchecked<2>();
    auto s_idx = sender_idx.unchecked<1>();
    auto r_idx = receiver_idx.unchecked<1>();
    auto w = weights.unchecked<1>();

    const std::ptrdiff_t n_cells = expr.shape(0);
    const std::ptrdiff_t n_genes = expr.shape(1);
    const std::ptrdiff_t n_edges = s_idx.shape(0);
    const std::ptrdiff_t n_pairs = static_cast<std::ptrdiff_t>(ligand_cols.size());

    if (static_cast<std::ptrdiff_t>(receptor_cols.size()) != n_pairs) {
        throw std::invalid_argument("ligand_cols and receptor_cols must have the same length");
    }
    if (r_idx.shape(0) != n_edges || w.shape(0) != n_edges) {
        throw std::invalid_argument("sender_idx, receiver_idx, and weights must have the same length");
    }

    // Raw pointer for fast access
    const double* expr_ptr = expr.data(0, 0);
    const std::ptrdiff_t* s_ptr = s_idx.data(0);
    const std::ptrdiff_t* r_ptr = r_idx.data(0);
    const double* w_ptr = w.data(0);

    // ── Allocate output ────────────────────────────────────────────────────
    const std::ptrdiff_t total = n_pairs * n_edges;
    auto result = py::array_t<double>(total);
    double* out = result.mutable_data();

    // ── Score loop ─────────────────────────────────────────────────────────
#ifdef SPATIOLOJI_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (std::ptrdiff_t p = 0; p < n_pairs; ++p) {
        const auto& lcols = ligand_cols[p];
        const auto& rcols = receptor_cols[p];
        const std::ptrdiff_t n_lig = static_cast<std::ptrdiff_t>(lcols.size());
        const std::ptrdiff_t n_rec = static_cast<std::ptrdiff_t>(rcols.size());
        double* out_p = out + p * n_edges;

        for (std::ptrdiff_t e = 0; e < n_edges; ++e) {
            const std::ptrdiff_t si = s_ptr[e];
            const std::ptrdiff_t ri = r_ptr[e];
            const double wi = w_ptr[e];

            // L_i = min across ligand subunits
            double l_val;
            if (n_lig == 0) {
                l_val = 0.0;
            } else {
                const double* row_s = expr_ptr + si * n_genes;
                l_val = row_s[lcols[0]];
                for (std::ptrdiff_t k = 1; k < n_lig; ++k) {
                    double v = row_s[lcols[k]];
                    if (v < l_val) l_val = v;
                }
            }

            // R_j = min across receptor subunits
            double r_val;
            if (n_rec == 0) {
                r_val = 0.0;
            } else {
                const double* row_r = expr_ptr + ri * n_genes;
                r_val = row_r[rcols[0]];
                for (std::ptrdiff_t k = 1; k < n_rec; ++k) {
                    double v = row_r[rcols[k]];
                    if (v < r_val) r_val = v;
                }
            }

            // score = sqrt(max(L, 0) * max(R, 0)) * weight
            double lc = l_val > 0.0 ? l_val : 0.0;
            double rc = r_val > 0.0 ? r_val : 0.0;
            out_p[e] = std::sqrt(lc * rc) * wi;
        }
    }

    return result;
}

// ── permutation_test ────────────────────────────────────────────────────────
//
// For each permutation:
//   1. Shuffle type labels
//   2. Walk all edges, accumulate score into flat 3D array [lr][st][rt]
//   3. For each key, compare permuted sum to observed sum
// Returns int64 perm_counts (n_keys,)

py::array_t<int64_t> permutation_test(
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> sender_idx,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> receiver_idx,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> type_array,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> lr_int,
    py::array_t<double, py::array::c_style | py::array::forcecast> score_arr,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> key_lr,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> key_st,
    py::array_t<std::ptrdiff_t, py::array::c_style | py::array::forcecast> key_rt,
    py::array_t<bool, py::array::c_style | py::array::forcecast> key_valid,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_sums,
    int n_types,
    int n_lrs,
    int n_permutations,
    unsigned int seed)
{
    // ── Unchecked accessors ────────────────────────────────────────────────
    auto s_idx = sender_idx.unchecked<1>();
    auto r_idx = receiver_idx.unchecked<1>();
    auto types = type_array.unchecked<1>();
    auto lr = lr_int.unchecked<1>();
    auto scores = score_arr.unchecked<1>();
    auto klr = key_lr.unchecked<1>();
    auto kst = key_st.unchecked<1>();
    auto krt = key_rt.unchecked<1>();
    auto kvalid = key_valid.unchecked<1>();
    auto obs = obs_sums.unchecked<1>();

    const std::ptrdiff_t n_edges = s_idx.shape(0);
    const std::ptrdiff_t n_cells = types.shape(0);
    const std::ptrdiff_t n_keys = klr.shape(0);

    // Raw pointers for hot loop
    const std::ptrdiff_t* s_ptr = s_idx.data(0);
    const std::ptrdiff_t* r_ptr = r_idx.data(0);
    const std::ptrdiff_t* type_ptr = types.data(0);
    const std::ptrdiff_t* lr_ptr = lr.data(0);
    const double* score_ptr = scores.data(0);
    const std::ptrdiff_t* klr_ptr = klr.data(0);
    const std::ptrdiff_t* kst_ptr = kst.data(0);
    const std::ptrdiff_t* krt_ptr = krt.data(0);
    const bool* kvalid_ptr = kvalid.data(0);
    const double* obs_ptr = obs.data(0);

    // 3D accumulator stride: accum[lr_i * n_types * n_types + st * n_types + rt]
    const std::ptrdiff_t type_stride = static_cast<std::ptrdiff_t>(n_types);
    const std::ptrdiff_t accum_size = static_cast<std::ptrdiff_t>(n_lrs) * type_stride * type_stride;

    // ── Allocate output ────────────────────────────────────────────────────
    auto result = py::array_t<int64_t>(n_keys);
    int64_t* perm_counts = result.mutable_data();
    std::memset(perm_counts, 0, n_keys * sizeof(int64_t));

    // ── Permutation loop ───────────────────────────────────────────────────

#ifdef SPATIOLOJI_USE_OPENMP
    // Each thread: own RNG, shuffled array, accumulator, local counts
    const int n_threads = omp_get_max_threads();

    // Pre-allocate per-thread buffers
    std::vector<std::mt19937> rngs(n_threads);
    std::vector<std::vector<std::ptrdiff_t>> shuffled_bufs(n_threads);
    std::vector<std::vector<double>> accum_bufs(n_threads);
    std::vector<std::vector<int64_t>> local_counts(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        rngs[t] = std::mt19937(seed + static_cast<unsigned int>(t));
        shuffled_bufs[t].resize(n_cells);
        accum_bufs[t].resize(accum_size, 0.0);
        local_counts[t].resize(n_keys, 0);
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& rng = rngs[tid];
        auto& shuffled = shuffled_bufs[tid];
        auto& accum = accum_bufs[tid];
        auto& counts = local_counts[tid];

        // Initialize shuffled from type_array
        std::copy(type_ptr, type_ptr + n_cells, shuffled.data());

        #pragma omp for schedule(dynamic)
        for (int perm = 0; perm < n_permutations; ++perm) {
            // Shuffle type labels
            for (std::ptrdiff_t i = n_cells - 1; i > 0; --i) {
                std::uniform_int_distribution<std::ptrdiff_t> dist(0, i);
                std::ptrdiff_t j = dist(rng);
                std::swap(shuffled[i], shuffled[j]);
            }

            // Zero accumulator
            std::memset(accum.data(), 0, accum_size * sizeof(double));

            // Accumulate scores
            for (std::ptrdiff_t e = 0; e < n_edges; ++e) {
                const std::ptrdiff_t lr_i = lr_ptr[e];
                const std::ptrdiff_t st = shuffled[s_ptr[e]];
                const std::ptrdiff_t rt = shuffled[r_ptr[e]];
                accum[lr_i * type_stride * type_stride + st * type_stride + rt] += score_ptr[e];
            }

            // Compare to observed
            for (std::ptrdiff_t k = 0; k < n_keys; ++k) {
                if (!kvalid_ptr[k]) continue;
                double perm_sum = accum[klr_ptr[k] * type_stride * type_stride +
                                        kst_ptr[k] * type_stride + krt_ptr[k]];
                if (perm_sum >= obs_ptr[k]) {
                    counts[k] += 1;
                }
            }
        }
    }

    // Reduce thread-local counts
    for (int t = 0; t < n_threads; ++t) {
        for (std::ptrdiff_t k = 0; k < n_keys; ++k) {
            perm_counts[k] += local_counts[t][k];
        }
    }

#else
    // Single-threaded path
    std::mt19937 rng(seed);
    std::vector<std::ptrdiff_t> shuffled(type_ptr, type_ptr + n_cells);
    std::vector<double> accum(accum_size, 0.0);

    for (int perm = 0; perm < n_permutations; ++perm) {
        // Fisher-Yates shuffle
        for (std::ptrdiff_t i = n_cells - 1; i > 0; --i) {
            std::uniform_int_distribution<std::ptrdiff_t> dist(0, i);
            std::ptrdiff_t j = dist(rng);
            std::swap(shuffled[i], shuffled[j]);
        }

        // Zero accumulator
        std::memset(accum.data(), 0, accum_size * sizeof(double));

        // Accumulate scores into 3D flat array
        for (std::ptrdiff_t e = 0; e < n_edges; ++e) {
            const std::ptrdiff_t lr_i = lr_ptr[e];
            const std::ptrdiff_t st = shuffled[s_ptr[e]];
            const std::ptrdiff_t rt = shuffled[r_ptr[e]];
            accum[lr_i * type_stride * type_stride + st * type_stride + rt] += score_ptr[e];
        }

        // Compare each key to observed sum
        for (std::ptrdiff_t k = 0; k < n_keys; ++k) {
            if (!kvalid_ptr[k]) continue;
            double perm_sum = accum[klr_ptr[k] * type_stride * type_stride +
                                    kst_ptr[k] * type_stride + krt_ptr[k]];
            if (perm_sum >= obs_ptr[k]) {
                perm_counts[k] += 1;
            }
        }
    }
#endif

    return result;
}

// ── Module definition ──────────────────────────────────────────────────────

PYBIND11_MODULE(_scoring_cpp, m) {
    m.doc() = "C++ acceleration for spatioloji_s CCC scoring and permutation testing";

    m.def("score_edges_batch", &score_edges_batch,
          py::arg("expr_mat"),
          py::arg("sender_idx"),
          py::arg("receiver_idx"),
          py::arg("weights"),
          py::arg("ligand_cols"),
          py::arg("receptor_cols"),
          R"doc(
Score all LR pairs across all edges in a single vectorised pass.

For each pair p and edge e:
    L_i = min(expr_mat[sender[e], ligand_cols[p]...])
    R_j = min(expr_mat[receiver[e], receptor_cols[p]...])
    score = sqrt(max(L_i, 0) * max(R_j, 0)) * weight[e]

Args:
    expr_mat: float64 (n_cells, n_genes), C-contiguous expression matrix.
    sender_idx: intp (n_edges,), row indices into expr_mat.
    receiver_idx: intp (n_edges,), row indices into expr_mat.
    weights: float64 (n_edges,), edge weights.
    ligand_cols: list[list[int]], per-pair ligand column indices.
    receptor_cols: list[list[int]], per-pair receptor column indices.

Returns:
    Flat float64 array of shape (n_pairs * n_edges,).
)doc");

    m.def("permutation_test", &permutation_test,
          py::arg("sender_idx"),
          py::arg("receiver_idx"),
          py::arg("type_array"),
          py::arg("lr_int"),
          py::arg("score_arr"),
          py::arg("key_lr"),
          py::arg("key_st"),
          py::arg("key_rt"),
          py::arg("key_valid"),
          py::arg("obs_sums"),
          py::arg("n_types"),
          py::arg("n_lrs"),
          py::arg("n_permutations"),
          py::arg("seed"),
          R"doc(
Label-permutation significance test for CCC scores.

For each permutation, shuffles cell-type labels, accumulates edge scores
into a 3D (lr x sender_type x receiver_type) array, and counts how often
the permuted sum meets or exceeds the observed sum for each key.

Args:
    sender_idx: intp (n_edges,), cell indices into type_array.
    receiver_idx: intp (n_edges,), cell indices into type_array.
    type_array: intp (n_cells,), integer-encoded cell types.
    lr_int: intp (n_edges,), integer LR pair ID per edge.
    score_arr: float64 (n_edges,), edge scores.
    key_lr: intp (n_keys,), LR index for each summary key.
    key_st: intp (n_keys,), sender type index for each key.
    key_rt: intp (n_keys,), receiver type index for each key.
    key_valid: bool (n_keys,), whether each key should be tested.
    obs_sums: float64 (n_keys,), observed score sums.
    n_types: int, number of distinct cell types.
    n_lrs: int, number of distinct LR pairs.
    n_permutations: int, number of permutations.
    seed: unsigned int, RNG seed.

Returns:
    int64 array (n_keys,) of permutation counts.
)doc");
}
