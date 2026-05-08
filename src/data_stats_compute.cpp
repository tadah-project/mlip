#include <tadah/mlip/data_stats_compute.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tadah/core/logger.h>
#include <tadah/core/stats/histogram.h>
#include <tadah/core/stats/kde.h>
#include <tadah/core/stats/percentiles.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>

namespace tadah {
namespace mlip {

namespace {

using core::stats::DataStats;
using core::stats::PerPairStats;
using core::stats::Histogram;
using core::stats::histogram_fd;
using core::stats::kde_evaluate;
using core::stats::kde_peaks;
using core::stats::silverman_bandwidth;
using core::stats::summary_percentiles;

/** \brief Order an element-pair lex-ascending so {Na,Cl} and {Cl,Na}
 *  collapse to one bucket. */
inline std::pair<std::string, std::string>
canonical_pair(const std::string& a, const std::string& b) {
  return (a <= b) ? std::make_pair(a, b)
                  : std::make_pair(b, a);
}

/** \brief Linear grid [lo, hi] with n points (n >= 2). */
std::vector<double> linspace(double lo, double hi, std::size_t n) {
  std::vector<double> g(n);
  if (n == 1) { g[0] = lo; return g; }
  const double step = (hi - lo) / static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) g[i] = lo + step * static_cast<double>(i);
  return g;
}

}  // namespace

DataStats compute_data_stats(StructureDB& stdb,
                             tadah::core::Context& ctx,
                             bool /*with_descriptor_stats*/,
                             std::size_t sample_size) {
  DataStats out;

  if (!ctx.contains("RCUTMAX")) {
    throw std::runtime_error(
        "compute_data_stats: RCUTMAX is not set in the context. "
        "Ensure CoreUserConfigParser has postprocessed the config "
        "(set at least one of RCUT2B / RCUTMB).");
  }
  const double rcut_audit = ctx.get<double>("RCUTMAX");

  if (stdb.size() == 0) {
    return out;   // empty stdb -> empty stats
  }

  // Decide which structure indices participate in the scan.
  // sample_size == 0 means "use the entire DB".
  std::vector<std::size_t> idx;
  if (sample_size == 0 || sample_size >= stdb.size()) {
    idx.resize(stdb.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
  } else {
    // Deterministic random sample without replacement. Seed on
    // stdb.size() so repeated runs on the same DB pick the same
    // subset (reproducible audit findings).
    std::mt19937_64 rng(static_cast<std::uint64_t>(stdb.size()));
    std::vector<std::size_t> all(stdb.size());
    std::iota(all.begin(), all.end(), std::size_t{0});
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(sample_size);
    std::sort(all.begin(), all.end());           // cache-friendlier traversal
    idx = std::move(all);
  }

  // 1) Make sure neighbour lists are populated for the sampled subset.
  //    We touch only the subset so a 256-of-6288 audit is bounded by
  //    the sample, not the full DB.
  std::vector<std::size_t> need_nn_idx;
  need_nn_idx.reserve(idx.size());
  for (std::size_t s : idx) {
    if (!stdb(s).nn) need_nn_idx.push_back(s);
  }
  if (!need_nn_idx.empty()) {
    NNFinder nnf;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t k = 0; k < need_nn_idx.size(); ++k) {
      nnf.calc(stdb(need_nn_idx[k]), ctx);
    }
  }

  // 2) Collect per-pair r_ij distances + per-element neighbour counts +
  //    per-config energy/force scales.
  std::map<std::pair<std::string, std::string>, std::vector<double>> r_pairs;
  std::map<std::string, std::vector<double>> nn_count_per_elem;
  std::vector<double> e_per_atom;
  std::vector<double> f_norms;

  std::map<std::string, bool> seen_elem;

  // n_structures / n_atoms reflect the *scanned* set so downstream
  // percentiles are consistent with what was actually summarised.
  out.n_structures = idx.size();
  for (std::size_t s : idx) {
    const Structure& st = stdb(s);
    out.n_atoms += st.natoms();

    if (st.natoms() > 0) {
      e_per_atom.push_back(st.energy / static_cast<double>(st.natoms()));
    }

    for (size_t i = 0; i < st.natoms(); ++i) {
      const Atom& ai = st(i);
      const std::string Si = ai.symbol;
      seen_elem[Si] = true;

      // neighbour-count summary uses the audit cutoff (== RCUTMAX).
      double n_count = 0.0;
      const size_t nnsize = st.nn_size(i);
      for (size_t jj = 0; jj < nnsize; ++jj) {
        const tadah::core::Vec3d nnp = st.nn_pos(i, jj);
        const double dx = ai.position[0] - nnp[0];
        const double dy = ai.position[1] - nnp[1];
        const double dz = ai.position[2] - nnp[2];
        const double rij = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (rij <= 0.0 || rij > rcut_audit) continue;
        n_count += 1.0;
        const std::size_t neigh_idx = st.nn->idx[i][jj];
        const std::string Sj = st(neigh_idx).symbol;
        r_pairs[canonical_pair(Si, Sj)].push_back(rij);
      }
      nn_count_per_elem[Si].push_back(n_count);

      // Force-norm tracking: Atom inherits force[3] components.
      const double fx = ai.force[0], fy = ai.force[1], fz = ai.force[2];
      const double fn = std::sqrt(fx*fx + fy*fy + fz*fz);
      f_norms.push_back(fn);
    }
  }

  for (const auto& kv : seen_elem) out.elements.push_back(kv.first);
  std::sort(out.elements.begin(), out.elements.end());

  // 3) Per-pair statistics.
  for (auto& kv : r_pairs) {
    PerPairStats pp;
    pp.elements = kv.first;
    pp.n_pairs  = kv.second.size();
    if (pp.n_pairs == 0) continue;

    auto& v = kv.second;
    pp.r_min = *std::min_element(v.begin(), v.end());
    pp.r_max = *std::max_element(v.begin(), v.end());
    pp.r_percentiles = summary_percentiles(v);

    // FD histogram on this pair.
    pp.r_histogram = histogram_fd(v);

    // KDE peaks: evaluate density on a 200-point grid spanning
    // [r_min, r_max]; prominence threshold is 1% of max density (small
    // enough to keep secondary shells, large enough to suppress noise
    // floor — derived from the data, not a tunable).
    const double bw = silverman_bandwidth(v);
    if (bw > 0.0 && pp.r_max > pp.r_min) {
      auto grid = linspace(pp.r_min, pp.r_max, 200);
      auto density = kde_evaluate(v, grid, bw);
      double dmax = 0.0;
      for (double x : density) if (x > dmax) dmax = x;
      const double prom = 0.01 * dmax;
      pp.r_peaks = kde_peaks(grid, density, prom);
    }

    out.per_pair.push_back(std::move(pp));
  }

  // Sort per_pair by canonical element pair so output is deterministic.
  std::sort(out.per_pair.begin(), out.per_pair.end(),
            [](const PerPairStats& a, const PerPairStats& b) {
              return a.elements < b.elements;
            });

  // 4) Neighbour-count percentiles per element.
  for (auto& kv : nn_count_per_elem) {
    if (!kv.second.empty()) {
      out.nn_count_percentiles_per_element[kv.first] =
          summary_percentiles(kv.second);
    }
  }

  // 5) Energy / force scales.
  if (!e_per_atom.empty()) {
    out.e_per_atom_percentiles = summary_percentiles(e_per_atom);
  }
  if (!f_norms.empty()) {
    out.f_norm_percentiles = summary_percentiles(f_norms);
  }

  // 6) Material-class label.
  out.class_label = classify_dataset(out);

  return out;
}

std::string classify_dataset(const DataStats& d) {
  if (d.elements.empty() || d.per_pair.empty()) return "unclassified";

  // Median neighbour count over all elements (use the largest if multi).
  double max_med_nn = 0.0;
  for (const auto& kv : d.nn_count_percentiles_per_element) {
    const double med = kv.second[core::stats::IDX_P50];
    if (med > max_med_nn) max_med_nn = med;
  }

  if (max_med_nn < 10.0) return "cluster_or_dimer";

  // Multi-element heuristic.
  if (d.elements.size() >= 2) {
    // If one of the per-pair distributions is bimodal with deep
    // minimum, call it a molecular crystal; else covalent/ionic.
    bool bimodal = false;
    for (const auto& pp : d.per_pair) {
      if (pp.r_peaks.size() >= 2) {
        const double pk1 = pp.r_peaks[0].position;
        const double pk2 = pp.r_peaks[1].position;
        if (pk2 / std::max(pk1, 1e-9) > 1.6) { bimodal = true; break; }
      }
    }
    return bimodal ? "molecular_crystal" : "covalent_or_ionic";
  }

  // Single-element: distinguish dense crystalline metal vs. amorphous.
  // If the principal pair has 2+ resolved peaks, treat as crystalline.
  if (d.per_pair[0].r_peaks.size() >= 2) return "metal_like";

  return "amorphous_or_liquid";
}

}  // namespace mlip
}  // namespace tadah
