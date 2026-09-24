#ifndef TADAH_MLIP_DATA_STATS_COMPUTE_H
#define TADAH_MLIP_DATA_STATS_COMPUTE_H

// compute_data_stats — produce a populated DataStats from a StructureDB
// plus the (already postprocessed) Context that drives RCUTMAX.
//
// Phase 3 of the HPO bound-validation pipeline. Engine wraps this in
// `analysis_dataset_stats_engine`. Phases 4 / 5 consume the resulting
// DataStats.

#include <string>
#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/structure_db.h>

namespace tadah {
namespace mlip {

/** \brief Compute dataset statistics over a StructureDB.
 *
 *  Side effect: runs NNFinder::calc on each structure that participates
 *  in the scan if its NN list is missing (idempotent if NNs already
 *  populated). Hence stdb is non-const.
 *
 *  RCUTMAX is read from `ctx`. The caller is responsible for setting
 *  it (the standard CoreUserConfigParser path does this automatically
 *  from the configured RCUT2B/RCUTMB).
 *
 *  `sample_size` controls how many structures contribute to the stats:
 *    - `0` (default): use every structure in the DB.
 *    - `N > 0`: randomly sample `min(N, stdb.size())` structures with
 *      a deterministic seed (`stdb.size()` itself), so repeated runs
 *      on the same DB hit the same subset and stats are reproducible.
 *
 *  Sampling lets the caller bound the cost of the NN sweep on large
 *  training sets where the audit only needs a representative cross
 *  section (callers like `run_preflight_from_db` set this from the
 *  AUDIT_SAMPLE context key).
 *
 *  When `with_descriptor_stats` is true, callers should populate
 *  `out.rho_percentiles_per_descriptor` themselves (this function only
 *  fills the dataset-side stats; per-descriptor ρ requires a trained
 *  potential or seeded model and is left to a higher-level engine).
 */
tadah::core::stats::DataStats
compute_data_stats(StructureDB& stdb,
                   tadah::core::Context& ctx,
                   bool with_descriptor_stats = false,
                   std::size_t sample_size = 0);

/** \brief Heuristic classifier — runs after `compute_data_stats` and
 *  returns the material-class label (also written into
 *  `DataStats::class_label` by `compute_data_stats`).
 *
 *  Decision rules, in order: no elements or pair data gives
 *  "unclassified"; a largest per-element median neighbour count below 10
 *  gives "cluster_or_dimer"; with two or more elements, a pair whose second
 *  RDF peak lies beyond 1.6x the first gives "molecular_crystal", otherwise
 *  "covalent_or_ionic"; a single element gives "metal_like" when its pair
 *  RDF resolves two or more peaks, otherwise "amorphous_or_liquid".
 */
std::string classify_dataset(const tadah::core::stats::DataStats& d);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DATA_STATS_COMPUTE_H
