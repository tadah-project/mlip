#ifndef TADAH_MLIP_DATA_STATS_COMPUTE_H
#define TADAH_MLIP_DATA_STATS_COMPUTE_H

// compute_data_stats — produce a populated DataStats from a StructureDB
// plus the (already postprocessed) Context that drives RCUTMAX.
//
// Part of the HPO bound-validation pipeline. Engine wraps this in
// `analysis_dataset_stats_engine`. The bound suggester and the pre-flight
// audit consume the resulting DataStats.
//
// find_close_contacts lists the pairs of atoms closer than a threshold
// (CLOSE_CONTACT of `tadah analysis dataset_stats`).

#include <array>
#include <cstddef>
#include <string>
#include <vector>
#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/structure_db.h>

namespace tadah {
namespace mlip {

/** \brief Compute dataset statistics over a StructureDB.
 *
 *  Side effect: runs NNFinder::calc on each structure that participates
 *  in the scan if its NN list is missing or was built for a cutoff below
 *  RCUTMAX (idempotent if NNs already populated). Hence stdb is non-const.
 *
 *  Two atoms at one position (also through a periodic image) are refused,
 *  as by training, prediction and HPO (NNFinder; the message names the
 *  structure). With `list_coincident` true (tadah analysis dataset_stats,
 *  the audit of a dataset) the lists are built with
 *  NNFinder::list_coincident(true) instead: such atoms are a pair at
 *  distance 0, counted like any other pair strictly inside RCUTMAX (in
 *  n_pairs, the neighbour counts and the distance statistics), so per_pair
 *  r_min is 0 for such a structure. A list built so stays in `st.nn` and
 *  holds that pair: descriptors must not be computed from it (they divide
 *  by the distance).
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
 *  The energy-per-atom and force-magnitude statistics take the structures
 *  that have them: a structure with eweight 0 has no energy of its own and
 *  one with fweight 0 no forces (the values stored for them are
 *  placeholders; tadah data convert stores zero forces with fweight 0 for
 *  a file that gives none), so they are left out of those (not of the pair
 *  statistics).
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
                   std::size_t sample_size = 0,
                   bool list_coincident = false);

/** \brief A pair of atoms closer than a threshold (find_close_contacts). */
struct CloseContact {
  /** Index of the structure in the StructureDB (0-based). */
  std::size_t structure = 0;
  /** The two atoms (0-based), i <= j; i == j for an atom and its own
   *  periodic image (a cell vector shorter than the threshold). */
  std::size_t i = 0;
  std::size_t j = 0;
  /** The image of atom j that is close to atom i: atom j displaced by
   *  n0 a0 + n1 a1 + n2 a2 (a_k = cell.row(k)), the integer triple of
   *  NNTable::shift; (0, 0, 0) for a pair at the stored positions. */
  std::array<long long, 3> shift{{0, 0, 0}};
  /** Distance (Angstrom), computed as compute_data_stats computes it; 0
   *  for two atoms at one position. */
  double r = 0.0;
};

/** \brief Every pair of atoms closer than @p threshold (Angstrom) in the
 *  structures of @p stdb (`tadah analysis dataset_stats`, CLOSE_CONTACT).
 *
 *  A pair is listed when its distance r < threshold (strictly), computed
 *  as compute_data_stats computes it. Each unordered pair is listed once:
 *  the neighbour list holds a pair as i -> j (shift s) and as j -> i
 *  (shift -s), and it is reported with i < j or, for an atom and its own
 *  image (i == j), with the first non-zero component of s positive. A pair
 *  through a periodic image is included, and two atoms at one position
 *  are a pair at distance 0 (the lists are built with
 *  NNFinder::list_coincident(true)). Order: by structure, then by i, then
 *  by (j, s) as NNFinder sorts them, i.e. dataset order.
 *
 *  A structure's own list (`st.nn`, as NNFinder built it for the stored
 *  positions) is scanned when its cutoff covers the threshold, e.g. the
 *  lists compute_data_stats built at RCUTMAX. Otherwise (no list, or a
 *  threshold beyond its cutoff) a list out to the threshold is built for
 *  the scan only and not kept, so the answer is complete whatever the
 *  threshold and the stored lists are left as they are. The structures
 *  are scanned in parallel (OpenMP).
 *
 *  Throws std::invalid_argument for a threshold that is not a finite
 *  number greater than 0. A neighbour-list error (e.g. a cell too small
 *  for the threshold, see NNFinder::build) is carried out of the parallel
 *  loop with the same type, the lowest-numbered structure's first,
 *  prefixed "find_close_contacts: structure <i> of <N> ('<label>'): "
 *  (1-based i).
 */
std::vector<CloseContact>
find_close_contacts(const StructureDB& stdb, double threshold);

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
