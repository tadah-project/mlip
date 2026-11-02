#ifndef TADAH_MLIP_CHECK_BOUNDS_H
#define TADAH_MLIP_CHECK_BOUNDS_H

// `tadah analysis check_bounds`. CI-friendly dry-run.
//
// Reads the OPTIM lines of an HPOTARGET file as tadah hpo reads them
// (MLIP's OPTIM reader, hpotarget_optim.h: index lists, AUTO boxes, DM_Embed
// component paths, FIXWEIGHT coefficients), builds a bound box per selected
// value, samples corners (full 2^k for k ≤ 12, otherwise
// `min(2^k, ⌈k·log₂(k)⌉)` Latin-hypercube samples), and runs the
// pre-flight audit at each corner. A corner value below the low
// bound the seed's density form or a radial bump channel allows
// (models::optim_low_bound_rule(key, seed), the boxes tadah hpo refuses)
// is a FAIL finding too; a DM_REAM seed must declare RHOFORM. Returns 1 if ANY corner produces a FAIL finding, 0
// otherwise. Intended for CI gating.

#include <cstdint>
#include <string>
#include <vector>

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

class StructureDB;

/** \brief One bound entry from the HPOTARGET file: the box of one value
 *  an OPTIM line selects. `OPTIM CGRID2B (1-4) 0 5` gives 4 Bound rows,
 *  arg_pos 0..3; an AUTO line the box tadah hpo derives for each value. */
struct Bound {
  std::string key{};         // upper-cased; a DM_Embed component path too
  /** The 0-based position of the value among the key's tokens; -1 (a
   *  bound made by hand) is position 0. */
  int         arg_pos{-1};
  double      lo{0.0};
  double      hi{0.0};
  std::size_t index{0};      // the OPTIM index as written (FIXWEIGHT: the coefficient)
  std::size_t line{0};       // the last physical line of its OPTIM line
};

/** \brief The boxes of the OPTIM lines of the HPOTARGET file @p path, one
 *  per value each line selects, read as tadah hpo reads them against the
 *  training configuration @p seed (read_optim_lines): @p seed is prepared
 *  first as tadah hpo prepares its configuration (prepare_optim_seed: the
 *  radial grids expanded, the values of a potential its descriptors do not
 *  read left out, a DM_REAM configuration refused without RHOFORM), and a
 *  key of a line inserts its blueprint default into it. Every logical line
 *  whose key is OPTIM is read, inside a block too; other lines are not.
 *  Throws std::runtime_error "check_bounds: cannot open HPOTARGET file
 *  '<path>'", and, for a line tadah hpo refuses, "tadah analysis
 *  check_bounds: line <N> of <path>: <tadah hpo's reason>\nLine: <the
 *  line>". */
std::vector<Bound> parse_hpo_target_bounds(const std::string& path,
                                           tadah::core::Context& seed);

/** \brief Sample corner combinations of `bounds`. For `bounds.size() <= 12`
 *  returns the full `2^k` corner set. For larger problems returns a
 *  Latin-hypercube subsample of `min(2^k, ⌈k·log₂(k)⌉)` rows. Each
 *  inner vector has the same length as `bounds` and contains either
 *  `bounds[i].lo` or `bounds[i].hi`.
 */
std::vector<std::vector<double>>
sample_corners(const std::vector<Bound>& bounds, std::uint64_t seed = 0);

/** \brief The context a corner is audited in, the one tadah hpo evaluates
 *  (update_candidate_context): a copy of @p seed with entry i of each
 *  bound's key (arg_pos, or the first entry for arg_pos -1) set to the
 *  corner value and every other entry kept. Several bounds on one key each
 *  set their own entry. A key the seed does not set takes the corner value
 *  alone. A DM_Embed component path (a key holding '.') sets the value of
 *  its component (models::write_embed_coordinates, as tadah hpo writes a
 *  candidate). The cutoff of a knot channel (RCUT2B of a D2_Knot5
 *  line, RCUTMB of a knot DM_REAM line) is then its largest knot
 *  (models::potedit::derived_knot_rcut), as tadah hpo re-derives it for
 *  each candidate. Throws when a bound names an entry past the seed's
 *  values (std::runtime_error) or @p corner and @p bounds differ in size
 *  (std::invalid_argument). */
tadah::core::Context corner_context(const tadah::core::Context& seed,
                                    const std::vector<Bound>& bounds,
                                    const std::vector<double>& corner);

/** \brief Run the pre-flight audit at every corner of the bound box (in the
 *  corner_context of each corner, with the statistics of the dataset taken
 *  at the largest cutoff of the seed and of every corner, the RADIUS of a
 *  DM_Embed component's density included; each descriptor
 *  judged by those of the element pairs of its own line, each pair on its
 *  own, as run_preflight_soft() judges it), after the
 *  stated cutoff of a knot channel of `seed` is replaced with its largest
 *  knot (models::potedit::normalize_knot_rcut, a WARNING, as tadah hpo and
 *  tadah train replace it), plus the low-bound rules of `seed`
 *  (models::optim_low_bound_rule(key, seed): under RHOFORM NONNEG /
 *  MONOTONE AMPGRIDMB >= 0, WATOM > 0, SGRIDMB > 0 under MONOTONE and, for
 *  the Gaussian basis, under NONNEG; in every form every radial bump width
 *  > 0, SGRIDMB of a Blip/Blip5 density and SGRID2B of a D2_Blip/D2_Blip5
 *  pair term). A DM_REAM `seed` without RHOFORM throws
 *  models::DensityFormError (the migration text), as tadah hpo does. A
 *  dataset whose statistics cannot be computed (compute_data_stats: two
 *  atoms at one position, a cell too small for the cutoff, ...) throws
 *  that error, naming the structure: audited against empty statistics
 *  every corner would pass.
 *
 *  Side-effect: emits one INFO line per corner via Logger and a
 *  WARNING summarising the worst corner.
 *
 *  \return number of corners that produced any FAIL finding.
 */
struct CheckBoundsResult {
  std::size_t n_corners{0};
  std::size_t n_corners_failing{0};
  std::vector<std::size_t> failing_corner_indices{};
};

CheckBoundsResult check_bounds_at_corners(
    StructureDB&                     stdb,
    tadah::core::Context             seed,
    const std::vector<Bound>&        bounds,
    const std::vector<std::vector<double>>& corners);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_CHECK_BOUNDS_H
