#ifndef TADAH_MLIP_CHECK_BOUNDS_H
#define TADAH_MLIP_CHECK_BOUNDS_H

// `tadah analysis check_bounds`. CI-friendly dry-run.
//
// Reads an HPOTARGET file's `OPTIM <KEY> (range) <lo> <hi>` lines,
// builds a bound box, samples corners (full 2^k for k ≤ 12, otherwise
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

/** \brief One bound entry from the HPOTARGET file. Each entry covers
 *  one (key, arg_pos) tuple with explicit `(lo, hi)`. Vector entries
 *  expanded — `OPTIM CGRID2B (1-4) 0 5` becomes 4 separate Bound
 *  rows with arg_pos 0..3. */
struct Bound {
  std::string key{};
  int         arg_pos{-1};   // -1 scalar; 0..n-1 vector
  double      lo{0.0};
  double      hi{0.0};
};

/** \brief Parse an HPOTARGET file for `OPTIM <KEY> (range) <lo> <hi>`
 *  rows. Tolerant — unrecognised lines are skipped. Throws on any
 *  malformed `OPTIM` line (e.g. missing range or bounds). */
std::vector<Bound> parse_hpo_target_bounds(const std::string& path);

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
 *  bound's key (arg_pos, or the first entry for a scalar `(1)` bound) set
 *  to the corner value and every other entry kept. Several bounds on one
 *  key each set their own entry. A key the seed does not set takes the
 *  corner value alone. Throws when a bound names an entry past the seed's
 *  values (std::runtime_error) or @p corner and @p bounds differ in size
 *  (std::invalid_argument). */
tadah::core::Context corner_context(const tadah::core::Context& seed,
                                    const std::vector<Bound>& bounds,
                                    const std::vector<double>& corner);

/** \brief Run the pre-flight audit at every corner of the bound box (in the
 *  corner_context of each corner), plus the low-bound rules of `seed`
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
